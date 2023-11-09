import glob
import json
import logging
import os
import re
import time
from contextlib import contextmanager
from itertools import zip_longest
from multiprocessing import Pool
from threading import Thread
from typing import Optional, TypeVar, TypedDict, Callable
from uuid import uuid4

import openai
import tiktoken
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types import Completion


LOCAL_CACHE = {}
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

prompt_tokens, completion_tokens = 0, 0

client = OpenAI()


def configure(api_key=None):
    client.api_key = api_key


def user(content: str) -> dict:
    return dict(role="user", content=content)


def system(content: str) -> dict:
    return dict(role="system", content=content)


def assistant(content: str) -> dict:
    return dict(role="assistant", content=content)


def load_template(filepath: str) -> str:
    with open(filepath, "r") as f:
        description, text = f.read().split("\n", 1)
        if not description.startswith("#"):
            # Not a proper prompt file
            logger.warning(
                f"File {filepath} does not start with a `# description line`."
            )
            text = description + "\n" + text
        return text.strip()


class Message(TypedDict):
    role: str
    content: str


class FunctionT(TypedDict):
    name: str
    description: str
    parameters: dict[str, dict[str, str | dict]]


Prompt = TypeVar("Prompt", str, list[Message])

API_PARAMS = dict(
    model="",
    deployment_id=None,
    max_tokens=256,
    temperature=1,
    top_p=1,
    n=1,
    stop=None,
    presence_penalty=0,
    frequency_penalty=0,
    best_of=1,
    logprobs=None,
    logit_bias=None,
)


def set_logging(level=logging.INFO):
    logging.basicConfig(level=level)
    logger.setLevel(logging.INFO)


def get_token_counts():
    global prompt_tokens, completion_tokens

    return prompt_tokens, completion_tokens


def reset_token_counts():
    global prompt_tokens, completion_tokens

    prompt_tokens, completion_tokens = 0, 0


def try_make(folder_name):
    try:
        os.makedirs(folder_name, exist_ok=True)
    except Exception:
        logger.info(f"Cannot create folder {folder_name}")


class fake_file:
    def write(self, *args, **kwargs):
        pass

    def close(self, *args, **kwargs):
        pass

    def read(self, *args, **kwargs):
        return "{}"


@contextmanager
def try_open(filename, mode="r"):
    try:
        f = open(filename, mode, encoding="utf-8")
    except Exception:
        f = fake_file()

    try:
        yield f

    finally:
        f.close()


def cache(fn):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{fn.__name__}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    # Load from cache
    with try_open(cache_file) as w:
        cache = json.load(w)

    def wrapper(*args, **kwargs):
        if kwargs.get("nocache", False):
            kwargs.pop("nocache", None)
            return fn(*args, **kwargs)

        key = str(args + tuple(kwargs.items()))
        if key not in cache:
            res = fn(*args, **kwargs)
            cache[key] = res
            with try_open(cache_file, "w") as w:
                json.dump(cache, w)
        return cache[key]

    return wrapper


def catch(fn):
    backoff = {0: 0}

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except openai.RateLimitError as e:
            logger.info("Rate limit error")
            backoff[0] += 1
            if backoff[0] > 9:
                raise e
            time.sleep(min(2 ** backoff[0], 60))
            return wrapper(*args, **kwargs)
        except openai.APIStatusError as e:
            logger.info("API Error: " + str(e))
            backoff[0] += 1
            if e.code and e.code >= 500 and backoff[0] <= 9:
                time.sleep(min(2 ** backoff[0], 60))
                return wrapper(*args, **kwargs)
            else:
                raise e
        finally:
            backoff[0] = 0

    return wrapper


@catch
@cache
def chat(
    messages,
    model="gpt-3.5-turbo",
    max_tokens: Optional[int] = None,
    functions: Optional[list[FunctionT]] = None,
    temperature: float = 1,
    top_p: float = 1,
    n: int = 1,
    stop: Optional[str | list[str]] = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    logit_bias: Optional[dict[int, float]] = None,
    deployment_id: Optional[str] = None,
    cache_version: Optional[str] = None,
) -> str:
    global prompt_tokens, completion_tokens

    if deployment_id:
        model = None

    default_params = {
        "model": None,
        "max_tokens": None,
        "temperature": 1,
        "top_p": 1,
        "n": 1,
        "stop": None,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "logit_bias": None,
        "deployment_id": None,
    }
    kwargs = {
        k: v
        for k, v in locals().items()
        if k in default_params and v != default_params[k]
    }
    max_tokens_str = "infinity" if max_tokens is None else str(max_tokens)
    model_str = model if model else deployment_id
    logger.info(
        f"Calling {model_str} using at most {max_tokens_str} with messages: {messages}"
    )

    response: ChatCompletion = client.chat.completions.create(
        messages=messages,
        **kwargs,
    )

    message = response.choices[0].message
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

    if message.content:
        logger.info(f"Response: {message.content}")
        return message.content

    raise Exception("No content found in response.")


@catch
@cache
def call(
    messages,
    model="gpt-3.5-turbo",
    max_tokens: int = 256,
    functions: Optional[list[FunctionT]] = None,
    temperature: float = 1,
    top_p: float = 1,
    n: int = 1,
    stop: Optional[str | list[str]] = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    logit_bias: Optional[dict[int, float]] = None,
    cache_version: Optional[str] = None,
) -> str:
    """Generate a function call based on the messages. Use `create_function_call` to prepare `functions` arg."""
    global prompt_tokens, completion_tokens

    default_params = {
        "temperature": 1,
        "top_p": 1,
        "n": 1,
        "functions": None,
        "stop": None,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "logit_bias": None,
    }
    kwargs = {
        k: v
        for k, v in locals().items()
        if k in default_params and v != default_params[k]
    }

    response: ChatCompletion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        **kwargs,
    )
    message = response.choices[0].message
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

    if message.function_call:
        return message.function_call

    raise Exception("No function call found in response. %s" % str(message))


@catch
@cache
def complete(
    prompt: str = "<|endoftext|>",
    model: str = "text-davinci-003",
    max_tokens: int = 16,
    temperature: float = 1,
    top_p: float = 1,
    n: int = 1,
    logprobs: Optional[int] = None,
    stop: Optional[str | list[str]] = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    best_of: int = 1,
    logit_bias: Optional[dict[int, float]] = None,
    cache_version: Optional[str] = None,
) -> str:
    global prompt_tokens, completion_tokens

    default_params = {
        "prompt": "<|endoftext|>",
        "temperature": 1,
        "top_p": 1,
        "n": 1,
        "max_tokens": 16,
        "logprobs": None,
        "stop": None,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "best_of": 1,
        "logit_bias": None,
    }
    kwargs = {
        k: v
        for k, v in locals().items()
        if k in default_params and v != default_params[k]
    }

    response: Completion = client.completions.create(
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )
    message = response.choices[0].text
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

    return message


@catch
@cache
def embed(
    text: str | list[str], engine="text-embedding-ada-002", deployment_id: str = None
) -> list[float] | list[list[float]]:
    if deployment_id:
        engine = None
    kwargs = {
        k: v
        for k, v in locals().items()
        if k in ["engine", "deployment_id"] and v is not None
    }
    if isinstance(text, list):
        text = [t.replace("\n", " ") for t in text]
        response = client.embeddings.create(input=text, **kwargs)["data"]
        return [x["embedding"] for x in response]
    else:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], **kwargs)["data"][0]["embedding"]


def estimate(messages_or_prompt: Prompt, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    total = 0

    if isinstance(messages_or_prompt, str):
        total = len(enc.encode(messages_or_prompt))
    else:
        for text in messages_or_prompt:
            total += len(enc.encode(text["content"]))

    model_cost = {
        "gpt-3.5-turbo": 0.002,
        "gpt-4": 0.03,
        "text-davinci-003": 0.02,
        "text-davinci-002": 0.012,
    }
    return {"tokens": total, "cost": total * model_cost.get(model) / 1000}


def load_cache(base, t):
    global LOCAL_CACHE

    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    # Load from cache
    with try_open(cache_file) as w:
        cache = json.load(w)

    LOCAL_CACHE = cache.get(t, {})


def check_cache(base, t, c):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    # Load from cache
    with try_open(cache_file) as w:
        cache = json.load(w)

    if t in cache and c in cache[t]:
        return cache[t][c]

    return None


def save_tmp_cache(base, t, c, result):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.{str(uuid4())}.tmp.json")

    if not os.path.exists(folder_name):
        try_make(folder_name)

    with try_open(cache_file, "w") as w:
        json.dump({t: {c: result}}, w)


def mp_call(args):
    # 1. check true cache for t&c
    c = args[0][1]
    if c in LOCAL_CACHE:
        return LOCAL_CACHE[c]

    # 2. if not in true cache, compute
    fn = args[1]["__function"]
    fns = {"complete": complete, "chat": chat}
    result = fns.get(fn)(
        args[0][0],
        **{k: v for k, v in args[1].items() if not k.startswith("__")},
        nocache=True,
    )
    # 3. add to tmp cache
    t = json.dumps(args[1]["__template"])
    save_tmp_cache(fn, t, c, result)
    return result


def to_slices(template: Prompt, iters, constants):
    for slices in zip_longest(*iters, fillvalue=constants):
        key_values = {k: v for d in slices for k, v in d.items()}
        yield (format_prompt(template, **key_values), str(key_values))


def format_prompt(template: Prompt, **kwargs):
    if isinstance(template, str):
        return format(template, **kwargs)
    else:
        return [
            {
                "role": message["role"],
                "content": format(message["content"], **kwargs),
            }
            for message in template
        ]


def map_reduce(template: Prompt, n=8, **kwargs):
    params = {}
    for key, value in kwargs.items():
        if key in API_PARAMS:
            params[key] = value
    params["__template"] = template

    if isinstance(template, str):
        params["__function"] = "complete"
    else:
        params["__function"] = "chat"

    # Create an list of slices for each template
    iters = [[]]
    constants = {}
    max_len = 0
    for key, value in kwargs.items():
        if hasattr(value, "__iter__") and not isinstance(value, str):
            iters.append([{key: v} for v in value])
            max_len = max(max_len, len(value))
        else:
            constants[key] = value

    iterator = to_slices(template, iters, constants)

    try:
        with Pool(
            processes=min(n, max_len),
            initializer=load_cache,
            initargs=(params["__function"], json.dumps(params["__template"])),
        ) as pool:
            for res in pool.imap(mp_call, zip_longest(iterator, [], fillvalue=params)):
                yield res
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    except Exception as e:
        import traceback

        traceback.print_exc()
        print("EXCEPTION!", e)
    finally:
        collate_caches(params["__function"])


def thread_map_reduce(template: Prompt, n=8, **kwargs):
    params = {}
    for key, value in kwargs.items():
        if key in API_PARAMS:
            params[key] = value

    if isinstance(template, str):
        fn = complete
    else:
        fn = chat

    # Create an list of slices for each template
    iters = [[]]
    constants = {}
    max_len = 0
    for key, value in kwargs.items():
        if hasattr(value, "__iter__") and not isinstance(value, str):
            iters.append([{key: v} for v in value])
            max_len = max(max_len, len(value))
        else:
            constants[key] = value

    iterator = to_slices(template, iters, constants)

    results = ["" for _ in range(max_len)]

    def thread_call(message, idx):
        try:
            results[idx] = fn(
                message, **{k: v for k, v in params.items()}, nocache=True
            )
        except openai.RateLimitError:
            results[idx] = ""

    num_threads = min(n, max_len)
    threads = []
    for idx, (msg, _) in enumerate(iterator):
        threads.append(Thread(target=thread_call, args=(msg, idx)))
        threads[-1].start()

        if len(threads) >= num_threads:
            for thread in threads:
                thread.join()
            threads = []

    for thread in threads:
        thread.join()

    return results


def collate_caches(function_name):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{function_name}.json")
    if not os.path.exists(cache_file):
        try_make(folder_name)

    with try_open(cache_file) as w:
        cache = json.load(w)

    for f in glob.glob(os.path.join(folder_name, f"{function_name}.*.tmp.json")):
        with try_open(f) as w:
            tmp_cache = json.load(w)
            key = list(tmp_cache.keys())[0]
            if key not in cache:
                cache[key] = {}
            cache[key].update(tmp_cache[key])
        os.remove(f)

    with try_open(cache_file, "w") as w:
        json.dump(cache, w)


def to_type_name(_type: str):
    if "list" in _type or "tuple" in _type:
        return "array", {"items": {"type": "string"}}

    return {
        "str": "string",
        "int": "number",
    }.get(_type, _type), {}


def parse_doc(doc: str):
    if not doc:
        return "", {}, []

    lines = doc.split(":param:")
    fn_description = lines[0].strip()
    properties = {}
    required = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue

        parts = line.split(":")
        name = parts[0].strip()
        _type = parts[1].strip()
        _type = re.sub(r"Optional\[(.*)\]", r"\1", _type)
        description = ":".join(parts[2:]).strip()
        if name.startswith("*"):
            name = name[1:]
            required.append(name)

        type_name, type_items = to_type_name(_type)
        properties[name] = {
            "type": type_name,
            "description": description,
        }
        if type_items:
            properties[name].update(type_items)

    return fn_description, properties, required


def create_function_call(fn: Callable) -> FunctionT:
    description, properties, required = parse_doc(fn.__doc__)
    return {
        "name": fn.__name__,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def format(s: str, **kwargs):
    for key, value in kwargs.items():
        s = s.replace("{" + key + "}", str(value))
    return s
