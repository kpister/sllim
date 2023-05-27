import glob
import json
import logging
import os
import time
from itertools import zip_longest
from multiprocessing import Pool
from typing import Optional, TypeVar, TypedDict
from uuid import uuid4

import openai
import tiktoken
from openai.error import RateLimitError


LOCAL_CACHE = {}


class Message(TypedDict):
    role: str
    content: str


Prompt = TypeVar("Prompt", str, list[Message])

API_PARAMS = dict(
    model="",
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


def set_key(key):
    openai.api_key = key


def cache(fn):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{fn.__name__}.json")
    if not os.path.exists(cache_file):
        os.makedirs(folder_name, exist_ok=True)
        with open(cache_file, "w") as w:
            w.write("{}")

    # Load from cache
    with open(cache_file) as w:
        cache = json.load(w)

    def wrapper(*args, **kwargs):
        if kwargs.get("cache", True):
            return fn(*args, **kwargs)

        kwargs.pop("cache")

        key = str(args + tuple(kwargs.items()))
        if key not in cache:
            res = fn(*args, **kwargs)
            cache[key] = res
            with open(cache_file, "w") as w:
                json.dump(cache, w)
        return cache[key]

    return wrapper


def catch(fn):
    backoff = {0: 0}

    def wrapper(*args, **kwargs):
        try:
            backoff[0] = 0
            return fn(*args, **kwargs)
        except RateLimitError:
            logging.info("Rate limit error")
            time.sleep(2 ** backoff[0])
            return fn(*args, **kwargs)
        except openai.APIError as e:
            print(e)
            if e.code and e.code >= 500:
                logging.info("API Error")
                time.sleep(2 ** backoff[0])
                return fn(*args, **kwargs)
            else:
                raise e

    return wrapper


@catch
@cache
def chat(
    messages,
    model="gpt-3.5-turbo",
    max_tokens: int = 256,
    temperature: float = 1,
    top_p: float = 1,
    n: int = 1,
    stop: Optional[str | list[str]] = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    logit_bias: Optional[dict[int, float]] = None,
    cache_version: Optional[str] = None,
) -> str:
    default_params = {
        "temperature": 1,
        "top_p": 1,
        "n": 1,
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
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        **kwargs,
    )

    return response.choices[0].message.content


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
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        logprobs=logprobs,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        best_of=best_of,
        logit_bias=logit_bias,
    )

    return response.choices[0].text


@catch
@cache
def embed(text: str | list[str]) -> list[float] | list[list[float]]:
    embed_model = "text-embedding-ada-002"
    if isinstance(text, list):
        text = [t.replace("\n", " ") for t in text]
        response = openai.Embedding.create(input=text, model=embed_model)["data"]
        return [x["embedding"] for x in response]
    else:
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=embed_model)["data"][0][
            "embedding"
        ]


def estimate(messages_or_prompt: Prompt, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    total = 0

    if isinstance(messages_or_prompt, str):
        total = len(enc.encode(messages_or_prompt))
    else:
        for text in messages_or_prompt:
            total += len(enc.encode(text["content"]))

    model_cost = {"gpt-3.5-turbo": 0.002, "gpt-4": 0.03, "text-davinci-003": 0.02}
    return {"tokens": total, "cost": total * model_cost.get(model) / 1000}


def load_cache(base, t):
    global LOCAL_CACHE

    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.json")
    if not os.path.exists(cache_file):
        os.makedirs(folder_name, exist_ok=True)
        with open(cache_file, "w") as w:
            w.write("{}")

    # Load from cache
    with open(cache_file) as w:
        cache = json.load(w)

    LOCAL_CACHE = cache.get(t, {})


def check_cache(base, t, c):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.json")
    if not os.path.exists(cache_file):
        os.makedirs(folder_name, exist_ok=True)
        with open(cache_file, "w") as w:
            w.write("{}")

    # Load from cache
    with open(cache_file) as w:
        cache = json.load(w)

    if t in cache and c in cache[t]:
        return cache[t][c]

    return None


def save_tmp_cache(base, t, c, result):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{base}.{str(uuid4())}.tmp.json")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)

    with open(cache_file, "w", encoding="utf-8") as w:
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
        cache=False,
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
        return template.format(
            **{
                key: value
                for key, value in kwargs.items()
                if "{" + key + "}" in template
            }
        )
    else:
        return [
            {
                "role": message["role"],
                "content": message["content"].format(
                    **{
                        key: value
                        for key, value in kwargs.items()
                        if "{" + key + "}" in message["content"]
                    }
                ),
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
    for key, value in kwargs.items():
        if hasattr(value, "__iter__") and not isinstance(value, str):
            iters.append([{key: v} for v in value])
        else:
            constants[key] = value

    iterator = to_slices(template, iters, constants)

    try:
        with Pool(
            processes=n,
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
        pool.terminate()
        pool.join()
        print("EXCEPTION!", e)
    finally:
        collate_caches(params["__function"])


def collate_caches(function_name):
    folder_name = ".cache"
    cache_file = os.path.join(folder_name, f"{function_name}.json")
    if not os.path.exists(cache_file):
        os.makedirs(folder_name, exist_ok=True)
        with open(cache_file, "w") as w:
            w.write("{}")

    with open(cache_file) as w:
        cache = json.load(w)

    for f in glob.glob(os.path.join(folder_name, f"{function_name}.*.tmp.json")):
        with open(f, encoding="utf-8") as w:
            tmp_cache = json.load(w)
            key = list(tmp_cache.keys())[0]
            if key not in cache:
                cache[key] = {}
            cache[key].update(tmp_cache[key])
        os.remove(f)

    with open(cache_file, "w", encoding="utf-8") as w:
        json.dump(cache, w)
