import os
import json
import time
from typing import Optional

import openai
import tiktoken
from openai.error import RateLimitError


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
        key = str(args + tuple(kwargs.items()))
        if key not in cache:
            res = fn(*args, **kwargs)
            cache[key] = res
            with open(cache_file, "w") as w:
                json.dump(cache, w)
        return cache[key]

    return wrapper


def catch(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            print("Rate limit error")
            time.sleep(1)
            return fn(*args, **kwargs)
        except openai.APIError as e:
            if e.code >= 500:
                print("API error")
                time.sleep(1)
                return fn(*args, **kwargs)
            else:
                raise e

    return wrapper


@catch
@cache
def chat(
    messages,
    model="gpt-3.5-turbo",
    max_tokens: int = 16,
    temperature: float = 1,
    top_p: float = 1,
    n: int = 1,
    stop: Optional[str | list[str]] = None,
    presence_penalty: float = 0,
    frequency_penalty: float = 0,
    logit_bias: Optional[dict[int, float]] = None,
    cache_version: Optional[str] = None,
) -> str:
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        logit_bias=logit_bias,
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


def estimate(texts: list[str]):
    enc = tiktoken.encoding_for_model("gpt-4")
    total = 0

    for text in texts:
        total += len(enc.encode(text))

    return {"tokens": total, "cost": total * 0.03 / 1000}
