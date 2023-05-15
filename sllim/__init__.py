import os
import json
import time
import openai
from openai.error import RateLimitError


def set_key(key):
    openai.api_key = key

def cache(fn):
    folder_name = "." + fn.__name__
    cache_file = os.path.join(folder_name, "cache.json")
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

@cache
def chat(messages, model="gpt-3.5-turbo"):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
        )
    except RateLimitError:
        print("Rate limit error")
        time.sleep(1)
        return chat(messages)
    except openai.APIError as e:
        if e.code == "502":
            print("API error")
            time.sleep(1)
            return chat(messages)
        else:
            raise e
    
    return response.choices[0].message.content

@cache
def complete(prompt, model="text-davinci-003", **kwargs):
    try:
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            **kwargs,
        )
    except RateLimitError:
        time.sleep(1)
        return complete(prompt, model, **kwargs)
    except openai.APIError as e:
        if e.code == 502:
            time.sleep(1)
            return complete(prompt, model, **kwargs)
        else:
            raise e
    
    return response.choices[0].text

@cache
def embed(text: str | list[str]) -> list[float] | list[list[float]]:
    if isinstance(text, list):
        text = [t.replace("\n", " ") for t in text]
        response = openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"]
        return [x["embedding"] for x in response]
    else:
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]