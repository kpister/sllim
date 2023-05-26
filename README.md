# Simple Large Language Inference Model

`sllim` serves as a quality of life wrapper around the `openai-python` library.
I found myself writing and rewriting the same helper functions with each new project I began, so now I am working to put these functions together into a easy to use library.

Nothing here is ground-breaking; everything here is opinionated.

## Usage

Use the `chat` function to connect with the `ChatCompletion.create` models. By default, it uses the `gpt-3.5-turbo` model, but you can pass a `model` param to use `gpt-4`
```
from sllim import chat

chat(
    [
        {
            "role": "system",
            "content": "Example system message",
        },
        {
            "role": "user",
            "content": "Example user message",
        }
    ]
)
```

`complete` works just like `Completion.create`, and `embed` is `Embedding.create`.


## Benefits

* Local file caching. Each of the functions is locally cached in request-response key-pairs to prevent excessive network activity.

* Auto-retry. Timeouts for rate limits, retry for internal errors (>=500 status code).

* Parameter names are in the functions so that you don't have to go looking at the docs constantly.

* Map reduce prompts onto data

* TODO: Cost estimates before running long tasks

* TODO: Describe task -> run task

* TODO: Allow easy estimate

* TODO: Allow easy logging
