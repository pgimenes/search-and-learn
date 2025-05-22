import os
from openai import OpenAI, AsyncOpenAI
from os import getenv
import numpy as np
import asyncio

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

default_client = OpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key="EMPTY",
)
default_async_client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1", 
    api_key="EMPTY",
)

def completions_with_backoff(**kwargs):
    if "client" in kwargs and kwargs["client"] is not None:
        client = kwargs["client"]
    else:
        client = default_client

    # remove client from kwargs
    _ = kwargs.pop("client", None)

    return client.chat.completions.create(**kwargs)

def llm(
    prompt,
    model="", 
    temperature=1, 
    max_tokens=10000, 
    n=1, 
    stop=None, 
    client=None,
) -> list:
    if not isinstance(prompt, list):
        prompt = [prompt]

    outputs = []
    for p in prompt:
        messages = [{"role": "user", "content": p}]    

        res = completions_with_backoff(
            model=model, 
            messages=messages, 
            temperature=temperature, 
            max_tokens=max_tokens, 
            stop=stop,
            client=client,
        )

        outputs.append(res)

    return outputs

async def async_llm(
    prompt,
    model="", 
    temperature=1, 
    max_tokens=1000, 
    n=1, 
    stop=None, 
    client=None,
) -> list:
    
    
    if not isinstance(prompt, list):
        prompt = [prompt]
    
    if client is None:
        client = default_async_client

    jobs = []
    for p in prompt:
        messages = [{"role": "user", "content": p}]    
        
        # launch asynchronously
        jobs.append(
            client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=temperature, 
                max_tokens=max_tokens, 
                stop=stop,
            )
        )


    return await asyncio.gather(*jobs)