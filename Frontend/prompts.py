import json
import re
import httpx
import config
from dotenv import load_dotenv
import os
from pathlib import Path

class BackendError(Exception):
    """An LLM backend is unreachable or returned an unusable response."""

    def __init__(self, backend, url, message):
        self.backend = backend
        self.url = url
        super().__init__(f"{backend} backend at {url}: {message}")

def check_backends():
    """Ping both vLLM servers. Returns {"summarizer": bool, "stance": bool}; never raises."""
    status = {}

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('USAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    for name, base_url in (
        ("summarizer", "https://api.doc.usai.gov/api/v1/models"),
        ("stance", "https://api.doc.usai.gov/api/v1/models"),
    ):
        try:
            resp = httpx.get(url=base_url,
                             headers=headers)
            status[name] = resp.status_code == 200
        except httpx.HTTPError:
            status[name] = False
    return status


def ai_summarize(tweets):
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('USAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config.SUMMARIZER_MODEL,
        "messages": [
            {"role": "system", "content": Path("../prompts/ai_summary_prompt.txt").read_text()},
            {"role": "user", "content": tweets}
        ]
    }

    response = httpx.post(url=config.SUMMARIZER_BASE_URL,
                            headers=headers,
                            json=payload,
                            timeout=None)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    return content


def parse_stance_response(text, start, end):
    """Map tweet ids in [start, end) to int stances from a model response.

    Decodes the first JSON object in the text (tolerating prose before and
    junk after it); any id that is missing or has a non-integer value gets -1.
    """
    parsed = {}
    idx = text.find("{")
    if idx != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text[idx:])
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    results = {}
    for j in range(start, end):
        raw = parsed.get(f"tweet-{j}", -1)
        try:
            results[j] = int(raw)
        except (TypeError, ValueError):
            results[j] = -1
    return results


async def stance_annotation(tweets, topic, stances, examples):
    formattedStances = []
    examplePrompt = "### Examples: \ninput:\n"
    for i in range(len(stances)):
        if stances[i] != "":
            currStanceNum = len(formattedStances)
            formattedStances.append(f"{currStanceNum}: {stances[i]}")

    exampleCount = 0
    for key in examples.keys():
        examplePrompt += f"{exampleCount}-{key}\n"
        exampleCount += 1
    examplePrompt += "output:\n{\n"
    exampleCount = 0
    for key in examples.keys():
        examplePrompt += f'"tweet-{exampleCount}": {examples[key]},\n'
        exampleCount += 1
    examplePrompt += "}"
    if len(examples) == 0:
        examplePrompt = ""

    AI_STANCE_PROMPT = f"""You are a human annotator. You will be presented with
    a list of tweets (labeled with id numbers), delimited by triple backticks,
    concerning '{topic}'. Please make the following assessment without further 
    commentary:"""
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {os.getenv('USAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": config.STANCE_MODEL,
        #"model": "gpt-5.5-latest-guardrails-defaultv2",
        "messages": [
            {"role": "system", "content": AI_STANCE_PROMPT},
            {"role": "user", "content": f"""Determine whether each tweet discusses the topic of '{topic}'. If it does, indicate the stance of the Twitter user who posted the tweet as one of '{formattedStances}', otherwise label the stance as -1. Your response should be in JSON format as shown below, do not provide any other output:
    {{
        "tweet-<tweetID>" : "stance_number"
    }}

    The stance number must be between -1 and {len(formattedStances)-1}, no other text should be in the stance number field.

    {examplePrompt}

    ### Your Task:
    Tweets: ```{tweets}```
    """}
        ]
    }

    response = httpx.post(url=config.STANCE_BASE_URL,
                          #url="https://api.doc.usai.gov/api/v1/chat/completions",
                          headers=headers,
                          json=payload,
                          timeout=None)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    return content
