import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# 1. Load env / init client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY is missing; populate .env before running.")
    sys.exit(1)
client = OpenAI(api_key=api_key)


# 2. Pretend we got some metrics from a CSV or monitoring system
data = {
    "timestamp": [
        "2025-11-03T20:00Z",
        "2025-11-03T20:01Z",
        "2025-11-03T20:02Z",
        "2025-11-03T20:03Z",
        "2025-11-03T20:04Z",
    ],
    "requests": [120, 118, 121, 119, 35],
    "errors":   [0,   1,   0,   2,   40],
    "latency_ms": [110, 112, 111, 109, 400],
}

df = pd.DataFrame(data)
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# 3. Basic local (non-AI) analysis
avg_latency = df["latency_ms"].mean()
total_errors = df["errors"].sum()

logger.info("Local stats:")
logger.info("Avg latency: %.2f ms", avg_latency)
logger.info("Total errors: %d", total_errors)
logger.debug("Raw frame:\n%s", df)

if len(df) > 1:
    prior = df.iloc[:-1]
    last = df.iloc[-1]
    prior_errors_mean = prior["errors"].mean()
    prior_latency_mean = prior["latency_ms"].mean()
    if prior_errors_mean > 0 and last["errors"] > prior_errors_mean * 5:
        logger.warning(
            "Error spike detected at %s: last=%s prior_mean=%.2f",
            last["timestamp"],
            last["errors"],
            prior_errors_mean,
        )
    if prior_latency_mean > 0 and last["latency_ms"] > prior_latency_mean * 2:
        logger.warning(
            "Latency spike detected at %s: last=%s prior_mean=%.2f",
            last["timestamp"],
            last["latency_ms"],
            prior_latency_mean,
        )
else:
    last = df.iloc[-1]

summary = (
    f"avg_latency={avg_latency:.2f}, total_errors={total_errors}, "
    f"last_row={last.to_dict()}"
)
recent_rows = df.tail(3).to_string(index=False)

# 4. Ask OpenAI to interpret the pattern
prompt = f"""
You are an SRE on call.

Metrics summary:
{summary}

Recent rows:
{recent_rows}

Explain in concise bullet points:
1. What looks wrong?
2. What might have happened at 20:04Z?
3. What should the next troubleshooting step be?
"""

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a senior reliability engineer. Be concise and practical."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=250,
    )
except Exception:
    logger.exception("OpenAI call failed.")
    sys.exit(1)

choices = getattr(response, "choices", None)
if not choices:
    logger.error("OpenAI response contained no choices.")
    sys.exit(1)

choice = choices[0]
message = getattr(choice, "message", None)
content = getattr(message, "content", None) if message is not None else None
if content is None and isinstance(choice, dict):
    content = choice.get("message", {}).get("content")
if not content:
    logger.error("OpenAI choice missing content.")
    sys.exit(1)

logger.info("AI analysis:\n%s", content.strip())
