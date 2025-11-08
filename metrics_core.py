import logging
import os
import time
from typing import Any, Dict, Iterable

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_DATA: Dict[str, list] = {
    "timestamp": [
        "2025-11-03T20:00Z",
        "2025-11-03T20:01Z",
        "2025-11-03T20:02Z",
        "2025-11-03T20:03Z",
        "2025-11-03T20:04Z",
    ],
    "requests": [120, 118, 121, 119, 35],
    "errors": [0, 1, 0, 2, 40],
    "latency_ms": [110, 112, 111, 109, 400],
}


def init_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing; populate .env before running.")
    return OpenAI(api_key=api_key)


def build_dataframe(data: Dict[str, Iterable[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("One or more timestamps could not be parsed.")
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if not {"errors", "latency_ms"}.issubset(df.columns):
        raise ValueError("Data must include 'errors' and 'latency_ms' columns.")

    avg_latency = float(df["latency_ms"].mean())
    total_errors = int(df["errors"].sum())
    last_row = df.iloc[-1].copy()
    if "timestamp" in last_row:
        last_row["timestamp"] = str(last_row["timestamp"])

    warnings: list[str] = []
    if len(df) > 1:
        prior = df.iloc[:-1]
        prior_errors_mean = prior["errors"].mean()
        prior_latency_mean = prior["latency_ms"].mean()
        if prior_errors_mean > 0 and last_row["errors"] > prior_errors_mean * 5:
            warnings.append(
                "Error spike detected at {timestamp}: last={last} prior_mean={mean:.2f}".format(
                    timestamp=last_row.get("timestamp", "latest"),
                    last=last_row["errors"],
                    mean=prior_errors_mean,
                )
            )
        if prior_latency_mean > 0 and last_row["latency_ms"] > prior_latency_mean * 2:
            warnings.append(
                "Latency spike detected at {timestamp}: last={last} prior_mean={mean:.2f}".format(
                    timestamp=last_row.get("timestamp", "latest"),
                    last=last_row["latency_ms"],
                    mean=prior_latency_mean,
                )
            )

    summary = (
        f"avg_latency={avg_latency:.2f}, total_errors={total_errors}, last_row={dict(last_row)}"
    )
    recent_rows = df.tail(3).to_string(index=False)

    return {
        "avg_latency": avg_latency,
        "total_errors": total_errors,
        "warnings": warnings,
        "summary": summary,
        "recent_rows": recent_rows,
        "row_count": len(df),
        "last_row": dict(last_row),
    }


def build_prompt(metrics: Dict[str, Any]) -> str:
    warnings = metrics.get("warnings") or []
    warnings_text = "\n".join(warnings) if warnings else "No local anomalies detected."

    return f"""
You are an SRE on call.

Metrics summary:
{metrics['summary']}

Local anomaly warnings:
{warnings_text}

Recent rows:
{metrics['recent_rows']}

Explain in concise bullet points:
1. What looks wrong?
2. What might have happened at the most recent timestamp?
3. What should the next troubleshooting step be?
"""


def call_openai(
    client: OpenAI,
    prompt: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 250,
    retries: int = 1,
    retry_delay: float = 1.5,
) -> str:
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a senior reliability engineer. Be concise and practical."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except Exception as exc:
            logger.exception("OpenAI call failed on attempt %s.", attempt + 1)
            if attempt >= retries:
                raise RuntimeError(f"OpenAI call failed: {exc}") from exc
            attempt += 1
            time.sleep(retry_delay)

    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("OpenAI response contained no choices.")

    choice = choices[0]
    message = getattr(choice, "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(choice, dict):
        content = choice.get("message", {}).get("content")
    if not content:
        raise RuntimeError("OpenAI response missing content.")

    return content.strip()


def analyse_telemetry(data: Dict[str, Iterable[Any]], client: OpenAI | None = None) -> Dict[str, Any]:
    df = build_dataframe(data)
    metrics = compute_metrics(df)
    prompt = build_prompt(metrics)
    active_client = client or init_client()
    ai_text = call_openai(active_client, prompt)
    return {
        "metrics": metrics,
        "analysis": ai_text,
        "prompt": prompt,
    }
