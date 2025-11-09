import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from dotenv import load_dotenv

from azure_runtime import ensure_venv_site_packages_precedence

ensure_venv_site_packages_precedence()

from openai import OpenAI

logger = logging.getLogger(__name__)
API_KEY_RE = re.compile(r"sk-[A-Za-z0-9]{10,}")


def _sanitize_error_message(message: str) -> str:
    return API_KEY_RE.sub("sk-****", message)

COLUMN_ALIASES: Dict[str, list[str]] = {
    "latency_ms": [
        "latency_ms",
        "latency",
        "latencies",
        "response_time",
        "response_times",
        "response_time_ms",
        "response_ms",
        "response_duration",
        "response_durations",
        "duration_ms",
        "duration",
        "durationms",
    ],
    "errors": [
        "errors",
        "error_count",
        "error_counts",
        "failures",
        "failed_requests",
        "error_total",
        "error_totals",
    ],
    "status_code": [
        "status_code",
        "status",
        "response_code",
        "code",
        "http_status",
    ],
    "region": [
        "region",
        "regions",
        "api_region",
        "geo_region",
        "server_region",
        "datacenter",
    ],
}


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


ALIAS_LOOKUP: Dict[str, str] = {}
for canonical, aliases in COLUMN_ALIASES.items():
    for alias in aliases:
        ALIAS_LOOKUP[_normalize_column_name(alias)] = canonical

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
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing; populate .env before running.")
    return OpenAI(api_key=api_key)


def build_dataframe(data: Dict[str, Iterable[Any]]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    rename_map: Dict[str, str] = {}
    existing = set(df.columns)
    for column in df.columns:
        normalized = _normalize_column_name(column)
        canonical = ALIAS_LOOKUP.get(normalized)
        if not canonical:
            continue
        if column == canonical:
            continue
        if canonical in existing:
            continue
        rename_map[column] = canonical
        existing.add(canonical)
    if rename_map:
        df = df.rename(columns=rename_map)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("One or more timestamps could not be parsed.")
    return df


def compute_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    last_row = df.iloc[-1].copy()
    if "timestamp" in last_row:
        last_row["timestamp"] = str(last_row["timestamp"])

    avg_latency: Optional[float] = None
    total_errors: Optional[int] = None
    warnings: list[str] = []

    latency_series: Optional[pd.Series] = None
    errors_series: Optional[pd.Series] = None
    status_series: Optional[pd.Series] = None

    if "latency_ms" in df.columns:
        latency_series = _coerce_numeric(df["latency_ms"])
        avg_latency = float(latency_series.mean())
        last_row["latency_ms"] = float(latency_series.iloc[-1])

    if "errors" in df.columns:
        errors_series = _coerce_numeric(df["errors"]).fillna(0)
        total_errors = int(errors_series.sum())
        last_row["errors"] = float(errors_series.iloc[-1])

    if "status_code" in df.columns:
        status_series = _coerce_numeric(df["status_code"])

    if total_errors is None and status_series is not None:
        error_mask = ((status_series < 200) | (status_series >= 300)) & status_series.notna()
        total_errors = int(error_mask.sum())
        if not status_series.empty:
            last_row["inferred_error_flag"] = int(error_mask.iloc[-1])

    if len(df) > 1:
        if errors_series is not None:
            prior_errors_mean = errors_series.iloc[:-1].mean()
            last_errors = errors_series.iloc[-1]
            if prior_errors_mean > 0 and last_errors > prior_errors_mean * 5:
                warnings.append(
                    "Error spike detected at {timestamp}: last={last} prior_mean={mean:.2f}".format(
                        timestamp=last_row.get("timestamp", "latest"),
                        last=last_errors,
                        mean=prior_errors_mean,
                    )
                )
        elif status_series is not None:
            prior_status = status_series.iloc[:-1]
            if not prior_status.empty:
                prior_error_rate = (
                    ((prior_status < 200) | (prior_status >= 300)) & prior_status.notna()
                ).mean()
                last_status = status_series.iloc[-1]
                if prior_error_rate > 0 and not pd.isna(last_status):
                    last_flag = int(last_status < 200 or last_status >= 300)
                    if last_flag > prior_error_rate * 5:
                        warnings.append(
                            "Error spike detected (status) at {timestamp}: last={last} prior_rate={mean:.2f}".format(
                                timestamp=last_row.get("timestamp", "latest"),
                                last=last_flag,
                                mean=prior_error_rate,
                            )
                        )
        if latency_series is not None:
            prior_latency_mean = latency_series.iloc[:-1].mean()
            last_latency = latency_series.iloc[-1]
            if prior_latency_mean > 0 and last_latency > prior_latency_mean * 2:
                warnings.append(
                    "Latency spike detected at {timestamp}: last={last} prior_mean={mean:.2f}".format(
                        timestamp=last_row.get("timestamp", "latest"),
                        last=last_latency,
                        mean=prior_latency_mean,
                    )
                )

    summary_parts = []
    if avg_latency is not None:
        summary_parts.append(f"avg_latency={avg_latency:.2f}")
    if total_errors is not None:
        summary_parts.append(f"total_errors={total_errors}")
    summary_parts.append(f"last_row={dict(last_row)}")
    if avg_latency is None:
        summary_parts.append("latency column missing")
    if total_errors is None:
        summary_parts.append("errors column missing")

    recent_rows = df.tail(3).to_string(index=False)
    charts = _build_charts(df)

    return {
        "avg_latency": avg_latency,
        "total_errors": total_errors,
        "warnings": warnings,
        "summary": ", ".join(summary_parts),
        "recent_rows": recent_rows,
        "row_count": len(df),
        "last_row": dict(last_row),
        "missing_latency": avg_latency is None,
        "missing_errors": total_errors is None,
        "top_api_response": _top_value(df, ["status_code", "status", "response_code"]),
        "top_error_cause": _top_value(df, ["error_cause", "error_type", "error_reason"]),
        "top_error_message": _top_value(df, ["error_message", "error_msg", "message", "error"]),
        "top_client_ip": _top_value(df, ["client_ip", "ip", "source_ip"]),
        "charts": charts,
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


def _top_value(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[Dict[str, Any]]:
    for column in candidates:
        if column in df.columns:
            series = df[column].dropna()
            if series.empty:
                continue
            counts = series.value_counts()
            value = counts.index[0]
            count = int(counts.iloc[0])
            return {"column": column, "value": str(value), "count": count}
    return None




def _coerce_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    digits = series.astype(str).str.extract(r"(\d+)")
    return pd.to_numeric(digits[0], errors="coerce")


def _build_charts(df: pd.DataFrame) -> List[Dict[str, Any]]:
    charts: List[Dict[str, Any]] = []
    charts.extend(_build_timeseries_charts(df))

    response_chart = _build_category_chart(
        df,
        ["status_code", "status", "response_code", "code", "http_status"],
        "API Responses",
    )
    if response_chart:
        charts.append(response_chart)

    region_chart = _build_category_chart(
        df,
        ["region", "api_region", "geo_region", "server_region", "datacenter"],
        "API Regions",
    )
    if region_chart:
        charts.append(region_chart)

    error_chart = _build_category_chart(
        df,
        ["error_message", "error_msg", "error", "message"],
        "Top Errors",
    )
    if error_chart:
        charts.append(error_chart)

    latency_chart = _build_numeric_histogram(df.get("latency_ms"), "Latency / Response Time (ms)")
    if latency_chart:
        charts.append(latency_chart)

    return charts


def _build_timeseries_charts(df: pd.DataFrame, bucket_count: int = 10) -> List[Dict[str, Any]]:
    if "timestamp" not in df.columns:
        return []
    working = df.copy()
    working["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    working = working.dropna(subset=["timestamp"])
    if working.empty:
        return []
    start = working["timestamp"].min()
    end = working["timestamp"].max()
    if pd.isna(start) or pd.isna(end):
        return []
    if start == end:
        working["bucket_label"] = start.isoformat()
    else:
        bucket_count = max(1, bucket_count)
        edges = [start + (end - start) * (i / bucket_count) for i in range(bucket_count + 1)]
        edges[-1] = end
        labels = [
            f"{edges[i].isoformat()} - {edges[i + 1].isoformat()}"
            for i in range(len(edges) - 1)
        ]
        working["bucket_label"] = pd.cut(
            working["timestamp"],
            bins=edges,
            include_lowest=True,
            right=False,
            labels=labels,
        ).astype(str)
    numeric_columns = working.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in {"bucket_label"}]
    if not numeric_columns:
        return []
    grouped = working.groupby("bucket_label")
    charts: List[Dict[str, Any]] = []
    for column in numeric_columns:
        hourly = grouped[column].mean().dropna()
        if hourly.empty:
            continue
        points = [{"timestamp": bucket, "value": float(value)} for bucket, value in hourly.items()]
        charts.append({
            "type": "timeseries",
            "title": f"{column} over time",
            "points": points,
        })
    return charts


def _build_category_chart(
    df: pd.DataFrame,
    candidates: Sequence[str],
    title: str,
    limit: int = 10,
) -> Optional[Dict[str, Any]]:
    for column in candidates:
        if column in df.columns:
            series = df[column].dropna()
            if series.empty:
                continue
            counts = series.astype(str).value_counts().head(limit)
            labels = counts.index.tolist()
            values = [int(v) for v in counts.values]
            return {
                "type": "category",
                "title": title,
                "labels": labels,
                "values": values,
            }
    return None


def _build_numeric_histogram(series: Optional[pd.Series], title: str, bins: int = 10) -> Optional[Dict[str, Any]]:
    if series is None:
        return None
    numeric = _coerce_numeric(series).dropna()
    if numeric.empty:
        return None
    if numeric.min() == numeric.max():
        labels = [f"{numeric.min():.2f}"]
        values = [len(numeric)]
    else:
        bucket_labels = pd.cut(numeric, bins=bins, include_lowest=True)
        counts = bucket_labels.value_counts().sort_index()
        labels = [str(interval) for interval in counts.index]
        values = [int(v) for v in counts.values]
    return {
        "type": "category",
        "title": title,
        "labels": labels,
        "values": values,
    }

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
            sanitized = _sanitize_error_message(str(exc))
            logger.exception("OpenAI call failed on attempt %s: %s", attempt + 1, sanitized)
            if attempt >= retries:
                raise RuntimeError(f"OpenAI call failed: {sanitized}") from exc
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
