import io
import html
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs

from email.parser import BytesParser
from email.policy import default as default_policy

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
VENV_SITE = APP_DIR / "antenv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

from metrics_core import DEFAULT_DATA, build_dataframe, build_prompt, call_openai, compute_metrics, init_client

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CLIENT = None
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))  # 5 MB default
MAX_TEXT_BYTES = int(os.getenv("MAX_TEXT_BYTES", 2 * 1024 * 1024))  # textarea guard
PREVIEW_MAX_ROWS = int(os.getenv("PREVIEW_MAX_ROWS", "200"))
PREVIEW_MAX_COLUMNS = int(os.getenv("PREVIEW_MAX_COLUMNS", "25"))


def ensure_client() -> None:
    global CLIENT
    if CLIENT is None:
        CLIENT = init_client()


def _read_exact(stream, size: int) -> bytes:
    if size < 0:
        raise ValueError("Invalid Content-Length header.")
    if size == 0:
        return b""
    remaining = size
    chunks: list[bytes] = []
    while remaining:
        chunk = stream.read(min(remaining, 64 * 1024))
        if not chunk:
            raise ValueError("Client closed the connection before sending the full body.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _parse_multipart_form(body: bytes, content_type: str) -> tuple[dict[str, list[str]], Optional[dict]]:
    if not content_type:
        raise ValueError("Content-Type header missing for multipart submission.")
    header = f"Content-Type: {content_type}\n\n".encode("utf-8")
    parser = BytesParser(policy=default_policy)
    try:
        message = parser.parsebytes(header + body)
    except Exception as exc:
        raise ValueError(f"Unable to parse multipart payload: {exc}") from exc
    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data payload.")
    fields: dict[str, list[str]] = {}
    uploaded_file: Optional[dict] = None
    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            uploaded_file = {
                "field_name": name,
                "filename": filename,
                "content": payload,
            }
        else:
            charset = part.get_content_charset() or "utf-8"
            value = payload.decode(charset, errors="replace")
            fields.setdefault(name, []).append(value)
    return fields, uploaded_file


def _load_dataframe_from_upload(filename: str, file_bytes: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    lower_name = (filename or "").lower()
    if lower_name.endswith(".csv"):
        return pd.read_csv(buffer)
    try:
        return pd.read_excel(buffer)
    except ImportError as exc:
        raise RuntimeError(
            "Reading Excel files requires the 'openpyxl' dependency. Install it with 'pip install openpyxl'."
        ) from exc


def _build_payload_preview(df: pd.DataFrame) -> str:
    trimmed = df
    if PREVIEW_MAX_ROWS > 0:
        trimmed = trimmed.head(PREVIEW_MAX_ROWS)
    if PREVIEW_MAX_COLUMNS > 0:
        trimmed = trimmed.iloc[:, :PREVIEW_MAX_COLUMNS]
    preview_dict = trimmed.to_dict(orient="list")
    preview = json.dumps(preview_dict, indent=2)
    max_bytes = MAX_TEXT_BYTES if MAX_TEXT_BYTES > 0 else None
    if max_bytes is not None and len(preview.encode("utf-8")) > max_bytes:
        encoded = preview.encode("utf-8")
        cutoff = max(max_bytes - 3, 0)
        truncated = encoded[:cutoff].decode("utf-8", errors="ignore")
        preview = f"{truncated}..."
    return preview


def format_stats(stats: dict) -> str:
    lines = [
        f"Avg latency: {stats['avg_latency']:.2f} ms" if stats["avg_latency"] is not None else "Avg latency: N/A",
        f"Total errors: {stats['total_errors']}" if stats["total_errors"] is not None else "Total errors: N/A",
        f"Rows: {stats['row_count']}",
    ]
    if stats.get("top_api_response"):
        entry = stats["top_api_response"]
        lines.append(f"Top API response ({entry['column']}): {entry['value']} [{entry['count']}]")
    if stats.get("top_error_cause"):
        entry = stats["top_error_cause"]
        lines.append(f"Top error cause ({entry['column']}): {entry['value']} [{entry['count']}]")
    if stats.get("top_client_ip"):
        entry = stats["top_client_ip"]
        lines.append(f"Top client IP ({entry['column']}): {entry['value']} [{entry['count']}]")
    if stats["warnings"]:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in stats["warnings"])
    else:
        lines.append("Warnings: none")
    return "\n".join(lines)


class TelemetryHandler(BaseHTTPRequestHandler):
    default_payload = json.dumps(DEFAULT_DATA, indent=2)

    def do_GET(self) -> None:
        self._respond()

    def do_POST(self) -> None:
        telemetry_dict: Optional[dict] = None
        payload: Optional[str] = None
        file_name: Optional[str] = None

        content_length_header = self.headers.get("Content-Length")
        if content_length_header is None:
            self._respond(
                error="Content-Length header is required.",
                payload=self.default_payload,
                status_code=411,
            )
            return
        try:
            content_length = int(content_length_header)
        except ValueError:
            self._respond(error="Invalid Content-Length header.", payload=self.default_payload, status_code=400)
            return
        if content_length < 0:
            self._respond(error="Content-Length cannot be negative.", payload=self.default_payload, status_code=400)
            return

        content_type = self.headers.get("Content-Type", "")
        is_multipart = content_type.startswith("multipart/form-data")
        limit = MAX_UPLOAD_BYTES if is_multipart else MAX_TEXT_BYTES
        if limit > 0 and content_length > limit:
            if is_multipart:
                message = f"Upload exceeds limit of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            else:
                message = f"Telemetry text exceeds limit of {MAX_TEXT_BYTES // 1024} KB."
            self._respond(error=message, payload=self.default_payload, status_code=413)
            return

        try:
            body = _read_exact(self.rfile, content_length)
        except ValueError as exc:
            self._respond(error=str(exc), payload=self.default_payload, status_code=400)
            return

        if is_multipart:
            form_fields: dict[str, list[str]] = {}
            file_part: Optional[dict] = None
            if body:
                try:
                    form_fields, file_part = _parse_multipart_form(body, content_type)
                except ValueError as exc:
                    self._respond(error=str(exc), payload=self.default_payload, status_code=400)
                    return
            action = (form_fields.get("action") or ["analyze"])[0].lower()
            if action == "clear":
                self._respond(payload=self.default_payload)
                return
            payload = (form_fields.get("telemetry") or [""])[0].strip()
            if file_part:
                file_name = file_part["filename"]
                try:
                    data_frame = _load_dataframe_from_upload(file_name, file_part["content"])
                except RuntimeError as exc:
                    logger.error("Dependency missing while reading upload: %s", exc)
                    self._respond(error=str(exc), payload=payload, status_code=400)
                    return
                except Exception as exc:
                    logger.error("Failed to read uploaded file %s: %s", file_name, exc)
                    self._respond(error=f"Failed to read uploaded file: {exc}", payload=payload, status_code=400)
                    return
                telemetry_dict = data_frame.to_dict(orient="list")
                payload = _build_payload_preview(data_frame)
            elif payload:
                try:
                    telemetry_dict = json.loads(payload)
                except json.JSONDecodeError as exc:
                    self._respond(error=f"Invalid JSON: {exc}", payload=payload, status_code=400)
                    return
        else:
            raw_body = body.decode("utf-8", errors="replace") if body else ""
            params = parse_qs(raw_body)
            action = params.get("action", ["analyze"])[0].lower()
            if action == "clear":
                self._respond(payload=self.default_payload)
                return
            payload = params.get("telemetry", [""])[0].strip()
            payload_bytes = len(payload.encode("utf-8"))
            if payload_bytes > MAX_TEXT_BYTES > 0:
                message = f"Telemetry text exceeds limit of {MAX_TEXT_BYTES // 1024} KB."
                self._respond(error=message, payload=self.default_payload, status_code=413)
                return
            if payload:
                try:
                    telemetry_dict = json.loads(payload)
                except json.JSONDecodeError as exc:
                    self._respond(error=f"Invalid JSON: {exc}", payload=payload, status_code=400)
                    return

        if telemetry_dict is None:
            telemetry_dict = DEFAULT_DATA
            payload = json.dumps(DEFAULT_DATA, indent=2)

        try:
            df = build_dataframe(telemetry_dict)
            stats = compute_metrics(df)
        except ValueError as exc:
            self._respond(
                error=f"Telemetry validation failed: {exc}",
                payload=payload,
                file_name=file_name,
                status_code=400,
            )
            return
        except Exception as exc:
            logger.error("Telemetry processing failed: %s", exc)
            self._respond(
                error=f"Telemetry processing failed: {exc}",
                payload=payload,
                file_name=file_name,
                status_code=500,
            )
            return

        prompt = build_prompt(stats)
        try:
            ensure_client()
            analysis = call_openai(CLIENT, prompt)
        except RuntimeError as exc:
            self._respond(
                error=str(exc),
                payload=payload,
                stats=format_stats(stats),
                file_name=file_name,
                status_code=502,
            )
            return

        self._respond(
            payload=payload,
            stats=format_stats(stats),
            analysis=analysis,
            file_name=file_name,
        )

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def _respond(
        self,
        *,
        error: Optional[str] = None,
        payload: Optional[str] = None,
        stats: Optional[str] = None,
        analysis: Optional[str] = None,
        file_name: Optional[str] = None,
        status_code: int = 200,
    ) -> None:
        textarea_value = payload or self.default_payload
        body = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Telemetry Frontend</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2rem; }}
    textarea {{ width: 100%; height: 200px; }}
    pre.resizable {{
      background: #f5f5f5;
      padding: 1rem;
      border-radius: 4px;
      white-space: pre-wrap;
      word-break: break-word;
      width: 100%;
      min-height: 150px;
      max-height: 60vh;
      overflow: auto;
    }}
    .error {{ color: #b00020; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>Telemetry Frontend</h1>
  <p>Paste JSON telemetry data or upload a CSV/Excel log, then analyze or clear the panel to start fresh.</p>
  {f'<p class="error">{html.escape(error)}</p>' if error else ''}
  <form method="post" enctype="multipart/form-data">
    <label for="telemetry">Telemetry JSON</label><br/>
    <textarea id="telemetry" name="telemetry">{html.escape(textarea_value)}</textarea><br/>
    <label for="telemetry_file">Upload log file (.csv/.xlsx/.xls)</label><br/>
    <input type="file" id="telemetry_file" name="telemetry_file" accept=".csv,.xlsx,.xls"/><br/><br/>
    <button type="submit" name="action" value="analyze">Analyze</button>
    <button type="submit" name="action" value="clear">Clear</button>
  </form>
  {f'<p>Last uploaded file: {html.escape(file_name)}</p>' if file_name else ''}
  {f'<h2>Local Metrics</h2><pre class="resizable">{html.escape(stats)}</pre>' if stats else ''}
  {f'<h2>AI Analysis</h2><pre class="resizable">{html.escape(analysis)}</pre>' if analysis else ''}
</body>
</html>
"""
        encoded = body.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    host = host or os.getenv("HOST", "0.0.0.0")
    port = port or int(os.getenv("PORT", "8000"))
    try:
        ensure_client()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)
    server = ThreadingHTTPServer((host, port), TelemetryHandler)
    logger.info("Frontend running at http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down frontend...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
