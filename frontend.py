import cgi
import html
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import parse_qs

import pandas as pd

from metrics_core import DEFAULT_DATA, build_dataframe, build_prompt, call_openai, compute_metrics, init_client

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CLIENT = None


def ensure_client() -> None:
    global CLIENT
    if CLIENT is None:
        CLIENT = init_client()


def format_stats(stats: dict) -> str:
    lines = [
        f"Avg latency: {stats['avg_latency']:.2f} ms",
        f"Total errors: {stats['total_errors']}",
        f"Rows: {stats['row_count']}",
    ]
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
        payload = None
        file_name = None

        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("multipart/form-data"):
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                },
            )
            payload = (form.getfirst("telemetry") or "").strip()
            file_field = form["telemetry_file"] if "telemetry_file" in form else None
            if file_field is not None and getattr(file_field, "filename", ""):
                file_name = file_field.filename
                try:
                    file_field.file.seek(0)
                    excel_df = pd.read_excel(file_field.file)
                except ImportError as exc:
                    message = (
                        "Reading Excel files requires the 'openpyxl' dependency. "
                        "Install it with 'pip install openpyxl'."
                    )
                    logger.error(message)
                    self._respond(error=message, payload=payload)
                    return
                except Exception as exc:
                    logger.error("Failed to read Excel file: %s", exc)
                    self._respond(error=f"Failed to read Excel file: {exc}", payload=payload)
                    return
                telemetry_dict = excel_df.to_dict(orient="list")
                payload = json.dumps(telemetry_dict, indent=2)
            elif payload:
                try:
                    telemetry_dict = json.loads(payload)
                except json.JSONDecodeError as exc:
                    logger.error("Invalid JSON submitted: %s", exc)
                    self._respond(error=f"Invalid JSON: {exc}", payload=payload)
                    return
        else:
            content_length = int(self.headers.get("Content-Length") or 0)
            raw_body = self.rfile.read(content_length).decode("utf-8")
            params = parse_qs(raw_body)
            payload = params.get("telemetry", [""])[0].strip()
            if payload:
                try:
                    telemetry_dict = json.loads(payload)
                except json.JSONDecodeError as exc:
                    logger.error("Invalid JSON submitted: %s", exc)
                    self._respond(error=f"Invalid JSON: {exc}", payload=payload)
                    return

        if telemetry_dict is None:
            telemetry_dict = DEFAULT_DATA
            payload = json.dumps(DEFAULT_DATA, indent=2)

        try:
            df = build_dataframe(telemetry_dict)
            stats = compute_metrics(df)
        except Exception as exc:
            logger.error("Telemetry processing failed: %s", exc)
            self._respond(error=f"Telemetry processing failed: {exc}", payload=payload, file_name=file_name)
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
    pre {{ background: #f5f5f5; padding: 1rem; border-radius: 4px; }}
    .error {{ color: #b00020; margin-bottom: 1rem; }}
  </style>
</head>
<body>
  <h1>Telemetry Frontend</h1>
  <p>Paste JSON telemetry data or upload an Excel file and submit to run the GPT analysis.</p>
  {f'<p class="error">{html.escape(error)}</p>' if error else ''}
  <form method="post" enctype="multipart/form-data">
    <label for="telemetry">Telemetry JSON</label><br/>
    <textarea id="telemetry" name="telemetry">{html.escape(textarea_value)}</textarea><br/>
    <label for="telemetry_file">Excel upload (.xlsx/.xls)</label><br/>
    <input type="file" id="telemetry_file" name="telemetry_file" accept=".xlsx,.xls"/><br/><br/>
    <button type="submit">Analyze</button>
  </form>
  {f'<p>Last uploaded file: {html.escape(file_name)}</p>' if file_name else ''}
  {f'<h2>Local Metrics</h2><pre>{html.escape(stats)}</pre>' if stats else ''}
  {f'<h2>AI Analysis</h2><pre>{html.escape(analysis)}</pre>' if analysis else ''}
</body>
</html>
"""
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
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
