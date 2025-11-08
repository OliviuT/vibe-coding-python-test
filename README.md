# Vibe Coding Metrics Utilities

Small collection of scripts for experimenting with OpenAI models over simple operational metrics.

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (contains `pandas`, `python-dotenv`, `openai`)
- `.env` file with `OPENAI_API_KEY=...` (and optional `LOG_LEVEL` such as `DEBUG`, `INFO`, etc.)

## Scripts

### `analyse-metrics.py`

1. Loads environment variables and validates the OpenAI key.
2. Builds an in-memory dataframe with sample request/error/latency metrics.
3. Logs local stats, performs a lightweight anomaly check (spikes in last row vs prior mean), and prepares a compact summary of the data.
4. Calls the OpenAI Chat Completions API asking for an SRE-style interpretation.

Run:
```bash
python analyse-metrics.py
```

The script logs structured progress and exits non‑zero if the API key is missing or the OpenAI request fails.

### `test-openai.py`

Minimal harness that wraps Chat Completions in a `ChatCompletionRunner` class with basic validation/error handling. Useful for quick connectivity checks.

Run:
```bash
python test-openai.py
```

## Logging

Both scripts honor `LOG_LEVEL` (default `INFO`). For more detail:

```bash
LOG_LEVEL=DEBUG python analyse-metrics.py
```

## Git Workflow

Typical workflow from the repo root:

```bash
git add .
git commit -m "Describe your change"
git push origin main
```

Remember to keep `.env` and other secrets out of version control (add them to `.gitignore`).

## Web Frontend

`frontend.py` starts a small standard-library HTTP server with a textarea where you can paste telemetry JSON or upload an Excel sheet (with `timestamp`, `requests`, `errors`, `latency_ms` columns). The server parses the input, runs the same anomaly detection, and shows both local metrics and the GPT analysis.

```bash
python frontend.py
```

Open http://127.0.0.1:8000/ in your browser, paste sample data or upload an Excel file, and click **Analyze**. The textarea retains the most recent JSON so you can tweak values and re-run quickly. If you plan to upload `.xlsx` files, ensure `openpyxl` is installed (`pip install openpyxl`).
