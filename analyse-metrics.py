import logging
import os
import sys

from metrics_core import (
    DEFAULT_DATA,
    build_dataframe,
    build_prompt,
    call_openai,
    compute_metrics,
    init_client,
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        client = init_client()
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    df = build_dataframe(DEFAULT_DATA)
    logger.debug("Raw frame:\n%s", df)

    metrics = compute_metrics(df)
    logger.info("Local stats:")
    logger.info("Avg latency: %.2f ms", metrics["avg_latency"])
    logger.info("Total errors: %d", metrics["total_errors"])
    for warning in metrics["warnings"]:
        logger.warning(warning)

    prompt = build_prompt(metrics)
    try:
        analysis = call_openai(client, prompt)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("AI analysis:\n%s", analysis)


if __name__ == "__main__":
    main()
