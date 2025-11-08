import logging
import os
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


class ChatCompletionRunner:
    """Encapsulates OpenAI chat completions with basic safety checks."""

    def __init__(self, *, model: str = "gpt-4o-mini", dotenv_path: str | None = None) -> None:
        load_dotenv(dotenv_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is missing; populate .env before running.")
            raise RuntimeError("OPENAI_API_KEY is missing; populate .env before running.")

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, messages: Iterable[dict[str, str]], *, temperature: float = 0.2, max_tokens: int = 200) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=list(messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.exception("OpenAI chat completion failed.")
            raise RuntimeError("OpenAI chat completion failed.") from exc

        if not response.choices:
            logger.error("OpenAI returned no choices.")
            raise RuntimeError("OpenAI returned no choices.")

        content = response.choices[0].message.content
        if not content:
            logger.error("OpenAI returned an empty message.")
            raise RuntimeError("OpenAI returned an empty message.")

        return content.strip()


def main() -> None:
    runner = ChatCompletionRunner()
    message = runner.complete(
        [
            {"role": "system", "content": "You are a cat."},
            {
                "role": "user",
                "content": "Write a haiku about debugging code with cat sounds only as speech.",
            },
        ]
    )
    logger.info("AI response:\n%s", message)


if __name__ == "__main__":
    main()
