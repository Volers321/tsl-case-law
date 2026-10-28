"""Thin provider-agnostic LLM client (Anthropic or OpenAI), configured via env vars.

Shared with the sibling project rag-eval-pl; budgets account for reasoning tokens."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines, # comments); real env vars take precedence."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split(" #", 1)[0].strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        value = value.strip("'\"")
        if value and not os.environ.get(key):
            os.environ[key] = value


@dataclass
class LLM:
    provider: str
    model: str
    last_truncated: bool = False

    @classmethod
    def from_env(cls) -> LLM:
        load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}")
        key_var = f"{provider.upper()}_API_KEY"
        if not os.getenv(key_var):
            raise SystemExit(f"{key_var} is not set — add it to .env (see .env.example).")
        return cls(provider, os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider])

    def complete(self, system: str, user: str, *, max_tokens: int = 4096) -> str:
        """Return the model's text. If the output hit the token limit (recent models spend
        part of the budget on internal reasoning before answering), retry once with double
        the budget. ``self.last_truncated`` tells callers whether the text is still cut."""
        for budget in (max_tokens, max_tokens * 2):
            try:
                text, truncated = self._call(system, user, budget)
            except Exception as e:  # provider SDKs raise their own AuthenticationError
                if type(e).__name__ in ("AuthenticationError", "PermissionDeniedError"):
                    raise SystemExit(
                        f"\nAPI key rejected ({type(e).__name__}). The key in .env is invalid, "
                        "deleted or disabled — create a new one in the provider console, put it "
                        "in .env and run the same command again (finished work is cached)."
                    ) from None
                raise
            if not truncated:
                break
        self.last_truncated = truncated
        return text

    def _call(self, system: str, user: str, max_tokens: int) -> tuple[str, bool]:
        if self.provider == "anthropic":
            import anthropic

            # No sampling parameters: recent SDK/model versions reject `temperature`.
            # Reproducibility comes from the fixed prompts and eval set instead.
            msg = anthropic.Anthropic().messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content)
            return text, msg.stop_reason == "max_tokens"

        from openai import OpenAI

        resp = OpenAI().chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        choice = resp.choices[0]
        return choice.message.content or "", choice.finish_reason == "length"
