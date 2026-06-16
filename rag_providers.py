"""Shared model provider helpers for the notebooks."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openrouter/free"

# Cheap OpenRouter choices for learning notebooks. Prices and availability can
# change, so keep OPENROUTER_MODEL configurable in .env.
OPENROUTER_MODELS = {
    "free_router": "openrouter/free",
    "stable_ultra_cheap_paid": "inclusionai/ling-2.6-flash",
    "better_low_cost_rag": "qwen/qwen3-235b-a22b-2507",
}


def chat_provider_name() -> str:
    """Return the configured notebook chat provider."""

    load_dotenv()
    provider = os.getenv("RAG_CHAT_PROVIDER", "gemini")
    if provider:
        return provider.lower()
    return "gemini"


def embedding_provider_name() -> str:
    """Return the configured notebook embedding provider."""

    load_dotenv()
    return os.getenv("RAG_EMBEDDING_PROVIDER", "local").lower()


class LocalTfidfEmbeddings:
    """Small local embedding model for API-free notebook retrieval examples.

    This helper is intentionally lightweight and stateful: it fits a TF-IDF
    vocabulary on the first document batch and reuses that vocabulary for later
    queries. It is useful for local demos, but not for persisted vector stores
    or workflows that need to add documents with new vocabulary after indexing.
    Use a hosted or fixed-vocabulary embedding model for those cases.
    """

    def __init__(self, max_features: int = 2048):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            norm="l2",
        )
        self._is_fit = False
        self._fit_source: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._is_fit and self._fit_source == "documents":
            matrix = self.vectorizer.transform(texts)
        else:
            matrix = self.vectorizer.fit_transform(texts)
            self._is_fit = True
            self._fit_source = "documents"
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, text: str) -> list[float]:
        if not self._is_fit:
            self.vectorizer.fit([text])
            self._is_fit = True
            self._fit_source = "query"
        return self.vectorizer.transform([text]).toarray()[0].astype(float).tolist()


def embedding_model(provider: str | None = None, **kwargs: Any):
    """Return the configured embedding model for notebook retrieval cells."""

    selected_provider = (provider or embedding_provider_name()).lower()

    if selected_provider == "local":
        return LocalTfidfEmbeddings(**kwargs)

    if selected_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        load_dotenv()
        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL",
                            "gemini-embedding-2-preview"),
            **kwargs,
        )

    raise ValueError(
        f"Unsupported RAG_EMBEDDING_PROVIDER={selected_provider!r}. "
        "Use 'local' or 'gemini'."
    )


def _openrouter_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set OPENROUTER_API_KEY in .env before calling OpenRouter.")
    return api_key


def openrouter_headers() -> dict[str, str]:
    """Optional OpenRouter app attribution headers."""

    load_dotenv()
    headers: dict[str, str] = {}
    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME")

    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-OpenRouter-Title"] = app_name

    return headers


def openrouter_model_name(default: str = OPENROUTER_DEFAULT_MODEL) -> str:
    load_dotenv()
    return os.getenv("OPENROUTER_MODEL", default)


def openrouter_chat(
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
    **kwargs: Any,
):
    """Return a LangChain chat model backed by OpenRouter's OpenAI-compatible API."""

    from langchain_openai import ChatOpenAI

    load_dotenv()
    params: dict[str, Any] = {
        "model": model or openrouter_model_name(),
        "api_key": _openrouter_api_key(),
        "base_url": os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        "default_headers": openrouter_headers(),
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    return ChatOpenAI(**params)


def chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0,
    max_tokens: int | None = None,
    **kwargs: Any,
):
    """Return the configured chat model for notebook generation cells."""

    selected_provider = (provider or chat_provider_name()).lower()

    if selected_provider == "openrouter":
        return openrouter_chat(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    if selected_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        load_dotenv()
        params: dict[str, Any] = {
            "model": model or os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            "temperature": temperature,
            **kwargs,
        }
        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**params)

    raise ValueError(
        f"Unsupported RAG_CHAT_PROVIDER={selected_provider!r}. "
        "Use 'openrouter' or 'gemini'."
    )


def openrouter_client():
    """Return the raw OpenAI SDK client pointed at OpenRouter."""

    from openai import OpenAI

    load_dotenv()
    return OpenAI(
        api_key=_openrouter_api_key(),
        base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
        default_headers=openrouter_headers(),
    )
