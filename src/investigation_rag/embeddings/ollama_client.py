from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaGeneration:
    response: str
    done_reason: str = ""
    prompt_eval_count: int = 0
    eval_count: int = 0


@dataclass
class OllamaClient:
    base_url: str
    timeout: int = 120
    allow_remote: bool = False
    num_ctx: int = 0
    request_count: int = field(init=False, default=0)
    _session: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.base_url = validate_ollama_base_url(self.base_url, allow_remote=self.allow_remote)
        self.num_ctx = self.num_ctx if self.num_ctx > 0 else 0

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not model:
            raise OllamaError("OLLAMA_EMBED_MODEL is not configured.")
        if not texts:
            return []

        request_json: dict[str, Any] = {"model": model, "input": texts}
        if self.num_ctx:
            request_json["options"] = {"num_ctx": self.num_ctx}

        try:
            response = self._request(
                "POST",
                "/api/embed",
                json=request_json,
            )
            if response.status_code == 404:
                return [self._embed_legacy(model, text) for text in texts]
            response.raise_for_status()
            payload = response.json()
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaError(f"Could not reach local Ollama embeddings endpoint: {exc}") from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise OllamaError("Ollama embeddings response did not include embeddings.")
        return embeddings

    def _embed_legacy(self, model: str, text: str) -> list[float]:
        request_json: dict[str, Any] = {"model": model, "prompt": text}
        if self.num_ctx:
            request_json["options"] = {"num_ctx": self.num_ctx}

        try:
            response = self._request("POST", "/api/embeddings", json=request_json)
            response.raise_for_status()
            payload = response.json()
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaError(f"Could not reach local Ollama legacy embeddings endpoint: {exc}") from exc

        embedding = payload.get("embedding")
        if not isinstance(embedding, list):
            raise OllamaError("Ollama legacy embeddings response did not include embedding.")
        return embedding

    def list_models(self) -> list[dict[str, Any]]:
        try:
            response = self._request("GET", "/api/tags")
            response.raise_for_status()
            payload = response.json()
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaError(f"Could not reach local Ollama model list endpoint: {exc}") from exc

        models = payload.get("models")
        if not isinstance(models, list):
            raise OllamaError("Ollama model list response did not include models.")
        return [model for model in models if isinstance(model, dict)]

    def show_model(self, model: str) -> dict[str, Any]:
        if not model:
            raise OllamaError("Model name is required.")

        try:
            response = self._request(
                "POST",
                "/api/show",
                json={"model": model},
            )
            response.raise_for_status()
            payload = response.json()
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaError(f"Could not reach local Ollama model details endpoint: {exc}") from exc

        if not isinstance(payload, dict):
            raise OllamaError("Ollama model details response was not an object.")
        return payload

    def generate(self, model: str, prompt: str) -> str:
        return self.generate_detailed(model, prompt).response

    def generate_detailed(
        self,
        model: str,
        prompt: str,
        *,
        options: dict[str, Any] | None = None,
        think: bool | None = None,
    ) -> OllamaGeneration:
        if not model:
            raise OllamaError("OLLAMA_LLM_MODEL is not configured.")

        request_json: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
        generation_options = dict(options or {})
        if self.num_ctx:
            generation_options["num_ctx"] = self.num_ctx
        if generation_options:
            request_json["options"] = generation_options
        if think is not None:
            request_json["think"] = think

        try:
            response = self._request(
                "POST",
                "/api/generate",
                json=request_json,
            )
            response.raise_for_status()
            payload = response.json()
        except OllamaError:
            raise
        except Exception as exc:
            raise OllamaError(f"Could not reach local Ollama generation endpoint: {exc}") from exc
        return OllamaGeneration(
            response=str(payload.get("response", "")).strip(),
            done_reason=str(payload.get("done_reason", "") or ""),
            prompt_eval_count=int(payload.get("prompt_eval_count", 0) or 0),
            eval_count=int(payload.get("eval_count", 0) or 0),
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            import requests
        except ImportError as exc:
            raise OllamaError("requests is required for Ollama integration.") from exc

        self.request_count += 1
        if self._session is None:
            self._session = requests.Session()
            self._session.trust_env = False
        return self._session.request(method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs)

    def close(self) -> None:
        if self._session is None:
            return
        close = getattr(self._session, "close", None)
        if callable(close):
            close()
        self._session = None


def validate_ollama_base_url(base_url: str, allow_remote: bool = False) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise OllamaError("OLLAMA_BASE_URL is empty.")

    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise OllamaError("OLLAMA_BASE_URL must use http or https.")
    if not parsed.netloc or parsed.hostname is None:
        raise OllamaError("OLLAMA_BASE_URL must include a host.")
    if parsed.username or parsed.password:
        raise OllamaError("OLLAMA_BASE_URL must not include credentials.")
    if parsed.query or parsed.fragment:
        raise OllamaError("OLLAMA_BASE_URL must not include a query string or fragment.")

    if not allow_remote and not is_loopback_host(parsed.hostname):
        raise OllamaError(
            "Remote Ollama endpoint blocked by local-only mode. "
            "Use localhost, 127.0.0.1, or ::1, or set INVESTIGATION_RAG_ALLOW_REMOTE_OLLAMA=1 "
            "only for a trusted non-sensitive deployment."
        )
    return cleaned


def is_loopback_host(host: str) -> bool:
    normalized = host.strip("[]").rstrip(".").casefold()
    if normalized == "localhost":
        return True
    try:
        return ip_address(normalized).is_loopback
    except ValueError:
        return False
