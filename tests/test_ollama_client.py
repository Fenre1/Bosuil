import sys

import pytest

from investigation_rag.embeddings.ollama_client import OllamaClient, OllamaError, validate_ollama_base_url


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434",
        "http://localhost.:11434/",
        "http://127.0.0.1:11434",
        "http://127.4.5.6:11434",
        "http://[::1]:11434",
    ],
)
def test_loopback_ollama_urls_are_allowed(base_url):
    client = OllamaClient(base_url)

    assert client.base_url == base_url.rstrip("/")


@pytest.mark.parametrize(
    "base_url",
    [
        "http://192.168.1.20:11434",
        "https://ollama.example.com",
    ],
)
def test_remote_ollama_urls_are_blocked_by_default(base_url):
    with pytest.raises(OllamaError, match="Remote Ollama endpoint blocked"):
        OllamaClient(base_url)


def test_remote_ollama_urls_require_explicit_override():
    client = OllamaClient("https://ollama.example.com/", allow_remote=True)

    assert client.base_url == "https://ollama.example.com"


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "file://127.0.0.1/model",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434@example.com",
        "http://127.0.0.1:11434?token=secret",
    ],
)
def test_ollama_base_url_rejects_unsafe_shapes(base_url):
    with pytest.raises(OllamaError):
        validate_ollama_base_url(base_url)


def test_ollama_requests_ignore_environment_proxies(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"models": []}

    class FakeSession:
        latest = None

        def __init__(self):
            self.trust_env = True
            self.calls = []
            FakeSession.latest = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def request(self, method, url, timeout, **kwargs):
            self.calls.append((method, url, timeout, kwargs, self.trust_env))
            return FakeResponse()

    class FakeRequests:
        Session = FakeSession

    monkeypatch.setitem(sys.modules, "requests", FakeRequests)

    models = OllamaClient("http://127.0.0.1:11434").list_models()

    assert models == []
    assert FakeSession.latest is not None
    assert FakeSession.latest.calls == [("GET", "http://127.0.0.1:11434/api/tags", 120, {}, False)]
