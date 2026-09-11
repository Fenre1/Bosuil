from investigation_rag.app import preload_torch_windows_dll


def test_torch_preload_is_safe_when_torch_is_unavailable(monkeypatch):
    monkeypatch.setattr("investigation_rag.app.find_spec", lambda _name: None)
    preload_torch_windows_dll()
