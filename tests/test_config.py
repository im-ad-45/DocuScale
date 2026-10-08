import pytest

from src import config
from src.config import DEFAULT_ANSWER_PROMPT, Settings, load_settings


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Ignore any developer .env and DOCUSCALE_* variables."""
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    import os
    for key in [k for k in os.environ if k.startswith("DOCUSCALE_")]:
        monkeypatch.delenv(key)


def test_defaults():
    s = load_settings()
    assert s.chunk_size == 200 and s.chunk_overlap == 40
    assert s.use_hyde is False and s.llm_api_key is None


def test_env_overrides_are_coerced(monkeypatch):
    monkeypatch.setenv("DOCUSCALE_USE_HYDE", "true")
    monkeypatch.setenv("DOCUSCALE_CHUNK_SIZE", "123")
    monkeypatch.setenv("DOCUSCALE_STORAGE_PATH", "/tmp/x")
    s = load_settings()
    assert s.use_hyde is True and s.chunk_size == 123 and str(s.storage_path).endswith("x")


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=10, chunk_overlap=10)


def test_api_key_is_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("DOCUSCALE_LLM_API_KEY", "sk-secret-123")
    s = load_settings()
    assert "sk-secret-123" not in repr(s)
    assert s.llm_api_key.get_secret_value() == "sk-secret-123"


def test_answer_prompt_formats_with_refusal():
    assert "exactly: NOPE" in DEFAULT_ANSWER_PROMPT.format(refusal="NOPE")


def test_settings_are_immutable():
    with pytest.raises(Exception):
        Settings().chunk_size = 5
