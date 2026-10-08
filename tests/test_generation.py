import litellm
import pytest
from pydantic import SecretStr

from src.generation import llm as llm_module
from src.generation.llm import LiteLLMClient
from src.generation.synthesizer import Synthesizer, extract_citations
from src.ingestion.chunker import chunk_document
from src.retrieval.hybrid import HybridHit
from tests.corpus import DOCS
from tests.fakes import FakeLLM

HITS = [HybridHit(chunk=c, rrf_score=0.1) for c in chunk_document(DOCS["kyc"], "kyc", 40, 10)]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A [kyc#0]. B [kyc#0][fatf#2].", ["kyc#0", "fatf#2"]),
        ("Grouped [a#1, b#3; c#4].", ["a#1", "b#3", "c#4"]),
        ("Spaces in ids [my doc.pdf#4].", ["my doc.pdf#4"]),
        ("No tags, plain [1] brackets [note] and a #hashtag.", []),
        ("", []),
    ],
)
def test_extract_citations(text, expected):
    assert extract_citations(text) == expected


def answer_with(text: str, settings):
    return Synthesizer(FakeLLM(answer=text), settings).answer("q", HITS)


def test_grounded_answer(settings):
    a = answer_with("Banks verify identity [kyc#0]. They monitor risk [kyc#1].", settings)
    assert a.grounded and not a.refused
    assert a.citations == ["kyc#0", "kyc#1"] and a.unknown_citations == []
    assert [c.chunk_index for c in a.sources] == [0, 1]


def test_hallucinated_citation_is_flagged(settings):
    a = answer_with("Made up [zzz#9]. Real [kyc#0].", settings)
    assert not a.grounded
    assert a.unknown_citations == ["zzz#9"] and a.citations == ["kyc#0"]


def test_answer_without_citations_is_not_grounded(settings):
    a = answer_with("Banks check ID.", settings)
    assert not a.grounded and not a.refused and a.citations == []


@pytest.mark.parametrize("wrap", ["{}", '"{}"', "{} ", "  {}"])
def test_refusal_is_detected_by_exact_string_contract(settings, wrap):
    a = answer_with(wrap.format(settings.refusal_message), settings)
    assert a.refused and not a.grounded


def test_refusal_detection_is_case_insensitive_but_not_fuzzy(settings):
    assert answer_with(settings.refusal_message.upper(), settings).refused
    assert not answer_with("I'm not sure, sorry.", settings).refused


def test_empty_retrieval_refuses_without_calling_the_llm(settings):
    llm = FakeLLM()
    a = Synthesizer(llm, settings).answer("q", [])
    assert a.refused and a.answer == settings.refusal_message and llm.calls == []


def test_prompt_contains_rules_sources_and_question(settings):
    llm = FakeLLM()
    Synthesizer(llm, settings).answer("why?", HITS)
    kind, system, user = llm.calls[0]
    assert kind == "answer" and settings.refusal_message in system
    assert user.startswith("SOURCES:\n[kyc#0]\n") and HITS[0].chunk.text in user
    assert user.endswith("QUESTION: why?")


def test_stream_yields_tokens_that_audit_to_the_same_answer(settings):
    llm = FakeLLM()
    synth = Synthesizer(llm, settings)
    tokens = list(synth.stream("q", HITS))
    assert len(tokens) > 1
    streamed = synth.audit("".join(tokens), HITS)
    assert streamed == synth.answer("q", HITS)
    assert streamed.grounded and streamed.citations == ["kyc#0"]


def test_stream_with_no_hits_yields_the_refusal_and_skips_the_llm(settings):
    llm = FakeLLM()
    synth = Synthesizer(llm, settings)
    assert list(synth.stream("q", [])) == [settings.refusal_message]
    assert llm.calls == []


class TestLiteLLMClient:
    @pytest.fixture
    def captured(self, monkeypatch):
        real, seen = litellm.completion, []

        def fake_completion(**kwargs):
            seen.append(kwargs)
            return real(mock_response="hello streaming world", **kwargs)

        monkeypatch.setattr(llm_module.litellm, "completion", fake_completion)
        return seen

    def test_complete(self, settings, captured):
        assert LiteLLMClient(settings).complete("sys", "usr", 50) == "hello streaming world"
        kw = captured[0]
        assert kw["model"] == settings.llm_model and kw["stream"] is False
        assert kw["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
        assert kw["max_tokens"] == 50 and kw["api_key"] is None and kw["api_base"] is None

    def test_stream(self, settings, captured):
        parts = list(LiteLLMClient(settings).stream("sys", "usr", 50))
        assert "".join(parts) == "hello streaming world" and len(parts) > 1
        assert captured[0]["stream"] is True

    def test_provider_settings_are_forwarded(self, settings, captured):
        s = settings.model_copy(
            update={"llm_model": "ollama_chat/llama3.1", "llm_api_base": "http://localhost:11434", "llm_api_key": SecretStr("k")}
        )
        LiteLLMClient(s).complete("a", "b", 10)
        kw = captured[0]
        assert kw["model"] == "ollama_chat/llama3.1" and kw["api_base"] == "http://localhost:11434"
        assert kw["api_key"] == "k"
