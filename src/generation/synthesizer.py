"""Grounded answer synthesis with programmatically verified citations."""
import re

from pydantic import BaseModel

from src.config import Settings
from src.generation.llm import ChatModel
from src.ingestion.chunker import Chunk
from src.retrieval.hybrid import HybridHit

_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_REF = re.compile(r"\s*(.+?)#(\d+)\s*")


def chunk_label(chunk: Chunk) -> str:
    """The citation label for a chunk, e.g. ``kyc#0``."""
    return f"{chunk.doc_id}#{chunk.chunk_index}"


def extract_citations(text: str) -> list[str]:
    """Pull ``doc_id#index`` labels out of ``[...]`` groups, in order, de-duplicated.

    Tolerates grouped forms like ``[a#0, b#2]`` that small models sometimes emit.
    """
    found: list[str] = []
    for inner in _BRACKET.findall(text):
        for part in re.split(r"[;,]", inner):
            match = _REF.fullmatch(part)
            if match:
                label = f"{match[1]}#{match[2]}"
                if label not in found:
                    found.append(label)
    return found


class Answer(BaseModel):
    """A generated answer plus the evidence that it is (or isn't) grounded."""

    answer: str
    refused: bool  # the model (or an empty context) declined to answer
    grounded: bool  # not refused, cites >= 1 real source, cites no unknown source
    citations: list[str]  # labels cited that exist in the retrieved context
    unknown_citations: list[str]  # labels cited that were NOT in the context
    sources: list[Chunk]  # chunks behind ``citations``


class Synthesizer:
    """Builds a source-restricted prompt, calls the LLM, then audits the citations."""

    def __init__(self, llm: ChatModel, settings: Settings) -> None:
        self._llm = llm
        self._settings = settings

    def answer(self, question: str, hits: list[HybridHit]) -> Answer:
        """Answer ``question`` using only ``hits`` as evidence."""
        refusal = self._settings.refusal_message
        if not hits:  # nothing retrieved: don't pay for an LLM call that can only guess
            return Answer(
                answer=refusal, refused=True, grounded=False,
                citations=[], unknown_citations=[], sources=[],
            )

        by_label = {chunk_label(h.chunk): h.chunk for h in hits}
        context = "\n\n".join(f"[{label}]\n{chunk.text}" for label, chunk in by_label.items())
        text = self._llm.complete(
            system=self._settings.answer_prompt.format(refusal=refusal),
            user=f"SOURCES:\n{context}\n\nQUESTION: {question}",
            max_tokens=self._settings.llm_max_tokens,
        ).strip()

        refused = text.strip("\"'").lower().startswith(refusal.lower())
        cited = extract_citations(text)
        valid = [c for c in cited if c in by_label]
        unknown = [c for c in cited if c not in by_label]
        return Answer(
            answer=text,
            refused=refused,
            grounded=not refused and bool(valid) and not unknown,
            citations=valid,
            unknown_citations=unknown,
            sources=[by_label[c] for c in valid],
        )
