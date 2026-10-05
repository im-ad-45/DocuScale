"""End-to-end check: ingest -> [HyDE] -> hybrid retrieve -> rerank -> grounded answer.

Usage:  python run.py            (HyDE per DOCUSCALE_USE_HYDE, default off)
        python run.py --hyde     (force HyDE on)
"""
import argparse
import logging
import sys
import textwrap

from litellm.exceptions import AuthenticationError

from src.config import load_settings
from src.generation.llm import LiteLLMClient
from src.generation.synthesizer import Synthesizer
from src.ingestion.chunker import chunk_document
from src.retrieval.hyde import HydeExpander
from src.retrieval.hybrid import HybridRetriever
from src.storage.vector_store import VectorStore

DOCS: dict[str, str] = {
    "vector-db": (
        "A vector database stores high-dimensional embeddings and retrieves them by "
        "similarity rather than exact match. Approximate nearest neighbour indexes such "
        "as HNSW build a layered graph so a query can hop toward its closest vectors "
        "without scanning every point. Distance metrics like cosine similarity, dot "
        "product and Euclidean distance define what close means. Payload filters let "
        "applications combine semantic search with structured constraints such as "
        "author, date or tenant, which is essential for multi-user retrieval systems."
    ),
    "kyc": (
        "Know Your Customer rules require banks to verify the identity of clients before "
        "opening accounts. Institutions collect government issued identification, proof "
        "of address and information about the source of funds. Higher risk customers, "
        "such as politically exposed persons, trigger enhanced due diligence with ongoing "
        "transaction monitoring. Failures in these checks expose a bank to regulatory "
        "fines and make it easier for criminals to launder money through the system."
    ),
    "sourdough": (
        "Sourdough bread relies on a live culture of wild yeast and lactic acid bacteria "
        "instead of commercial yeast. Bakers feed the starter with flour and water until "
        "it doubles in size and smells pleasantly tangy. The dough then ferments slowly, "
        "often overnight in the refrigerator, which develops flavour and a chewy crumb. "
        "Baking in a preheated covered pot traps steam and produces a crisp, blistered crust."
    ),
    "fatf": (
        "FATF Recommendation 16, known as the travel rule, requires financial institutions "
        "to pass originator and beneficiary information along with wire transfers. When a "
        "payment crosses borders, the sending bank must attach the payer's name, account "
        "number and address, and the receiving bank must check that the details are present. "
        "The rule helps investigators trace funds moved between institutions and jurisdictions."
    ),
}

# (question, doc that must be cited) -- None means the corpus can't answer: expect a refusal.
CASES: list[tuple[str, str | None]] = [
    ("What does FATF Recommendation 16 require for wire transfers?", "fatf"),
    ("How do lenders make sure applicants really are who they claim to be?", "kyc"),
    ("How do HNSW indexes find nearest vectors quickly?", "vector-db"),
    ("Who won the 2018 FIFA World Cup?", None),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="DocuScale end-to-end check")
    parser.add_argument("--hyde", action="store_true", help="force HyDE query expansion on")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    base = load_settings()
    # Small windows so this tiny corpus yields several overlapping chunks per doc.
    settings = base.model_copy(
        update={
            "chunk_size": 40, "chunk_overlap": 10, "candidate_k": 10, "final_top_k": 3,
            "use_hyde": args.hyde or base.use_hyde,
        }
    )
    llm = LiteLLMClient(settings)
    store = VectorStore(settings)
    failures = 0
    try:
        store.create_collection(recreate=True)
        chunks = [
            c
            for doc_id, text in DOCS.items()
            for c in chunk_document(
                text, doc_id, settings.chunk_size, settings.chunk_overlap, {"source": "demo"}
            )
        ]
        print(f"Indexed {store.upsert_documents(chunks)} chunks | HyDE: {settings.use_hyde} "
              f"| LLM: {settings.llm_model}\n")

        retriever = HybridRetriever(store, settings, hyde=HydeExpander(llm, settings))
        synthesizer = Synthesizer(llm, settings)

        for question, expected in CASES:
            result = retriever.retrieve(question)
            answer = synthesizer.answer(question, result.hits)

            print(f"Q: {question}")
            if result.hyde_passage:
                print(textwrap.fill(f"HyDE passage: {result.hyde_passage}", 90, initial_indent="  ",
                                    subsequent_indent="    "))
            print("  Retrieved: " + ", ".join(
                f"{h.chunk.doc_id}#{h.chunk.chunk_index}" for h in result.hits))
            print(textwrap.fill(f"A: {answer.answer}", 90, initial_indent="  ",
                                subsequent_indent="     "))
            print(f"  cited={answer.citations} unknown={answer.unknown_citations} "
                  f"grounded={answer.grounded} refused={answer.refused}")

            if expected is None:
                ok = answer.refused
            else:
                ok = answer.grounded and expected in {c.rsplit("#", 1)[0] for c in answer.citations}
            failures += not ok
            print(f"  -> {'PASS' if ok else 'FAIL'}\n")
    except AuthenticationError:
        print("LLM authentication failed. Set the provider key in .env "
              "(e.g. GROQ_API_KEY) or point DOCUSCALE_LLM_MODEL at a local Ollama model.")
        return 2
    finally:
        store.close()

    print("All checks passed." if not failures else f"{failures} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
