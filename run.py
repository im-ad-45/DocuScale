"""Sanity check: chunk -> embed -> store -> hybrid retrieve (dense + BM25 + RRF) -> rerank."""
from src.config import load_settings
from src.ingestion.chunker import chunk_document
from src.retrieval.hybrid import HybridHit, HybridRetriever
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

QUERIES: list[tuple[str, str]] = [
    ("What does FATF Recommendation 16 require for wire transfers?", "fatf"),
    ("How do banks confirm who their customers are?", "kyc"),
    ("nearest neighbour search over embeddings", "vector-db"),
]


def show(label: str, hits: list[HybridHit]) -> None:
    """Print one result list with the per-stage evidence."""
    print(f"  {label}")
    for i, h in enumerate(hits, start=1):
        rr = f"{h.rerank_score:.2f}" if h.rerank_score is not None else "-"
        print(
            f"   {i}. [{h.chunk.doc_id}#{h.chunk.chunk_index}] "
            f"dense={h.dense_rank} bm25={h.bm25_rank} rrf={h.rrf_score:.4f} rerank={rr}"
        )


def main() -> None:
    # Small windows so this tiny corpus yields several overlapping chunks per doc.
    settings = load_settings().model_copy(
        update={"chunk_size": 40, "chunk_overlap": 10, "candidate_k": 10, "final_top_k": 3}
    )
    store = VectorStore(settings)
    try:
        store.create_collection(recreate=True)
        chunks = [
            c
            for doc_id, text in DOCS.items()
            for c in chunk_document(
                text, doc_id, settings.chunk_size, settings.chunk_overlap, {"source": "demo"}
            )
        ]
        print(f"Indexed {store.upsert_documents(chunks)} chunks from {len(DOCS)} docs.\n")

        retriever = HybridRetriever(store, settings)
        for query, expected_doc in QUERIES:
            print(f"Q: {query}")
            show("RRF only:", retriever.retrieve(query, rerank=False))
            reranked = retriever.retrieve(query)
            show("RRF + cross-encoder:", reranked)
            top_docs = [r.chunk.doc_id for r in reranked[:2]]
            assert expected_doc in top_docs, f"expected {expected_doc} in {top_docs}"
            print()
        print("Sanity checks passed.")
    finally:
        store.close()


if __name__ == "__main__":
    main()