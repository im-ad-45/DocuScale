"""Phase 1 sanity check: chunk -> embed -> store -> search."""
from src.config import load_settings
from src.ingestion.chunker import chunk_document
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
}

QUERIES: list[tuple[str, str]] = [
    ("How do banks confirm who their customers are?", "kyc"),
    ("nearest neighbour search over embeddings", "vector-db"),
    ("why does my starter need feeding?", "sourdough"),
]


def main() -> None:
    settings = load_settings()
    # Small windows so this tiny corpus yields several overlapping chunks per doc.
    settings = settings.model_copy(update={"chunk_size": 40, "chunk_overlap": 10})

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
        print(f"Chunked {len(DOCS)} docs into {len(chunks)} chunks; upserting...")
        print(f"Stored {store.upsert_documents(chunks)} chunks.\n")

        for query, expected_doc in QUERIES:
            hits = store.dense_search(query, top_k=3)
            print(f"Q: {query}")
            for h in hits:
                print(f"  {h.score:.3f}  [{h.chunk.doc_id}#{h.chunk.chunk_index}] {h.chunk.text[:70]}...")
            assert hits[0].chunk.doc_id == expected_doc, f"expected top hit from {expected_doc}"
            print()
        print("Phase 1 sanity checks passed.")
    finally:
        store.close()


if __name__ == "__main__":
    main()
