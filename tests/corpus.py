"""Small, topically distinct corpus shared by the tests."""
from src.config import Settings
from src.ingestion.chunker import Chunk, chunk_document
from src.storage.vector_store import VectorStore

DOCS: dict[str, str] = {
    "kyc": (
        "Know Your Customer rules require banks to verify the identity of clients before "
        "opening accounts. Institutions collect government issued identification proof of "
        "address and information about the source of funds. Higher risk customers such as "
        "politically exposed persons trigger enhanced due diligence with ongoing "
        "transaction monitoring."
    ),
    "fatf": (
        "FATF Recommendation 16 known as the travel rule requires financial institutions "
        "to pass originator and beneficiary information along with wire transfers. When a "
        "payment crosses borders the sending bank must attach the payer name account "
        "number and address."
    ),
    "sourdough": (
        "Sourdough bread relies on a live culture of wild yeast and lactic acid bacteria "
        "instead of commercial yeast. Bakers feed the starter with flour and water until "
        "it doubles in size. The dough ferments slowly overnight in the refrigerator "
        "which develops flavour and a chewy crumb."
    ),
    "vector-db": (
        "A vector database stores high dimensional embeddings and retrieves them by "
        "similarity rather than exact match. Approximate nearest neighbour indexes such "
        "as HNSW build a layered graph so a query can hop toward its closest vectors "
        "without scanning every point."
    ),
}

FATF_QUESTION = "What does FATF Recommendation 16 require for wire transfers?"


def chunk_corpus(settings: Settings) -> list[Chunk]:
    return [
        c
        for doc_id, text in DOCS.items()
        for c in chunk_document(text, doc_id, settings.chunk_size, settings.chunk_overlap)
    ]


def index_corpus(store: VectorStore, settings: Settings) -> list[Chunk]:
    chunks = chunk_corpus(settings)
    store.upsert_documents(chunks)
    return chunks
