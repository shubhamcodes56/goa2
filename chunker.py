"""
Speculative Voice-RAG: Chunker & Indexing Pipeline (Member 2)
================================================================
Downloads MSMARCO-XI bilingual dataset, chunks text into parent-child
segments, encodes them via a multilingual SentenceTransformer, and indexes
everything into a local Qdrant vector database.

Designed for 8GB RAM laptops using streaming batch processing.
"""

import os
import uuid
import time
import logging
import nltk
import duckdb
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ─── Logging ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── NLTK Setup ───────────────────────────────────────────
for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)

from nltk.tokenize import sent_tokenize

# ─── Configuration ────────────────────────────────────────
DATASET_REPO = "ai4bharat/MSMARCO-XI"
FILENAME = "train/hintrain.parquet"
MAX_ROWS = 20_000                   # Safe for 8 GB RAM (local mode)
MIN_ANSWER_LEN = 100                # Only high-quality long answers
BATCH_UPSERT_SIZE = 1000            # Qdrant upsert batch size
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "msmarco_chunks"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_TOKEN = "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"

# ─── Chunker ──────────────────────────────────────────────
def chunk_text(text: str, child_target_tokens: int = 128) -> list[str]:
    """
    Split text into sentence-grouped child chunks.
    Estimate: 1 token ≈ 4 characters.
    """
    sentences = sent_tokenize(str(text))
    chunks, current_chunk, current_length = [], [], 0

    for sentence in sentences:
        token_est = len(sentence) / 4
        if current_length + token_est > child_target_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = token_est
        else:
            current_chunk.append(sentence)
            current_length += token_est

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ─── Main Pipeline ────────────────────────────────────────
def build_index():
    # 1. Download dataset from HuggingFace
    log.info("Downloading %s from HuggingFace...", FILENAME)
    file_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=FILENAME,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    file_path = file_path.replace("\\", "/")

    # 2. Load embedding model
    log.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_size = model.get_embedding_dimension()

    # 3. Initialize Qdrant
    log.info("Initializing Qdrant at %s", QDRANT_PATH)
    client = QdrantClient(path=QDRANT_PATH)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # 4. Stream data in chunks to save RAM
    log.info("Streaming up to %s rows in batch mode...", f"{MAX_ROWS:,}")
    con = duckdb.connect()
    res = con.execute(
        f"SELECT Answer, Eng_Answer, query_id FROM '{file_path}' LIMIT {MAX_ROWS}"
    )

    points: list[PointStruct] = []
    processed_rows = 0
    total_chunks = 0
    start_time = time.time()

    while True:
        # Fetch a small DataFrame chunk (saves RAM)
        try:
            df = res.fetch_df_chunk()
        except AttributeError:
            df = res.fetch_df()

        if df.empty:
            break

        # HIGH-QUALITY FILTER: both languages must exist and be long enough
        df = df.dropna(subset=["Answer", "Eng_Answer"])
        df = df[
            (df["Answer"].str.len() >= MIN_ANSWER_LEN)
            & (df["Eng_Answer"].str.len() >= MIN_ANSWER_LEN)
        ]

        for _, row in df.iterrows():
            if processed_rows >= MAX_ROWS:
                break

            passage_id = str(row.get("query_id", uuid.uuid4()))

            # Index BOTH Hindi and English answers for cross-lingual search
            for lang, answer in [("hi", row["Answer"]), ("en", row["Eng_Answer"])]:
                if not answer:
                    continue

                parent_context = str(answer)
                child_chunks = chunk_text(parent_context)

                for chunk in child_chunks:
                    vector = model.encode(chunk).tolist()
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=vector,
                            payload={
                                "passage_id": passage_id,
                                "language": lang,
                                "child_chunk": chunk,
                                "parent_context": parent_context,
                            },
                        )
                    )
                    total_chunks += 1

                    # Batch upsert to keep memory low
                    if len(points) >= BATCH_UPSERT_SIZE:
                        client.upsert(collection_name=COLLECTION_NAME, points=points)
                        points = []

            processed_rows += 1

            # Progress log every 500 rows
            if processed_rows % 500 == 0:
                elapsed = (time.time() - start_time) / 60
                log.info(
                    "Progress: %s rows | %s chunks | %.1f min elapsed",
                    f"{processed_rows:,}",
                    f"{total_chunks:,}",
                    elapsed,
                )

        if processed_rows >= MAX_ROWS:
            break

    # Final batch
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    elapsed = (time.time() - start_time) / 60
    log.info("=" * 50)
    log.info("DONE! %s rows → %s chunks in %.1f minutes", f"{processed_rows:,}", f"{total_chunks:,}", elapsed)
    log.info("Qdrant collection '%s' is ready!", COLLECTION_NAME)
    log.info("=" * 50)


if __name__ == "__main__":
    build_index()
