"""Modal embedding infrastructure - separate from CLI to avoid import issues."""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Modal configuration
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "d802ae16c9caed4d197895d27c6d529434cd8c6d"

# Modal setup at module level
try:
    import modal

    image = modal.Image.debian_slim().pip_install(
        "torch==2.6.0",
        "sentence-transformers==3.4.1",
        "einops==0.8.1",
        "numpy",
    )
    app = modal.App("controlcharts-embedding", image=image)

    CACHE_DIR = "/cache"
    cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)

    @app.cls(
        gpu="T4",
        volumes={CACHE_DIR: cache_vol},
        timeout=60 * 30,
        scaledown_window=60 * 5,
        max_containers=5,
    )
    class EmbeddingModel:
        @modal.enter()
        def setup(self):
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_folder=CACHE_DIR,
                trust_remote_code=True,
            )
            print(f"Modal: Loaded embedding model {MODEL_ID}")

        @modal.method()
        def embed_batch(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,
                convert_to_numpy=True,
            )
            return np.asarray(embeddings, dtype=np.float32)

    MODAL_AVAILABLE = True

except ImportError:
    modal = None
    app = None
    EmbeddingModel = None
    MODAL_AVAILABLE = False


ALPHABET_CACHE = "/home/ubuntu/helivan-chat-a100/projects/data/answer_alphabet.npz"
_ALPHABET_LUT: dict | None = None


def embed_texts(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Pure lookup against the precomputed answer-alphabet cache. Under the
    verbatim-relay protocol every response is a cached answer, "I don't know",
    a temporal integer, or the quine -- so no encoding happens at run time.
    A miss means a non-verbatim string leaked into circulation: raise loudly."""
    global _ALPHABET_LUT
    if _ALPHABET_LUT is None:
        import os
        d = np.load(os.environ.get("ALPHABET_CACHE", ALPHABET_CACHE), allow_pickle=True)
        _ALPHABET_LUT = {t: e for t, e in zip(d["texts"], d["embeddings"])}
    import re
    strip = lambda t: re.sub(r" \[rev \d+\]$", "", t)
    try:
        return np.stack([_ALPHABET_LUT[strip(t)] for t in texts])
    except KeyError as e:
        raise KeyError(
            f"response string not in the verbatim alphabet (protocol violation?): {str(e.args[0])[:120]!r}"
        ) from None


def embed_remote(questions: list[str], batch_size: int = 256) -> np.ndarray:
    """Embed questions using Modal with GPU acceleration."""
    if not MODAL_AVAILABLE:
        raise SystemExit("modal is required for remote embedding. Install with: pip install modal")

    logger.info("Setting up Modal for remote embedding...")

    # Process in chunks for parallelization
    chunk_size = 1000
    chunks = [questions[i:i + chunk_size] for i in range(0, len(questions), chunk_size)]
    logger.info(f"Processing {len(questions)} questions in {len(chunks)} chunks")

    with app.run():
        model = EmbeddingModel()

        # Submit all chunks in parallel
        futures = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Submitting chunk {i + 1}/{len(chunks)} ({len(chunk)} questions)")
            future = model.embed_batch.spawn(chunk, batch_size=batch_size)
            futures.append(future)

        # Collect results
        all_embeddings = []
        for i, future in enumerate(futures):
            logger.info(f"Waiting for chunk {i + 1}/{len(chunks)}...")
            chunk_embeddings = future.get()
            all_embeddings.append(chunk_embeddings)
            logger.info(f"Chunk {i + 1}/{len(chunks)} completed")

    embeddings = np.concatenate(all_embeddings, axis=0)
    logger.info(f"Remote embedding completed. Shape: {embeddings.shape}")
    return embeddings


def embed_local(questions: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed questions using local model (slower but no Modal required)."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading model {MODEL_ID} locally...")
    model = SentenceTransformer(MODEL_ID, trust_remote_code=True)

    logger.info(f"Embedding {len(questions)} questions locally...")
    embeddings = model.encode(
        questions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True
    )

    return np.asarray(embeddings, dtype=np.float32)
