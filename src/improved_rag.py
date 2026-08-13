from pathlib import Path
import hashlib
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/addinfo_retrieval")
CACHE_DIR = Path("cache/improved_rag")
MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers/all-MiniLM-L6-v2


class ImprovedFootballRAG:
    """
    Local retrieval engine for football background documents.

    Pipeline: Sentence-Transformers (all-MiniLM-L6-v2) -> FAISS flat
    inner-product (cosine, since embeddings are normalized) index.

    Embeddings + FAISS index + document metadata are cached to disk under
    cache/improved_rag/ so the ~10 minute embedding step only needs to run
    once. The cache is keyed to a cheap fingerprint (file count + total
    byte size of data/addinfo_retrieval/), so it will only be considered
    stale if that folder actually changes.
    """

    def __init__(self, force_rebuild: bool = False):
        self.model = SentenceTransformer(MODEL_NAME)
        self.documents: list[str] = []
        self.filenames: list[str] = []

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_ok = self._cache_matches_corpus() and not force_rebuild

        if cache_ok:
            print("Loading cached embeddings + FAISS index...")
            self._load_cache()
        else:
            print("Building embeddings + FAISS index (first run, this can take several minutes)...")
            self._load_documents()
            print(f"Loaded {len(self.documents)} football documents.")
            embeddings = self.model.encode(
                self.documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            self._save_cache(embeddings)

        print(f"FAISS index ready with {self.index.ntotal} vectors.")

    def _load_documents(self):
        for file in sorted(DATA_DIR.rglob("*")):
            if file.is_file():
                try:
                    text = file.read_text(encoding="utf-8", errors="ignore").strip()
                    if text:
                        text = text[:5000]
                        self.documents.append(text)
                        self.filenames.append(file.name)
                except Exception:
                    pass

    def _corpus_fingerprint(self) -> str:
        files = [f for f in DATA_DIR.rglob("*") if f.is_file()]
        total_size = sum(f.stat().st_size for f in files)
        sig = f"{MODEL_NAME}:{len(files)}:{total_size}"
        return hashlib.md5(sig.encode()).hexdigest()

    def _cache_matches_corpus(self) -> bool:
        fp_file = CACHE_DIR / "fingerprint.txt"
        index_file = CACHE_DIR / "faiss.index"
        docs_file = CACHE_DIR / "documents.json"
        if not (fp_file.exists() and index_file.exists() and docs_file.exists()):
            return False
        return fp_file.read_text().strip() == self._corpus_fingerprint()

    def _save_cache(self, embeddings: np.ndarray):
        faiss.write_index(self.index, str(CACHE_DIR / "faiss.index"))
        with open(CACHE_DIR / "documents.json", "w", encoding="utf-8") as f:
            json.dump(
                {"documents": self.documents, "filenames": self.filenames},
                f,
                ensure_ascii=False,
            )
        (CACHE_DIR / "fingerprint.txt").write_text(self._corpus_fingerprint())
        print(f"Cached embeddings/index to {CACHE_DIR}/")

    def _load_cache(self):
        self.index = faiss.read_index(str(CACHE_DIR / "faiss.index"))
        with open(CACHE_DIR / "documents.json", encoding="utf-8") as f:
            data = json.load(f)
        self.documents = data["documents"]
        self.filenames = data["filenames"]

    def retrieve(self, player, team, opponent, event, match_context, top_k=3):
        query = f"""
        Football match context:
        Player: {player}
        Team: {team}
        Opponent: {opponent}
        Event: {event}
        Match situation: {match_context}
        """
        return self.retrieve_raw(query, top_k=top_k)

    def retrieve_raw(self, query: str, top_k=3):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "score": float(score),
                    "filename": self.filenames[idx],
                    "background": self.documents[idx],
                }
            )
        return results


if __name__ == "__main__":
    rag = ImprovedFootballRAG()
    results = rag.retrieve(
        player="Eden Hazard",
        team="Chelsea",
        opponent="West Brom",
        event="attacking play",
        match_context="Chelsea are attacking in the second half",
        top_k=3,
    )
    print("\n===== RETRIEVED BACKGROUND =====\n")
    for i, result in enumerate(results, 1):
        print(f"[{i}] Similarity: {result['score']:.4f} ({result['filename']})")
        print(result["background"][:1000])
        print("\n" + "=" * 80 + "\n")
