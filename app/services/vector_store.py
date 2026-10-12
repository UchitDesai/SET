from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Optional


class VectorStore:
    def __init__(self, session_id: str):
        """
        Session-isolated FAISS vector store.
        Each session gets its own directory under vector_store/<session_id>
        """

        self.base_path  = os.path.join("vector_store", session_id)
        self.index_path = os.path.join(self.base_path, "faiss_index.idx")
        self.data_path  = os.path.join(self.base_path, "data.pkl")

        os.makedirs(self.base_path, exist_ok=True)

        # Multilingual embedding model
        self.embedding_model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.index = None
        self.data  = []
        self._load_or_initialize_index()

    def _load_or_initialize_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.data_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.data_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            dim = self.embedding_model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
            self.data  = []

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        if not texts:
            return

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True
        )
        embeddings = np.array(embeddings).astype("float32")

        self.index.add(embeddings)

        if metadatas is None:
            metadatas = [{}] * len(texts)

        for text, meta in zip(texts, metadatas):
            self.data.append({
                "text":     text,
                "metadata": meta
            })

        self.save()

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Returns top_k results with L2 distances included.

        Each result dict contains:
          - text:     stored text chunk
          - metadata: structured metadata (verdict, score, etc.)
          - distance: L2 distance from query embedding
                      (lower = more similar; used for distance-weighted memory)
        """
        if not self.data:
            return []

        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        actual_top_k = min(top_k, len(self.data))
        distances, indices = self.index.search(query_embedding, actual_top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.data):
                entry = self.data[idx].copy()
                entry["distance"] = float(dist)   # L2 distance from FAISS
                results.append(entry)

        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.data_path, "wb") as f:
            pickle.dump(self.data, f)

    def clear(self):
        dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.data  = []
        self.save()