"""
FAISS-backed vector store for document retrieval.
Uses FAISS for fast, local-only embeddings with proper chunking.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Optional
import faiss
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(
        self,
        data_dir: str,
        persist_dir: str = "./.faiss_db",
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize FAISS vector store with chunking.

        Args:
            data_dir: Root directory containing company subdirectories
            persist_dir: Directory for FAISS persistence
            chunk_size: Size of each text chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.index_path = self.persist_dir / "index.faiss"
        self.metadata_path = self.persist_dir / "metadata.pkl"

        # Initialize the embedding model
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_dim = self.encoder.get_embedding_dimension()

        # Load existing index if available
        self.index = None
        self.metadatas = []

        self._load_index()

    def _load_index(self):
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, 'rb') as f:
                self.metadatas = pickle.load(f)

    def _save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadatas, f)

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)

            # Move start position with overlap
            start += self.chunk_size - self.chunk_overlap

            # Break if we've covered the entire text
            if end >= len(text):
                break

        return chunks

    def index_documents(self, force_reindex: bool = False):
        """
        Index all markdown documents from data directory with chunking.

        Args:
            force_reindex: If True, delete existing collection and reindex
        """
        # Check if already indexed
        if self.index is not None and len(self.metadatas) > 0 and not force_reindex:
            print(f"Index already has {len(self.metadatas)} chunks. Skipping indexing.")
            return

        print("Starting indexing with chunking...")
        print(f"Chunk size: {self.chunk_size} chars, Overlap: {self.chunk_overlap} chars")

        chunks = []
        metadatas = []
        total_files = 0

        # Walk through data directory
        for company_dir in self.data_dir.iterdir():
            if not company_dir.is_dir():
                continue

            company_name = company_dir.name

            for md_file in company_dir.rglob("*.md"):
                try:
                    content = md_file.read_text(encoding='utf-8')

                    # Skip empty files
                    if not content.strip():
                        continue

                    total_files += 1

                    # Chunk the document
                    doc_chunks = self._chunk_text(content)

                    for chunk_idx, chunk in enumerate(doc_chunks):
                        chunks.append(chunk)
                        metadatas.append({
                            "content": chunk,
                            "company": company_name,
                            "file_path": str(md_file),
                            "relative_path": str(md_file.relative_to(company_dir)),
                            "chunk_id": chunk_idx,
                            "total_chunks": len(doc_chunks)
                        })

                except Exception as e:
                    print(f"Error reading {md_file}: {e}")

        if not chunks:
            print("No documents found to index!")
            return

        print(f"Processed {total_files} files into {len(chunks)} chunks")
        print(f"Encoding {len(chunks)} chunks...")

        # Encode chunks
        embeddings = self.encoder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)

        # Create FAISS index (Inner Product / Cosine Similarity since vectors are normalized)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        self.metadatas = metadatas

        self._save_index()
        print(f"Indexed {len(chunks)} chunks from {total_files} documents successfully.")

    def retrieve(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 3,
        min_similarity: float = 0.3
    ) -> List[Dict]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: Search query
            company: Filter by company name (None = search all)
            top_k: Number of chunks to return
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of dicts with keys: content, company, file_path, chunk_id, similarity
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        # Retrieve more results initially to allow for filtering
        fetch_k = top_k * 5 if company else top_k
        distances, indices = self.index.search(query_embedding, min(fetch_k, self.index.ntotal))

        retrieved_chunks = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1:
                continue

            similarity = distances[0][i]
            if similarity < min_similarity:
                continue

            metadata = self.metadatas[idx]

            # Apply company filter
            if company and company.lower() != "none" and metadata.get("company", "").lower() != company.lower():
                continue

            retrieved_chunks.append({
                "content": metadata.get("content", ""),
                "company": metadata.get("company", "Unknown"),
                "file_path": metadata.get("file_path", ""),
                "relative_path": metadata.get("relative_path", ""),
                "chunk_id": metadata.get("chunk_id", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "similarity": round(float(similarity), 3)
            })

            if len(retrieved_chunks) >= top_k:
                break

        return retrieved_chunks

    def get_stats(self) -> Dict:
        """Get collection statistics."""
        count = len(self.metadatas) if self.metadatas else 0

        companies = {}
        files = set()
        for meta in self.metadatas:
            company = meta.get('company', 'Unknown')
            companies[company] = companies.get(company, 0) + 1
            files.add(meta.get('file_path', ''))

        return {
            "total_chunks": count,
            "total_files": len(files),
            "chunks_per_company": companies
        }


def test_retrieval():
    """Test the vector store with chunking."""
    store = VectorStore(data_dir="../data", chunk_size=500, chunk_overlap=100)

    # Index documents
    store.index_documents(force_reindex=True)

    # Print stats
    stats = store.get_stats()
    print("\nVector Store Stats:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total files: {stats['total_files']}")
    print("Chunks per company:")
    for company, count in stats.get('chunks_per_company', {}).items():
        print(f"  {company}: {count}")

    # Test query
    print("\n--- Test Query ---")
    query = "How do I delete my account?"
    results = store.retrieve(query, company="HackerRank", top_k=3)

    print(f"\nQuery: {query}")
    print(f"Company filter: HackerRank")
    print(f"Results: {len(results)}")

    for i, chunk in enumerate(results, 1):
        print(f"\n{i}. Similarity: {chunk['similarity']}")
        print(f"   File: {chunk['relative_path']}")
        print(f"   Chunk: {chunk['chunk_id'] + 1}/{chunk['total_chunks']}")
        print(f"   Content: {chunk['content'][:200]}...")


if __name__ == "__main__":
    test_retrieval()
