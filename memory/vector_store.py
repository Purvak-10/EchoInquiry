import time
import logging
from typing import List, Dict, Any, Optional

import config
from utils.llm_helpers import llm_call_with_retry


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorStore:
    @staticmethod
    def _sanitize_meta_value(value: Any) -> Any:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            # Pinecone supports only list[str] for list metadata values.
            return [str(item) for item in value if item is not None]
        return str(value)

    def _build_safe_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        session_id: str,
        text_preview: str,
    ) -> Dict[str, Any]:
        raw = dict(metadata or {})
        raw.update(
            {
                "session_id": session_id,
                "title": raw.get("title", ""),
                "doi": raw.get("doi", ""),
                "year": raw.get("year", ""),
                "credibility_score": raw.get("credibility_score", 0.0),
                "text": text_preview,
            }
        )
        return {k: self._sanitize_meta_value(v) for k, v in raw.items()}

    def __init__(self):
        try:
            from pinecone import Pinecone, ServerlessSpec
            from sentence_transformers import SentenceTransformer

            logger.info("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            self.index_name = config.PINECONE_INDEX_NAME
            self.dimension = config.EMBEDDING_DIMENSION

            # Initialize embedding model and validate dimension
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}")
            self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            actual_dim = self.model.get_sentence_embedding_dimension()
            
            if actual_dim != self.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch! "
                    f"Model '{config.EMBEDDING_MODEL_NAME}' has {actual_dim} dimensions, "
                    f"but config specifies {self.dimension}. "
                    f"Update EMBEDDING_DIMENSION in .env to {actual_dim}"
                )
            logger.info(f"Embedding model dimension validated: {actual_dim}")

            # Ensure index exists
            existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=config.PINECONE_CLOUD,
                        region=config.PINECONE_REGION,
                    ),
                )

                # Poll until ready
                while True:
                    desc = self.pc.describe_index(self.index_name)
                    if desc.status["ready"]:
                        break
                    logger.info("Waiting for index to be ready...")
                    time.sleep(2)

            # Connect index
            self.index = self.pc.Index(self.index_name)
            logger.info("Connected to Pinecone index.")

            # Reuse the already-loaded model as the embedder (avoids loading twice)
            self.embedder = self.model
            logger.info("Embedding model loaded.")

        except Exception as e:
            logger.exception("Failed to initialize VectorStore")
            raise e

    # -----------------------------
    # ADD SINGLE SOURCE
    # -----------------------------
    def add_source(self, source_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        try:
            if not text:
                logger.warning(f"Empty text for source_id={source_id}")
                return False

            # Validate required metadata
            required_fields = [
                "session_id",
                "title",
                "doi",
                "year",
                "credibility_score",
            ]
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Missing metadata field: {field}")

            truncated_text = text[:1000]
            embedding = self.embedder.encode(truncated_text).tolist()

            metadata = self._build_safe_metadata(
                metadata,
                session_id=str(metadata.get("session_id", "")),
                text_preview=text[:500],
            )

            self.index.upsert(
                vectors=[
                    {
                        "id": source_id,
                        "values": embedding,
                        "metadata": metadata,
                    }
                ]
            )

            logger.info(f"Added source to vector store: {source_id}")
            return True

        except Exception as e:
            logger.exception(f"Failed to add source: {source_id}")
            return False

    # -----------------------------
    # ADD BATCH
    # -----------------------------
    def add_sources_batch(
        self, sources: List[Dict[str, Any]], session_id: str
    ) -> int:
        """
        sources: [
            {
                "id": str,
                "text": str,
                "metadata": {...}
            }
        ]
        """
        added_count = 0

        try:
            if not sources:
                return 0

            # Prepare texts
            texts = [s["text"][:1000] for s in sources]

            # Batch encode
            embeddings = self.embedder.encode(texts)

            vectors = []
            for i, source in enumerate(sources):
                metadata = source.get("metadata", {})
                safe_metadata = self._build_safe_metadata(
                    metadata,
                    session_id=session_id,
                    text_preview=source["text"][:500],
                )

                vectors.append(
                    {
                        "id": source["id"],
                        "values": embeddings[i].tolist(),
                        "metadata": safe_metadata,
                    }
                )

            # Upload in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch)
                added_count += len(batch)
                logger.info(f"Upserted batch {i // batch_size + 1}")

            logger.info(f"Total sources added: {added_count}")
            return added_count

        except Exception as e:
            logger.exception("Batch insert failed")
            return added_count

    # -----------------------------
    # SEARCH (SESSION FILTERED)
    # -----------------------------
    def search(
        self,
        query_text: str,
        n_results: int = 5,
        filter_session: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = self.embedder.encode(query_text).tolist()

            filter_dict = (
                {"session_id": {"$eq": filter_session}}
                if filter_session
                else None
            )

            response = self.index.query(
                vector=query_embedding,
                top_k=n_results,
                include_metadata=True,
                filter=filter_dict,
            )

            results = []
            for match in response.get("matches", []):
                result = {
                    "source_id": match["id"],
                    "score": match["score"],
                    **match.get("metadata", {}),
                }
                results.append(result)

            logger.info(f"Search returned {len(results)} results")
            return results

        except Exception as e:
            logger.exception("Search failed")
            return []

    # -----------------------------
    # SEARCH ACROSS SESSIONS
    # -----------------------------
    def search_across_sessions(
        self, query_text: str, n_results: int = 10
    ) -> List[Dict[str, Any]]:
        return self.search(query_text, n_results, filter_session=None)

    # -----------------------------
    # RETRIEVE + GENERATE
    # -----------------------------
    def retrieve_and_generate(self, query: str, session_id: str) -> str:
        try:
            results = self.search(query, n_results=5, filter_session=session_id)

            context_chunks = [
                r.get("text", "") for r in results if r.get("text")
            ]
            context = "\n\n".join(context_chunks)

            prompt = f"Context:\n{context}\n\nAnswer: {query}"

            response = llm_call_with_retry(
                prompt=prompt,
                session_id=session_id,
                step_name="kb_retrieve",
                parse_fn=lambda x: x,
                fallback="",
            )

            return response

        except Exception as e:
            logger.exception("retrieve_and_generate failed")
            return ""

    # -----------------------------
    # STATS
    # -----------------------------
    def stats(self) -> Dict[str, Any]:
        try:
            stats = self.index.describe_index_stats()

            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "index_name": self.index_name,
                "status": "connected",
            }

        except Exception as e:
            logger.exception("Failed to get stats")
            return {
                "total_vector_count": 0,
                "index_name": self.index_name,
                "status": "error",
            }
