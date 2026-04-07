import logging
from datetime import datetime
from typing import List, Dict, Any

import networkx as nx

from aws.s3_client import S3Client


logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Directed knowledge graph persisted in S3.
    Node types: concept, source, claim, author, session
    Edge types: supports, contradicts, from_source, authored_by,
                cites, related_to, appears_in
    """

    STOPWORDS = {
        "about", "above", "after", "again", "against", "among", "because",
        "before", "between", "could", "every", "first", "found", "great",
        "other", "their", "there", "these", "those", "which", "while",
        "where", "when", "with", "from", "into", "using", "study"
    }

    def __init__(self):
        self.s3 = S3Client()
        self.graph = nx.DiGraph()
        self.load()

    # ------------------------------------------------------------------
    # NODE HELPERS
    # ------------------------------------------------------------------

    def _add_node_if_not_exists(self, node_id: str, **attrs):
        if not self.graph.has_node(node_id):
            self.graph.add_node(node_id, **attrs)
        else:
            # merge attributes (non-destructive update)
            self.graph.nodes[node_id].update(attrs)

    def _extract_concepts(self, text: str) -> List[str]:
        if not text:
            return []

        words = [
            w.strip(".,:;!?()[]{}\"'").lower()
            for w in text.split()
        ]

        return [
            w for w in words
            if len(w) > 5 and w not in self.STOPWORDS
        ]

    # ------------------------------------------------------------------
    # NODE CREATION
    # ------------------------------------------------------------------

    def add_session_node(self, session_id: str, query: str):
        self._add_node_if_not_exists(
            session_id,
            type="session",
            query=query,
            created_at=datetime.utcnow().isoformat()
        )

    def add_source_node(self, source: Dict[str, Any], session_id: str):
        """
        Expected source fields:
        {
            id, title, doi, year, credibility_score, authors: [str]
        }
        """
        source_id = source.get("id") or source.get("doi") or source.get("title")

        if not source_id:
            logger.warning("Skipping source with no identifier")
            return

        # Add source node
        self._add_node_if_not_exists(
            source_id,
            type="source",
            title=source.get("title"),
            doi=source.get("doi"),
            year=source.get("year"),
            credibility_score=source.get("credibility_score"),
            session_id=session_id
        )

        # Link to session
        if self.graph.has_node(session_id):
            self.add_relation(source_id, session_id, "appears_in")

        # Authors
        for author in source.get("authors", []):
            author_id = f"author:{author.lower()}"

            self._add_node_if_not_exists(
                author_id,
                type="author",
                name=author
            )

            self.add_relation(source_id, author_id, "authored_by")

        # Concepts from title
        concepts = self._extract_concepts(source.get("title", ""))

        for concept in concepts:
            concept_id = f"concept:{concept}"

            self._add_node_if_not_exists(
                concept_id,
                type="concept",
                name=concept
            )

            self.add_relation(concept_id, source_id, "appears_in")

    def add_claim_node(self, claim_id: str, claim_text: str, source_id: str):
        self._add_node_if_not_exists(
            claim_id,
            type="claim",
            text=claim_text
        )

        self.add_relation(claim_id, source_id, "from_source")

    # ------------------------------------------------------------------
    # EDGES
    # ------------------------------------------------------------------

    def add_relation(
        self,
        node_a_id: str,
        node_b_id: str,
        relation_type: str,
        weight: float = 1.0
    ):
        if not self.graph.has_node(node_a_id) or not self.graph.has_node(node_b_id):
            return  # silent fail as requested

        self.graph.add_edge(
            node_a_id,
            node_b_id,
            type=relation_type,
            weight=weight
        )

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------

    def get_related_concepts(self, concept: str, depth: int = 2) -> List[str]:
        concept_id = f"concept:{concept.lower()}"

        if not self.graph.has_node(concept_id):
            return []

        visited = set()
        queue = [(concept_id, 0)]
        results = set()

        while queue:
            node, dist = queue.pop(0)

            if dist > depth:
                continue

            if node in visited:
                continue

            visited.add(node)

            # collect concept nodes only
            if self.graph.nodes[node].get("type") == "concept":
                results.add(self.graph.nodes[node].get("name"))

            for neighbor in self.graph.neighbors(node):
                queue.append((neighbor, dist + 1))

            for neighbor in self.graph.predecessors(node):
                queue.append((neighbor, dist + 1))

        return list(results)

    def find_cross_session_connections(
        self,
        session_a: str,
        session_b: str
    ) -> List[Dict[str, Any]]:

        def get_session_concepts(session_id: str):
            concepts = {}

            for node_id, data in self.graph.nodes(data=True):
                if data.get("type") != "source":
                    continue

                if data.get("session_id") != session_id:
                    continue

                # find concepts connected to this source
                for pred in self.graph.predecessors(node_id):
                    if self.graph.nodes[pred].get("type") == "concept":
                        concept_name = self.graph.nodes[pred]["name"]
                        concepts.setdefault(concept_name, []).append(node_id)

            return concepts

        concepts_a = get_session_concepts(session_a)
        concepts_b = get_session_concepts(session_b)

        shared = set(concepts_a.keys()) & set(concepts_b.keys())

        results = []

        for concept in shared:
            results.append({
                "concept": concept,
                "sources_a": concepts_a[concept],
                "sources_b": concepts_b[concept]
            })

        return results

    def stats(self) -> Dict[str, int]:
        node_types = nx.get_node_attributes(self.graph, "type")

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "concept_count": sum(1 for t in node_types.values() if t == "concept"),
            "source_count": sum(1 for t in node_types.values() if t == "source"),
            "session_count": sum(1 for t in node_types.values() if t == "session"),
        }

    # ------------------------------------------------------------------
    # PERSISTENCE (S3)
    # ------------------------------------------------------------------

    def save(self):
        try:
            graph_data = nx.node_link_data(self.graph)

            # Ensure explicit structure
            payload = {
                "nodes": graph_data.get("nodes", []),
                "links": graph_data.get("links", [])
            }

            self.s3.save_knowledge_graph(payload)

            logger.info("Knowledge graph saved to S3")

        except Exception as e:
            logger.exception(f"Failed to save knowledge graph: {e}")

    def load(self):
        try:
            graph_data = self.s3.load_knowledge_graph()

            if graph_data and "nodes" in graph_data:
                self.graph = nx.node_link_graph(graph_data)
                logger.info("Knowledge graph loaded from S3")
            else:
                self.graph = nx.DiGraph()

        except Exception as e:
            logger.warning(f"Failed to load graph from S3, starting fresh: {e}")
            self.graph = nx.DiGraph()
