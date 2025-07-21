"""
Vector operations for Neo4j Vector Graph Schema.

Provides vector similarity search, hybrid graph-vector queries,
and embedding management for the knowledge graph.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import structlog
import numpy as np
from neo4j import AsyncDriver
from neo4j.exceptions import Neo4jError

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import VectorOperationError, DatabaseError
from kg_rag.graph_schema.node_models import VectorEmbedding, NodeType

logger = structlog.get_logger(__name__)


class VectorGraphOperations:
    """Vector operations for Neo4j graph with embeddings."""
    
    def __init__(self, driver: AsyncDriver):
        """Initialize vector operations.
        
        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
        self.settings = get_settings()
        
    async def vector_similarity_search(
        self,
        query_vector: List[float],
        node_types: Optional[List[NodeType]] = None,
        embedding_field: str = "content_embedding",
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search.
        
        Args:
            query_vector: Query embedding vector
            node_types: Optional node types to search
            embedding_field: Embedding field to search (content_embedding or title_embedding)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Additional filters to apply
            
        Returns:
            List of similar nodes with similarity scores
        """
        logger.info(
            "Performing vector similarity search",
            embedding_field=embedding_field,
            limit=limit,
            threshold=similarity_threshold,
            node_types=node_types
        )
        
        try:
            # Build node type filter
            if node_types:
                node_labels = ":".join([f"`{nt.value}`" for nt in node_types])
                node_filter = f"(n:{node_labels})"
            else:
                node_filter = "(n)"
            
            # Build additional filters
            where_clauses = [f"n.{embedding_field} IS NOT NULL"]
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        where_clauses.append(f"n.{key} = '{value}'")
                    elif isinstance(value, list):
                        value_str = "', '".join(str(v) for v in value)
                        where_clauses.append(f"n.{key} IN ['{value_str}']")
                    else:
                        where_clauses.append(f"n.{key} = {value}")
            
            where_clause = " AND ".join(where_clauses)
            
            # Neo4j vector similarity query
            query = f"""
            CALL db.index.vector.queryNodes('{embedding_field.replace("_", "_")}s', {limit}, $query_vector)
            YIELD node, score
            WHERE {where_clause.replace("n.", "node.")} AND score >= $threshold
            RETURN 
                node.node_id as node_id,
                node.node_type as node_type,
                node.title as title,
                node.description as description,
                node.tags as tags,
                node.categories as categories,
                score as similarity_score,
                node
            ORDER BY score DESC
            """
            
            async with self.driver.session() as session:
                result = await session.run(
                    query,
                    query_vector=query_vector,
                    threshold=similarity_threshold
                )
                
                results = []
                async for record in result:
                    node_data = {
                        "node_id": record["node_id"],
                        "node_type": record["node_type"],
                        "title": record["title"],
                        "description": record["description"],
                        "tags": record["tags"] or [],
                        "categories": record["categories"] or [],
                        "similarity_score": record["similarity_score"],
                        "node_properties": dict(record["node"])
                    }
                    results.append(node_data)
                
                logger.info(
                    "Vector similarity search completed",
                    results_found=len(results),
                    max_score=max([r["similarity_score"] for r in results]) if results else 0,
                    min_score=min([r["similarity_score"] for r in results]) if results else 0
                )
                
                return results
                
        except Exception as e:
            error_msg = f"Vector similarity search failed: {str(e)}"
            logger.error(error_msg, error=str(e))
            raise VectorOperationError(error_msg) from e
    
    async def hybrid_search(
        self,
        query_vector: List[float],
        graph_filters: Optional[Dict[str, Any]] = None,
        relationship_filters: Optional[List[str]] = None,
        node_types: Optional[List[NodeType]] = None,
        embedding_field: str = "content_embedding",
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        limit: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform hybrid graph-vector search.
        
        Args:
            query_vector: Query embedding vector
            graph_filters: Graph-based filters
            relationship_filters: Relationship types to consider
            node_types: Node types to search
            embedding_field: Embedding field to use
            vector_weight: Weight for vector similarity
            graph_weight: Weight for graph connectivity
            limit: Maximum results
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of hybrid search results with combined scores
        """
        logger.info(
            "Performing hybrid graph-vector search",
            vector_weight=vector_weight,
            graph_weight=graph_weight,
            limit=limit
        )
        
        try:
            # Step 1: Get vector similarity results
            vector_results = await self.vector_similarity_search(
                query_vector=query_vector,
                node_types=node_types,
                embedding_field=embedding_field,
                limit=limit * 2,  # Get more candidates for reranking
                similarity_threshold=similarity_threshold,
                filters=graph_filters
            )
            
            if not vector_results:
                return []
            
            # Step 2: Enhance with graph connectivity scores
            node_ids = [result["node_id"] for result in vector_results]
            graph_scores = await self._calculate_graph_scores(
                node_ids=node_ids,
                relationship_filters=relationship_filters,
                graph_filters=graph_filters
            )
            
            # Step 3: Combine scores
            hybrid_results = []
            for result in vector_results:
                node_id = result["node_id"]
                vector_score = result["similarity_score"]
                graph_score = graph_scores.get(node_id, 0.0)
                
                # Calculate hybrid score
                hybrid_score = (vector_weight * vector_score) + (graph_weight * graph_score)
                
                hybrid_result = result.copy()
                hybrid_result.update({
                    "vector_score": vector_score,
                    "graph_score": graph_score,
                    "hybrid_score": hybrid_score
                })
                
                hybrid_results.append(hybrid_result)
            
            # Step 4: Sort by hybrid score and limit results
            hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            final_results = hybrid_results[:limit]
            
            logger.info(
                "Hybrid search completed",
                vector_results=len(vector_results),
                final_results=len(final_results),
                avg_hybrid_score=sum([r["hybrid_score"] for r in final_results]) / len(final_results) if final_results else 0
            )
            
            return final_results
            
        except Exception as e:
            error_msg = f"Hybrid search failed: {str(e)}"
            logger.error(error_msg, error=str(e))
            raise VectorOperationError(error_msg) from e
    
    async def _calculate_graph_scores(
        self,
        node_ids: List[str],
        relationship_filters: Optional[List[str]] = None,
        graph_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Calculate graph-based scores for nodes.
        
        Args:
            node_ids: List of node IDs to score
            relationship_filters: Relationship types to consider
            graph_filters: Additional graph filters
            
        Returns:
            Dict mapping node IDs to graph scores
        """
        try:
            # Build relationship filter
            rel_filter = ""
            if relationship_filters:
                rel_types = "|".join(relationship_filters)
                rel_filter = f":{rel_types}"
            
            # Calculate various graph metrics
            queries = {
                "degree_centrality": f"""
                UNWIND $node_ids as node_id
                MATCH (n {{node_id: node_id}})
                OPTIONAL MATCH (n)-[r{rel_filter}]-()
                RETURN node_id, count(r) as degree
                """,
                
                "pagerank": f"""
                UNWIND $node_ids as node_id
                MATCH (n {{node_id: node_id}})
                OPTIONAL MATCH (n)-[r{rel_filter}]-(connected)
                WITH n, count(connected) as connections
                RETURN n.node_id as node_id, 
                       CASE WHEN connections > 0 
                            THEN log(1 + connections) / 10.0 
                            ELSE 0.0 
                       END as pagerank_score
                """,
                
                "clustering": f"""
                UNWIND $node_ids as node_id
                MATCH (n {{node_id: node_id}})
                OPTIONAL MATCH (n)-[r1{rel_filter}]-(neighbor1)-[r2{rel_filter}]-(neighbor2)-[r3{rel_filter}]-(n)
                WITH n, count(DISTINCT neighbor1) as triangles, 
                     count(DISTINCT neighbor2) as total_neighbors
                RETURN n.node_id as node_id,
                       CASE WHEN total_neighbors > 1 
                            THEN toFloat(triangles) / (total_neighbors * (total_neighbors - 1) / 2)
                            ELSE 0.0 
                       END as clustering_score
                """
            }
            
            scores = {}
            
            async with self.driver.session() as session:
                # Initialize scores
                for node_id in node_ids:
                    scores[node_id] = 0.0
                
                # Calculate each metric
                for metric_name, query in queries.items():
                    try:
                        result = await session.run(query, node_ids=node_ids)
                        
                        metric_scores = {}
                        async for record in result:
                            node_id = record["node_id"]
                            score = record.get(f"{metric_name.split('_')[0]}_score", record.get("degree", 0))
                            metric_scores[node_id] = float(score) if score is not None else 0.0
                        
                        # Normalize scores to 0-1 range
                        if metric_scores:
                            max_score = max(metric_scores.values()) if metric_scores.values() else 1.0
                            if max_score > 0:
                                normalized_scores = {
                                    node_id: score / max_score 
                                    for node_id, score in metric_scores.items()
                                }
                            else:
                                normalized_scores = {node_id: 0.0 for node_id in metric_scores}
                            
                            # Combine with existing scores (weighted average)
                            weight = 1.0 / len(queries)  # Equal weight for each metric
                            for node_id in node_ids:
                                scores[node_id] += weight * normalized_scores.get(node_id, 0.0)
                    
                    except Exception as e:
                        logger.warning(f"Failed to calculate {metric_name}: {str(e)}")
                        continue
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate graph scores: {str(e)}")
            return {node_id: 0.0 for node_id in node_ids}
    
    async def find_similar_nodes_by_content(
        self,
        node_id: str,
        similarity_threshold: float = 0.8,
        limit: int = 5,
        exclude_self: bool = True
    ) -> List[Dict[str, Any]]:
        """Find nodes similar to a given node by content embedding.
        
        Args:
            node_id: Source node ID
            similarity_threshold: Minimum similarity score
            limit: Maximum number of results
            exclude_self: Whether to exclude the source node
            
        Returns:
            List of similar nodes
        """
        try:
            # First, get the source node's embedding
            get_embedding_query = """
            MATCH (source {node_id: $node_id})
            WHERE source.content_embedding IS NOT NULL
            RETURN source.content_embedding as embedding, source.node_type as node_type
            """
            
            async with self.driver.session() as session:
                result = await session.run(get_embedding_query, node_id=node_id)
                record = await result.single()
                
                if not record:
                    logger.warning(f"Node {node_id} not found or has no content embedding")
                    return []
                
                source_embedding = record["embedding"]
                source_node_type = record["node_type"]
                
                # Find similar nodes
                similar_nodes = await self.vector_similarity_search(
                    query_vector=source_embedding,
                    node_types=[NodeType(source_node_type)],
                    embedding_field="content_embedding",
                    limit=limit + (1 if not exclude_self else 0),
                    similarity_threshold=similarity_threshold
                )
                
                # Exclude self if requested
                if exclude_self:
                    similar_nodes = [
                        node for node in similar_nodes 
                        if node["node_id"] != node_id
                    ]
                
                return similar_nodes[:limit]
                
        except Exception as e:
            error_msg = f"Failed to find similar nodes for {node_id}: {str(e)}"
            logger.error(error_msg, node_id=node_id, error=str(e))
            raise VectorOperationError(error_msg) from e
    
    async def create_similarity_relationships(
        self,
        similarity_threshold: float = 0.85,
        batch_size: int = 100,
        relationship_type: str = "SIMILAR_TO",
        node_types: Optional[List[NodeType]] = None
    ) -> Dict[str, Any]:
        """Create similarity relationships between nodes based on embeddings.
        
        Args:
            similarity_threshold: Minimum similarity for creating relationships
            batch_size: Number of nodes to process per batch
            relationship_type: Type of relationship to create
            node_types: Node types to process
            
        Returns:
            Dict with creation statistics
        """
        logger.info(
            "Creating similarity relationships",
            threshold=similarity_threshold,
            batch_size=batch_size,
            relationship_type=relationship_type
        )
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "nodes_processed": 0,
            "relationships_created": 0,
            "batches_processed": 0,
            "errors": []
        }
        
        try:
            # Build node filter
            if node_types:
                node_labels = ":".join([f"`{nt.value}`" for nt in node_types])
                node_filter = f":{node_labels}"
            else:
                node_filter = ""
            
            # Get all nodes with embeddings
            get_nodes_query = f"""
            MATCH (n{node_filter})
            WHERE n.content_embedding IS NOT NULL
            RETURN n.node_id as node_id, n.content_embedding as embedding
            ORDER BY n.node_id
            """
            
            async with self.driver.session() as session:
                result = await session.run(get_nodes_query)
                nodes = [
                    {"node_id": record["node_id"], "embedding": record["embedding"]}
                    async for record in result
                ]
                
                logger.info(f"Processing {len(nodes)} nodes for similarity relationships")
                
                # Process nodes in batches
                for i in range(0, len(nodes), batch_size):
                    batch = nodes[i:i + batch_size]
                    batch_relationships = 0
                    
                    async with session.begin_transaction() as tx:
                        for j, node_a in enumerate(batch):
                            # Compare with all subsequent nodes to avoid duplicates
                            for node_b in nodes[i + j + 1:]:
                                try:
                                    # Calculate similarity
                                    similarity = self._calculate_cosine_similarity(
                                        node_a["embedding"], 
                                        node_b["embedding"]
                                    )
                                    
                                    if similarity >= similarity_threshold:
                                        # Create similarity relationship
                                        create_rel_query = f"""
                                        MATCH (a {{node_id: $node_a_id}})
                                        MATCH (b {{node_id: $node_b_id}})
                                        CREATE (a)-[r:{relationship_type} {{
                                            relationship_id: $rel_id,
                                            relationship_type: $rel_type,
                                            similarity_score: $similarity,
                                            similarity_metric: 'cosine',
                                            comparison_method: 'content_embedding',
                                            weight: $similarity,
                                            confidence: 1.0,
                                            created_at: $timestamp
                                        }}]->(b)
                                        """
                                        
                                        await tx.run(
                                            create_rel_query,
                                            node_a_id=node_a["node_id"],
                                            node_b_id=node_b["node_id"],
                                            rel_id=f"sim_{node_a['node_id']}_{node_b['node_id']}",
                                            rel_type=relationship_type,
                                            similarity=similarity,
                                            timestamp=datetime.utcnow().isoformat()
                                        )
                                        
                                        batch_relationships += 1
                                        
                                except Exception as e:
                                    error_msg = f"Failed to create similarity relationship between {node_a['node_id']} and {node_b['node_id']}: {str(e)}"
                                    results["errors"].append(error_msg)
                    
                    results["nodes_processed"] += len(batch)
                    results["relationships_created"] += batch_relationships
                    results["batches_processed"] += 1
                    
                    logger.debug(
                        f"Batch {results['batches_processed']} completed",
                        nodes_in_batch=len(batch),
                        relationships_created=batch_relationships
                    )
                
                logger.info(
                    "Similarity relationship creation completed",
                    total_nodes=len(nodes),
                    relationships_created=results["relationships_created"],
                    errors=len(results["errors"])
                )
                
        except Exception as e:
            error_msg = f"Failed to create similarity relationships: {str(e)}"
            logger.error(error_msg, error=str(e))
            results["errors"].append(error_msg)
            raise VectorOperationError(error_msg) from e
        
        return results
    
    def _calculate_cosine_similarity(self, vector_a: List[float], vector_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Convert to numpy arrays
            a = np.array(vector_a)
            b = np.array(vector_b)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, float(similarity)))
            
        except Exception as e:
            logger.warning(f"Failed to calculate cosine similarity: {str(e)}")
            return 0.0
    
    async def get_vector_statistics(self) -> Dict[str, Any]:
        """Get vector embedding statistics.
        
        Returns:
            Dict with vector statistics
        """
        logger.info("Collecting vector statistics")
        
        try:
            async with self.driver.session() as session:
                # Count nodes with embeddings by type
                embedding_counts = {}
                for node_type in NodeType:
                    # Content embeddings
                    result = await session.run(
                        f"MATCH (n:{node_type.value}) WHERE n.content_embedding IS NOT NULL RETURN count(n) as count"
                    )
                    record = await result.single()
                    content_count = record["count"] if record else 0
                    
                    # Title embeddings
                    result = await session.run(
                        f"MATCH (n:{node_type.value}) WHERE n.title_embedding IS NOT NULL RETURN count(n) as count"
                    )
                    record = await result.single()
                    title_count = record["count"] if record else 0
                    
                    embedding_counts[node_type.value] = {
                        "content_embeddings": content_count,
                        "title_embeddings": title_count,
                        "total_embeddings": content_count + title_count
                    }
                
                # Similarity relationships count
                result = await session.run(
                    "MATCH ()-[r:SIMILAR_TO]-() RETURN count(r) as count"
                )
                record = await result.single()
                similarity_relationships = record["count"] if record else 0
                
                # Vector index status
                result = await session.run("SHOW INDEXES")
                indexes = [record["name"] async for record in result]
                vector_indexes = [idx for idx in indexes if "embedding" in idx.lower()]
                
                statistics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "embedding_counts": embedding_counts,
                    "similarity_relationships": similarity_relationships,
                    "vector_indexes": {
                        "total": len(vector_indexes),
                        "indexes": vector_indexes
                    },
                    "total_nodes_with_embeddings": sum([
                        counts["total_embeddings"] 
                        for counts in embedding_counts.values()
                    ])
                }
                
                logger.info(
                    "Vector statistics collected",
                    total_embeddings=statistics["total_nodes_with_embeddings"],
                    similarity_relationships=similarity_relationships,
                    vector_indexes=len(vector_indexes)
                )
                
                return statistics
                
        except Exception as e:
            error_msg = f"Failed to collect vector statistics: {str(e)}"
            logger.error(error_msg, error=str(e))
            raise VectorOperationError(error_msg) from e