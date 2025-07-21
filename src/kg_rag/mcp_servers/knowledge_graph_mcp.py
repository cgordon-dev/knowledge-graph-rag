"""
Knowledge Graph MCP Server for Neo4j operations.

Provides tools for graph database operations, relationship management,
and hybrid graph-vector search capabilities.
"""

import json
from typing import Any, Dict, List, Optional, Union

from neo4j import AsyncGraphDatabase, AsyncDriver, Record
import structlog

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import GraphDatabaseError, GraphQueryError, VectorIndexError
from kg_rag.mcp_servers.base_mcp import BaseMCPServer


class KnowledgeGraphMCP(BaseMCPServer):
    """MCP server for Knowledge Graph operations using Neo4j."""
    
    def __init__(self):
        """Initialize Knowledge Graph MCP server."""
        settings = get_settings()
        super().__init__(
            name="knowledge_graph_mcp",
            description="Neo4j Knowledge Graph operations with vector support",
            port=settings.mcp_servers.knowledge_graph_mcp_port
        )
        self.driver: Optional[AsyncDriver] = None
    
    async def _initialize(self) -> None:
        """Initialize Neo4j connection."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.database.neo4j_uri,
                auth=(
                    self.settings.database.neo4j_user,
                    self.settings.database.neo4j_password
                ),
                max_connection_lifetime=self.settings.database.neo4j_max_connection_lifetime,
                max_connection_pool_size=self.settings.database.neo4j_max_connection_pool_size,
                connection_timeout=self.settings.database.neo4j_connection_timeout
            )
            
            # Verify connection
            await self.driver.verify_connectivity()
            
            self.logger.info(
                "Neo4j connection established",
                uri=self.settings.database.neo4j_uri,
                database=self.settings.database.neo4j_database
            )
            
        except Exception as e:
            raise GraphDatabaseError(f"Failed to connect to Neo4j: {e}")
    
    async def _cleanup(self) -> None:
        """Cleanup Neo4j connection."""
        if self.driver:
            await self.driver.close()
            self.driver = None
    
    async def _custom_health_check(self) -> Dict[str, Any]:
        """Custom health check for Neo4j connectivity."""
        try:
            if not self.driver:
                return {"neo4j_status": "disconnected"}
            
            await self.driver.verify_connectivity()
            
            # Get basic database info
            async with self.driver.session() as session:
                result = await session.run("CALL dbms.components() YIELD name, versions")
                components = await result.data()
                
                # Get node and relationship counts
                count_result = await session.run("""
                    MATCH (n) 
                    RETURN count(n) as node_count
                """)
                node_count = (await count_result.single())["node_count"]
                
                rel_result = await session.run("""
                    MATCH ()-[r]->() 
                    RETURN count(r) as rel_count
                """)
                rel_count = (await rel_result.single())["rel_count"]
                
                return {
                    "neo4j_status": "connected",
                    "components": components,
                    "node_count": node_count,
                    "relationship_count": rel_count
                }
                
        except Exception as e:
            return {
                "neo4j_status": "error",
                "error": str(e)
            }
    
    def register_tools(self) -> None:
        """Register Knowledge Graph MCP tools."""
        
        @self.tool(
            "execute_cypher_query",
            "Execute Cypher query against Neo4j knowledge graph",
            {
                "query": {"type": "string", "required": True, "description": "Cypher query to execute"},
                "parameters": {"type": "object", "required": False, "description": "Query parameters"}
            }
        )
        async def execute_cypher_query(
            query: str, 
            parameters: Optional[Dict[str, Any]] = None
        ) -> List[Dict[str, Any]]:
            """Execute Cypher query against Neo4j knowledge graph."""
            if not self.driver:
                raise GraphDatabaseError("Neo4j driver not initialized")
            
            try:
                async with self.driver.session() as session:
                    result = await session.run(query, parameters or {})
                    records = await result.data()
                    
                    self.logger.debug(
                        "Cypher query executed",
                        query=query[:100] + "..." if len(query) > 100 else query,
                        result_count=len(records)
                    )
                    
                    return records
                    
            except Exception as e:
                raise GraphQueryError(
                    f"Cypher query execution failed: {e}",
                    cypher_query=query,
                    parameters=parameters
                )
        
        @self.tool(
            "hybrid_search",
            "Perform hybrid graph + vector search",
            {
                "text_query": {"type": "string", "required": False, "description": "Text query for vector search"},
                "vector_query": {"type": "array", "required": False, "description": "Vector for similarity search"},
                "graph_depth": {"type": "integer", "required": False, "description": "Graph traversal depth"},
                "vector_threshold": {"type": "number", "required": False, "description": "Vector similarity threshold"},
                "limit": {"type": "integer", "required": False, "description": "Maximum results to return"}
            }
        )
        async def hybrid_search(
            text_query: Optional[str] = None,
            vector_query: Optional[List[float]] = None,
            graph_depth: int = 3,
            vector_threshold: float = 0.7,
            limit: int = 20
        ) -> Dict[str, Any]:
            """Perform hybrid graph + vector search."""
            if not text_query and not vector_query:
                raise GraphQueryError("Either text_query or vector_query must be provided", "")
            
            results = {"vector_results": [], "graph_results": []}
            
            try:
                # Vector similarity search
                if vector_query or text_query:
                    # If text_query provided, generate vector (this would integrate with embedding service)
                    search_vector = vector_query
                    if text_query and not vector_query:
                        # In real implementation, generate embedding here
                        self.logger.warning("Text to vector conversion not implemented, using placeholder")
                        search_vector = [0.0] * 1024  # Placeholder
                    
                    if search_vector:
                        vector_query_cypher = """
                        CALL db.index.vector.queryNodes($index_name, $limit, $vector)
                        YIELD node, score
                        WHERE score > $threshold
                        RETURN node, score
                        ORDER BY score DESC
                        """
                        
                        vector_results = await execute_cypher_query(
                            vector_query_cypher,
                            {
                                "index_name": "control_content_index",
                                "limit": limit,
                                "vector": search_vector,
                                "threshold": vector_threshold
                            }
                        )
                        results["vector_results"] = vector_results
                
                # Graph traversal for relationships
                if results["vector_results"]:
                    # Extract node IDs for graph traversal
                    node_ids = []
                    for result in results["vector_results"]:
                        if "node" in result:
                            node = result["node"]
                            if "control_id" in node:
                                node_ids.append(node["control_id"])
                    
                    if node_ids:
                        graph_query_cypher = """
                        MATCH (n:Control)
                        WHERE n.control_id IN $node_ids
                        MATCH path = (n)-[*1..$depth]-(related)
                        RETURN n, related, relationships(path) as path_rels,
                               reduce(score = 0, rel in relationships(path) | 
                                      score + coalesce(rel.strength, 0.5)) as path_score
                        ORDER BY path_score DESC
                        LIMIT $limit
                        """
                        
                        graph_results = await execute_cypher_query(
                            graph_query_cypher,
                            {
                                "node_ids": node_ids,
                                "depth": graph_depth,
                                "limit": limit
                            }
                        )
                        results["graph_results"] = graph_results
                
                # Calculate hybrid scores
                results["hybrid_scores"] = self._calculate_hybrid_scores(
                    results["vector_results"],
                    results["graph_results"]
                )
                
                return results
                
            except Exception as e:
                raise GraphQueryError(f"Hybrid search failed: {e}", "hybrid_search")
        
        @self.tool(
            "get_persona_recommendations",
            "Get personalized content recommendations for a persona",
            {
                "persona_id": {"type": "string", "required": True, "description": "Persona identifier"},
                "content_type": {"type": "string", "required": False, "description": "Type of content to recommend"},
                "limit": {"type": "integer", "required": False, "description": "Maximum recommendations"}
            }
        )
        async def get_persona_recommendations(
            persona_id: str,
            content_type: Optional[str] = None,
            limit: int = 10
        ) -> List[Dict[str, Any]]:
            """Get personalized content recommendations for a persona."""
            query = """
            MATCH (p:UserPersona {persona_id: $persona_id})
            MATCH (p)-[affinity:HAS_AFFINITY]->(content)
            WHERE ($content_type IS NULL OR content.type = $content_type)
            RETURN content, affinity.affinity_strength as relevance_score
            ORDER BY relevance_score DESC
            LIMIT $limit
            """
            
            return await execute_cypher_query(query, {
                "persona_id": persona_id,
                "content_type": content_type,
                "limit": limit
            })
        
        @self.tool(
            "update_persona_interaction",
            "Update persona interaction patterns for learning",
            {
                "persona_id": {"type": "string", "required": True, "description": "Persona identifier"},
                "content_id": {"type": "string", "required": True, "description": "Content identifier"},
                "interaction_type": {"type": "string", "required": True, "description": "Type of interaction"},
                "satisfaction_score": {"type": "number", "required": False, "description": "User satisfaction score"}
            }
        )
        async def update_persona_interaction(
            persona_id: str,
            content_id: str,
            interaction_type: str,
            satisfaction_score: Optional[float] = None
        ) -> Dict[str, Any]:
            """Update persona interaction patterns for learning."""
            query = """
            MATCH (p:UserPersona {persona_id: $persona_id})
            MATCH (c {id: $content_id})
            MERGE (p)-[interaction:HAS_AFFINITY]->(c)
            SET interaction.last_interaction = datetime(),
                interaction.interaction_count = coalesce(interaction.interaction_count, 0) + 1,
                interaction.interaction_type = $interaction_type
            WITH interaction
            WHERE $satisfaction_score IS NOT NULL
            SET interaction.satisfaction_score = $satisfaction_score
            RETURN interaction
            """
            
            result = await execute_cypher_query(query, {
                "persona_id": persona_id,
                "content_id": content_id,
                "interaction_type": interaction_type,
                "satisfaction_score": satisfaction_score
            })
            
            return {
                "updated": len(result) > 0,
                "interaction": result[0] if result else None
            }
        
        @self.tool(
            "create_vector_index",
            "Create vector index for a node property",
            {
                "index_name": {"type": "string", "required": True, "description": "Index name"},
                "node_label": {"type": "string", "required": True, "description": "Node label"},
                "property_name": {"type": "string", "required": True, "description": "Property name"},
                "vector_dimensions": {"type": "integer", "required": False, "description": "Vector dimensions"},
                "similarity_function": {"type": "string", "required": False, "description": "Similarity function"}
            }
        )
        async def create_vector_index(
            index_name: str,
            node_label: str,
            property_name: str,
            vector_dimensions: int = 1024,
            similarity_function: str = "cosine"
        ) -> Dict[str, Any]:
            """Create vector index for a node property."""
            try:
                query = f"""
                CREATE VECTOR INDEX {index_name} FOR (n:{node_label}) ON n.{property_name}
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dimensions},
                        `vector.similarity_function`: '{similarity_function}'
                    }}
                }}
                """
                
                await execute_cypher_query(query)
                
                return {
                    "index_name": index_name,
                    "node_label": node_label,
                    "property_name": property_name,
                    "vector_dimensions": vector_dimensions,
                    "similarity_function": similarity_function,
                    "created": True
                }
                
            except Exception as e:
                raise VectorIndexError(
                    f"Failed to create vector index: {e}",
                    index_name=index_name,
                    operation="create"
                )
        
        @self.tool(
            "get_graph_statistics",
            "Get comprehensive graph database statistics",
            {}
        )
        async def get_graph_statistics() -> Dict[str, Any]:
            """Get comprehensive graph database statistics."""
            try:
                # Get node counts by label
                node_stats_query = """
                MATCH (n)
                RETURN labels(n) as node_labels, count(n) as count
                ORDER BY count DESC
                """
                node_stats = await execute_cypher_query(node_stats_query)
                
                # Get relationship counts by type
                rel_stats_query = """
                MATCH ()-[r]->()
                RETURN type(r) as relationship_type, count(r) as count
                ORDER BY count DESC
                """
                rel_stats = await execute_cypher_query(rel_stats_query)
                
                # Get vector index information
                vector_index_query = """
                SHOW INDEXES
                YIELD name, type, labelsOrTypes, properties
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties
                """
                vector_indexes = await execute_cypher_query(vector_index_query)
                
                # Get database size information
                size_query = """
                CALL dbms.queryJmx('org.neo4j:instance=kernel#0,name=Store file sizes')
                YIELD attributes
                RETURN attributes.TotalStoreSize.value as total_size
                """
                size_info = await execute_cypher_query(size_query)
                
                return {
                    "node_statistics": node_stats,
                    "relationship_statistics": rel_stats,
                    "vector_indexes": vector_indexes,
                    "database_size": size_info[0]["total_size"] if size_info else None,
                    "timestamp": self.metrics.last_request_time.isoformat() if self.metrics.last_request_time else None
                }
                
            except Exception as e:
                self.logger.error("Failed to get graph statistics", error=str(e))
                return {"error": str(e)}
    
    def _calculate_hybrid_scores(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Calculate hybrid scores combining vector and graph results."""
        hybrid_scores = []
        
        # Create lookup for vector scores
        vector_scores = {}
        for result in vector_results:
            if "node" in result and "score" in result:
                node_id = result["node"].get("control_id") or result["node"].get("id")
                if node_id:
                    vector_scores[node_id] = result["score"]
        
        # Combine with graph scores
        for result in graph_results:
            node_id = None
            if "n" in result:
                node_id = result["n"].get("control_id") or result["n"].get("id")
            
            if node_id:
                vector_score = vector_scores.get(node_id, 0.0)
                graph_score = result.get("path_score", 0.0)
                
                # Weighted combination (60% vector, 40% graph)
                hybrid_score = (vector_score * 0.6) + (graph_score * 0.4)
                
                hybrid_scores.append({
                    "node_id": node_id,
                    "vector_score": vector_score,
                    "graph_score": graph_score,
                    "hybrid_score": hybrid_score,
                    "node_data": result.get("n", {}),
                    "related_nodes": result.get("related", [])
                })
        
        # Sort by hybrid score
        hybrid_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return hybrid_scores