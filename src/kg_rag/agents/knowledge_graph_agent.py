"""
Knowledge Graph Agent for specialized graph operations.

Provides domain-specific knowledge graph operations, entity resolution,
relationship analysis, and semantic search capabilities.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import structlog
from pydantic import BaseModel, Field
from neo4j import AsyncDriver

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import KnowledgeGraphError
from kg_rag.graph_schema import (
    GraphSchemaManager, VectorGraphOperations, GraphQueryBuilder, SchemaValidator
)
from kg_rag.graph_schema.node_models import NodeType, NodeFactory
from kg_rag.graph_schema.relationship_models import RelationshipType, RelationshipFactory

logger = structlog.get_logger(__name__)


class EntityResolutionRequest(BaseModel):
    """Request for entity resolution."""
    
    entity_text: str = Field(..., description="Entity text to resolve")
    context: Optional[str] = Field(None, description="Context for disambiguation")
    entity_type: Optional[str] = Field(None, description="Expected entity type")
    similarity_threshold: float = Field(default=0.8, description="Similarity threshold for matching")
    max_candidates: int = Field(default=5, description="Maximum candidate entities")


class EntityResolutionResult(BaseModel):
    """Result of entity resolution."""
    
    original_text: str = Field(..., description="Original entity text")
    resolved_entity: Optional[Dict[str, Any]] = Field(None, description="Resolved entity")
    candidates: List[Dict[str, Any]] = Field(default_factory=list, description="Candidate entities")
    confidence_score: float = Field(default=0.0, description="Resolution confidence")
    resolution_method: str = Field(default="", description="Method used for resolution")


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""
    
    query: str = Field(..., description="Search query")
    search_types: List[NodeType] = Field(default_factory=list, description="Node types to search")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Additional filters")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    limit: int = Field(default=20, description="Maximum results")
    include_paths: bool = Field(default=False, description="Include relationship paths")
    
    class Config:
        use_enum_values = True


class RelationshipAnalysisRequest(BaseModel):
    """Request for relationship analysis."""
    
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: Optional[str] = Field(None, description="Target entity ID (optional)")
    relationship_types: List[RelationshipType] = Field(default_factory=list, description="Relationship types to analyze")
    max_hops: int = Field(default=3, description="Maximum relationship hops")
    include_weights: bool = Field(default=True, description="Include relationship weights")
    
    class Config:
        use_enum_values = True


class KnowledgeGraphAgent:
    """Specialized agent for knowledge graph operations."""
    
    def __init__(self, agent_id: str, neo4j_driver: AsyncDriver):
        """Initialize knowledge graph agent.
        
        Args:
            agent_id: Unique agent identifier
            neo4j_driver: Neo4j database driver
        """
        self.agent_id = agent_id
        self.driver = neo4j_driver
        
        # Initialize graph components
        self.schema_manager = GraphSchemaManager(neo4j_driver)
        self.vector_ops = VectorGraphOperations(neo4j_driver)
        self.query_builder = GraphQueryBuilder(neo4j_driver)
        self.schema_validator = SchemaValidator(neo4j_driver)
        
        # Internal state
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
        self._path_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Knowledge Graph agent initialized", agent_id=agent_id)
    
    async def resolve_entity(self, request: EntityResolutionRequest) -> EntityResolutionResult:
        """Resolve entity to canonical form in knowledge graph.
        
        Args:
            request: Entity resolution request
            
        Returns:
            Entity resolution result
        """
        logger.info(
            "Resolving entity",
            agent_id=self.agent_id,
            entity_text=request.entity_text,
            entity_type=request.entity_type
        )
        
        try:
            # Step 1: Exact name matching
            exact_match = await self._exact_entity_match(request.entity_text, request.entity_type)
            if exact_match:
                return EntityResolutionResult(
                    original_text=request.entity_text,
                    resolved_entity=exact_match,
                    candidates=[exact_match],
                    confidence_score=1.0,
                    resolution_method="exact_match"
                )
            
            # Step 2: Alias matching
            alias_match = await self._alias_entity_match(request.entity_text, request.entity_type)
            if alias_match:
                return EntityResolutionResult(
                    original_text=request.entity_text,
                    resolved_entity=alias_match,
                    candidates=[alias_match],
                    confidence_score=0.95,
                    resolution_method="alias_match"
                )
            
            # Step 3: Fuzzy matching with embeddings
            fuzzy_candidates = await self._fuzzy_entity_match(
                request.entity_text,
                request.entity_type,
                request.similarity_threshold,
                request.max_candidates
            )
            
            if fuzzy_candidates:
                best_candidate = fuzzy_candidates[0]
                confidence = best_candidate.get("similarity_score", 0.0)
                
                # Resolve if confidence is high enough
                resolved_entity = best_candidate if confidence >= request.similarity_threshold else None
                
                return EntityResolutionResult(
                    original_text=request.entity_text,
                    resolved_entity=resolved_entity,
                    candidates=fuzzy_candidates,
                    confidence_score=confidence,
                    resolution_method="fuzzy_match"
                )
            
            # Step 4: Context-based resolution
            if request.context:
                context_candidates = await self._context_entity_match(
                    request.entity_text,
                    request.context,
                    request.entity_type
                )
                
                if context_candidates:
                    return EntityResolutionResult(
                        original_text=request.entity_text,
                        resolved_entity=context_candidates[0],
                        candidates=context_candidates,
                        confidence_score=context_candidates[0].get("confidence", 0.7),
                        resolution_method="context_match"
                    )
            
            # No resolution found
            return EntityResolutionResult(
                original_text=request.entity_text,
                resolved_entity=None,
                candidates=[],
                confidence_score=0.0,
                resolution_method="no_match"
            )
            
        except Exception as e:
            error_msg = f"Entity resolution failed: {str(e)}"
            logger.error(error_msg, agent_id=self.agent_id, entity_text=request.entity_text)
            raise KnowledgeGraphError(error_msg) from e
    
    async def semantic_search(self, request: SemanticSearchRequest) -> List[Dict[str, Any]]:
        """Perform semantic search across the knowledge graph.
        
        Args:
            request: Semantic search request
            
        Returns:
            List of search results with relevance scores
        """
        logger.info(
            "Performing semantic search",
            agent_id=self.agent_id,
            query=request.query[:100],
            search_types=request.search_types,
            limit=request.limit
        )
        
        try:
            # Generate query embedding
            query_embedding = await self._generate_query_embedding(request.query)
            
            # Perform vector similarity search
            vector_results = await self.vector_ops.vector_similarity_search(
                query_vector=query_embedding,
                node_types=request.search_types or None,
                limit=request.limit,
                similarity_threshold=request.similarity_threshold,
                filters=request.filters
            )
            
            search_results = []
            
            for result in vector_results:
                search_result = {
                    "node_id": result["node_id"],
                    "node_type": result["node_type"],
                    "title": result["title"],
                    "description": result.get("description", ""),
                    "similarity_score": result["similarity_score"],
                    "metadata": result.get("node_properties", {})
                }
                
                # Add relationship paths if requested
                if request.include_paths:
                    paths = await self._find_relationship_paths(
                        result["node_id"],
                        max_hops=2,
                        limit=3
                    )
                    search_result["relationship_paths"] = paths
                
                search_results.append(search_result)
            
            logger.info(
                "Semantic search completed",
                agent_id=self.agent_id,
                results_count=len(search_results),
                avg_similarity=sum([r["similarity_score"] for r in search_results]) / len(search_results) if search_results else 0
            )
            
            return search_results
            
        except Exception as e:
            error_msg = f"Semantic search failed: {str(e)}"
            logger.error(error_msg, agent_id=self.agent_id, query=request.query)
            raise KnowledgeGraphError(error_msg) from e
    
    async def analyze_relationships(self, request: RelationshipAnalysisRequest) -> Dict[str, Any]:
        """Analyze relationships between entities in the knowledge graph.
        
        Args:
            request: Relationship analysis request
            
        Returns:
            Relationship analysis results
        """
        logger.info(
            "Analyzing relationships",
            agent_id=self.agent_id,
            source_entity=request.source_entity_id,
            target_entity=request.target_entity_id,
            max_hops=request.max_hops
        )
        
        try:
            analysis_results = {
                "source_entity_id": request.source_entity_id,
                "target_entity_id": request.target_entity_id,
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "direct_relationships": [],
                "indirect_relationships": [],
                "relationship_paths": [],
                "relationship_strength": 0.0,
                "common_neighbors": [],
                "centrality_metrics": {}
            }
            
            # Analyze direct relationships
            direct_rels = await self._analyze_direct_relationships(
                request.source_entity_id,
                request.target_entity_id,
                request.relationship_types
            )
            analysis_results["direct_relationships"] = direct_rels
            
            # Analyze indirect relationships if target specified
            if request.target_entity_id:
                indirect_rels = await self._analyze_indirect_relationships(
                    request.source_entity_id,
                    request.target_entity_id,
                    request.max_hops,
                    request.relationship_types
                )
                analysis_results["indirect_relationships"] = indirect_rels
                
                # Find shortest paths
                paths = await self._find_shortest_paths(
                    request.source_entity_id,
                    request.target_entity_id,
                    request.max_hops
                )
                analysis_results["relationship_paths"] = paths
                
                # Calculate relationship strength
                strength = self._calculate_relationship_strength(direct_rels, indirect_rels, paths)
                analysis_results["relationship_strength"] = strength
                
                # Find common neighbors
                common_neighbors = await self._find_common_neighbors(
                    request.source_entity_id,
                    request.target_entity_id
                )
                analysis_results["common_neighbors"] = common_neighbors
            
            # Calculate centrality metrics for source entity
            centrality = await self._calculate_centrality_metrics(
                request.source_entity_id,
                request.relationship_types
            )
            analysis_results["centrality_metrics"] = centrality
            
            logger.info(
                "Relationship analysis completed",
                agent_id=self.agent_id,
                direct_relationships=len(analysis_results["direct_relationships"]),
                indirect_relationships=len(analysis_results["indirect_relationships"]),
                relationship_strength=analysis_results["relationship_strength"]
            )
            
            return analysis_results
            
        except Exception as e:
            error_msg = f"Relationship analysis failed: {str(e)}"
            logger.error(error_msg, agent_id=self.agent_id, source_entity=request.source_entity_id)
            raise KnowledgeGraphError(error_msg) from e
    
    async def discover_patterns(
        self,
        pattern_type: str,
        node_types: List[NodeType],
        min_frequency: int = 3,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """Discover patterns in the knowledge graph.
        
        Args:
            pattern_type: Type of pattern to discover (motifs, clusters, anomalies)
            node_types: Node types to consider
            min_frequency: Minimum pattern frequency
            max_results: Maximum results to return
            
        Returns:
            List of discovered patterns
        """
        logger.info(
            "Discovering patterns",
            agent_id=self.agent_id,
            pattern_type=pattern_type,
            node_types=node_types,
            min_frequency=min_frequency
        )
        
        try:
            if pattern_type == "motifs":
                return await self._discover_graph_motifs(node_types, min_frequency, max_results)
            elif pattern_type == "clusters":
                return await self._discover_node_clusters(node_types, min_frequency, max_results)
            elif pattern_type == "anomalies":
                return await self._discover_anomalies(node_types, max_results)
            elif pattern_type == "communities":
                return await self._discover_communities(node_types, max_results)
            else:
                raise ValueError(f"Unknown pattern type: {pattern_type}")
                
        except Exception as e:
            error_msg = f"Pattern discovery failed: {str(e)}"
            logger.error(error_msg, agent_id=self.agent_id, pattern_type=pattern_type)
            raise KnowledgeGraphError(error_msg) from e
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics.
        
        Returns:
            Graph statistics and metrics
        """
        try:
            # Get schema statistics
            schema_stats = await self.schema_manager.get_schema_statistics()
            
            # Get vector statistics
            vector_stats = await self.vector_ops.get_vector_statistics()
            
            # Calculate additional metrics
            additional_metrics = await self._calculate_additional_metrics()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "schema_statistics": schema_stats,
                "vector_statistics": vector_stats,
                "graph_metrics": additional_metrics,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            error_msg = f"Failed to get graph statistics: {str(e)}"
            logger.error(error_msg, agent_id=self.agent_id)
            raise KnowledgeGraphError(error_msg) from e
    
    # Private helper methods
    
    async def _exact_entity_match(self, entity_text: str, entity_type: Optional[str]) -> Optional[Dict[str, Any]]:
        """Find exact entity match by canonical name."""
        try:
            query_builder = self.query_builder.match_node(NodeType.ENTITY, variable="entity")
            query_builder = query_builder.where_property("entity", "canonical_name", entity_text)
            
            if entity_type:
                query_builder = query_builder.where_property("entity", "entity_type", entity_type)
            
            results = await query_builder.return_nodes("entity").limit(1).execute()
            
            if results:
                entity = results[0]["entity"]
                return {
                    "node_id": entity["node_id"],
                    "canonical_name": entity["canonical_name"],
                    "entity_type": entity["entity_type"],
                    "confidence": 1.0,
                    "match_type": "exact"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Exact entity match failed: {str(e)}", agent_id=self.agent_id)
            return None
    
    async def _alias_entity_match(self, entity_text: str, entity_type: Optional[str]) -> Optional[Dict[str, Any]]:
        """Find entity match by alias."""
        try:
            query_builder = (self.query_builder
                .match_node(NodeType.ENTITY, variable="entity")
                .where(f"'{entity_text}' IN entity.aliases"))
            
            if entity_type:
                query_builder = query_builder.where_property("entity", "entity_type", entity_type)
            
            results = await query_builder.return_nodes("entity").limit(1).execute()
            
            if results:
                entity = results[0]["entity"]
                return {
                    "node_id": entity["node_id"],
                    "canonical_name": entity["canonical_name"],
                    "entity_type": entity["entity_type"],
                    "confidence": 0.95,
                    "match_type": "alias"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Alias entity match failed: {str(e)}", agent_id=self.agent_id)
            return None
    
    async def _fuzzy_entity_match(
        self,
        entity_text: str,
        entity_type: Optional[str],
        threshold: float,
        max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Find fuzzy entity matches using embeddings."""
        try:
            # Generate embedding for entity text
            entity_embedding = await self._generate_query_embedding(entity_text)
            
            # Search for similar entities
            filters = {"entity_type": entity_type} if entity_type else {}
            
            similar_entities = await self.vector_ops.vector_similarity_search(
                query_vector=entity_embedding,
                node_types=[NodeType.ENTITY],
                limit=max_candidates,
                similarity_threshold=threshold,
                filters=filters
            )
            
            candidates = []
            for entity in similar_entities:
                candidates.append({
                    "node_id": entity["node_id"],
                    "canonical_name": entity["title"],
                    "entity_type": entity.get("entity_type", "unknown"),
                    "similarity_score": entity["similarity_score"],
                    "confidence": entity["similarity_score"],
                    "match_type": "fuzzy"
                })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Fuzzy entity match failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _context_entity_match(
        self,
        entity_text: str,
        context: str,
        entity_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Find entity matches using context."""
        try:
            # This is a simplified context matching
            # In production, use more sophisticated NLP techniques
            
            context_keywords = context.lower().split()
            
            # Find entities that have context overlap
            query_builder = self.query_builder.match_node(NodeType.ENTITY, variable="entity")
            
            # Look for entities with context that matches
            context_conditions = []
            for keyword in context_keywords[:3]:  # Limit to first 3 keywords
                context_conditions.append(f"entity.context CONTAINS '{keyword}'")
            
            if context_conditions:
                query_builder = query_builder.where(" OR ".join(context_conditions))
            
            if entity_type:
                query_builder = query_builder.where_property("entity", "entity_type", entity_type)
            
            # Also check if entity name is similar to the text
            query_builder = query_builder.where_property("entity", "canonical_name", entity_text, "contains")
            
            results = await query_builder.return_nodes("entity").limit(3).execute()
            
            candidates = []
            for result in results:
                entity = result["entity"]
                candidates.append({
                    "node_id": entity["node_id"],
                    "canonical_name": entity["canonical_name"],
                    "entity_type": entity["entity_type"],
                    "confidence": 0.7,  # Context-based confidence
                    "match_type": "context"
                })
            
            return candidates
            
        except Exception as e:
            logger.error(f"Context entity match failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for text (placeholder)."""
        # Placeholder implementation
        settings = get_settings()
        embedding_dim = settings.ai_models.embedding_dimension
        
        import random
        random.seed(hash(text) % (2**32))
        return [random.random() for _ in range(embedding_dim)]
    
    async def _find_relationship_paths(
        self,
        node_id: str,
        max_hops: int = 2,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find relationship paths from a node."""
        try:
            # Find paths of different lengths
            paths = []
            
            for hop_count in range(1, max_hops + 1):
                hop_results = await (self.query_builder
                    .match_path(f"(start {{node_id: '{node_id}'}})-[*{hop_count}]-(end)")
                    .return_custom([
                        "start.node_id as start_id",
                        "start.title as start_title",
                        "end.node_id as end_id", 
                        "end.title as end_title",
                        f"{hop_count} as path_length"
                    ])
                    .limit(limit)
                    .execute())
                
                for result in hop_results:
                    paths.append({
                        "start_node": {
                            "node_id": result["start_id"],
                            "title": result["start_title"]
                        },
                        "end_node": {
                            "node_id": result["end_id"],
                            "title": result["end_title"]
                        },
                        "path_length": result["path_length"]
                    })
            
            return paths[:limit]
            
        except Exception as e:
            logger.error(f"Find relationship paths failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _analyze_direct_relationships(
        self,
        source_id: str,
        target_id: Optional[str],
        relationship_types: List[RelationshipType]
    ) -> List[Dict[str, Any]]:
        """Analyze direct relationships."""
        try:
            query_builder = (self.query_builder
                .match_node(properties={"node_id": source_id}, variable="source")
                .match_relationship("source", "target", relationship_types or None, relationship_var="rel")
                .match_node(variable="target"))
            
            if target_id:
                query_builder = query_builder.where_property("target", "node_id", target_id)
            
            results = await (query_builder
                .return_custom([
                    "rel.relationship_type as rel_type",
                    "rel.weight as weight",
                    "rel.confidence as confidence",
                    "target.node_id as target_id",
                    "target.title as target_title"
                ])
                .execute())
            
            direct_relationships = []
            for result in results:
                direct_relationships.append({
                    "relationship_type": result["rel_type"],
                    "weight": result.get("weight", 0.0),
                    "confidence": result.get("confidence", 0.0),
                    "target_node": {
                        "node_id": result["target_id"],
                        "title": result["target_title"]
                    }
                })
            
            return direct_relationships
            
        except Exception as e:
            logger.error(f"Direct relationship analysis failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _analyze_indirect_relationships(
        self,
        source_id: str,
        target_id: str,
        max_hops: int,
        relationship_types: List[RelationshipType]
    ) -> List[Dict[str, Any]]:
        """Analyze indirect relationships."""
        try:
            # Find paths between source and target
            rel_filter = f":{RelationshipType.RELATED_TO.value}" if not relationship_types else ""
            
            results = await (self.query_builder
                .match_path(f"(source {{node_id: '{source_id}'}})-[*2..{max_hops}]-(target {{node_id: '{target_id}'}})")
                .return_custom([
                    "length(path) as path_length",
                    "relationships(path) as rels"
                ])
                .limit(10)
                .execute())
            
            indirect_relationships = []
            for result in results:
                path_info = {
                    "path_length": result["path_length"],
                    "relationship_count": len(result.get("rels", [])),
                    "path_strength": 1.0 / result["path_length"] if result["path_length"] > 0 else 0.0
                }
                indirect_relationships.append(path_info)
            
            return indirect_relationships
            
        except Exception as e:
            logger.error(f"Indirect relationship analysis failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _find_shortest_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between entities."""
        try:
            results = await (self.query_builder
                .match_path(f"path = shortestPath((source {{node_id: '{source_id}'}})-[*..{max_hops}]-(target {{node_id: '{target_id}'}}))")
                .return_custom([
                    "length(path) as path_length",
                    "nodes(path) as path_nodes",
                    "relationships(path) as path_rels"
                ])
                .limit(3)
                .execute())
            
            paths = []
            for result in results:
                path_info = {
                    "path_length": result["path_length"],
                    "node_count": len(result.get("path_nodes", [])),
                    "relationship_count": len(result.get("path_rels", []))
                }
                paths.append(path_info)
            
            return paths
            
        except Exception as e:
            logger.error(f"Shortest path finding failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    def _calculate_relationship_strength(
        self,
        direct_rels: List[Dict[str, Any]],
        indirect_rels: List[Dict[str, Any]],
        paths: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall relationship strength."""
        strength = 0.0
        
        # Direct relationship strength
        if direct_rels:
            direct_strength = sum([rel.get("weight", 0.0) * rel.get("confidence", 0.0) for rel in direct_rels])
            strength += direct_strength * 0.7  # 70% weight for direct relationships
        
        # Indirect relationship strength
        if indirect_rels:
            indirect_strength = sum([rel.get("path_strength", 0.0) for rel in indirect_rels])
            strength += (indirect_strength / len(indirect_rels)) * 0.2  # 20% weight for indirect
        
        # Path diversity bonus
        if paths:
            unique_lengths = len(set([path["path_length"] for path in paths]))
            path_bonus = min(unique_lengths * 0.1, 0.1)  # 10% max bonus for path diversity
            strength += path_bonus
        
        return min(strength, 1.0)  # Cap at 1.0
    
    async def _find_common_neighbors(self, source_id: str, target_id: str) -> List[Dict[str, Any]]:
        """Find common neighbors between two entities."""
        try:
            results = await (self.query_builder
                .match_node(properties={"node_id": source_id}, variable="source")
                .match_relationship("source", "neighbor", relationship_var="rel1")
                .match_node(variable="neighbor")
                .match_relationship("neighbor", "target", relationship_var="rel2")
                .match_node(properties={"node_id": target_id}, variable="target")
                .return_custom([
                    "neighbor.node_id as neighbor_id",
                    "neighbor.title as neighbor_title",
                    "rel1.weight as source_weight",
                    "rel2.weight as target_weight"
                ])
                .limit(10)
                .execute())
            
            common_neighbors = []
            for result in results:
                common_neighbors.append({
                    "neighbor_id": result["neighbor_id"],
                    "neighbor_title": result["neighbor_title"],
                    "source_weight": result.get("source_weight", 0.0),
                    "target_weight": result.get("target_weight", 0.0)
                })
            
            return common_neighbors
            
        except Exception as e:
            logger.error(f"Common neighbors finding failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _calculate_centrality_metrics(
        self,
        node_id: str,
        relationship_types: List[RelationshipType]
    ) -> Dict[str, float]:
        """Calculate centrality metrics for a node."""
        try:
            # Degree centrality
            degree_result = await (self.query_builder
                .match_node(properties={"node_id": node_id}, variable="node")
                .match_relationship("node", "neighbor", relationship_types or None)
                .return_custom(["count(neighbor) as degree"])
                .execute())
            
            degree = degree_result[0]["degree"] if degree_result else 0
            
            # Simplified betweenness centrality approximation
            # In production, use graph algorithms library
            betweenness = min(degree / 100.0, 1.0)  # Simplified approximation
            
            # Closeness centrality approximation
            closeness = 1.0 / (degree + 1) if degree > 0 else 0.0
            
            return {
                "degree_centrality": float(degree),
                "betweenness_centrality": betweenness,
                "closeness_centrality": closeness,
                "normalized_degree": min(degree / 50.0, 1.0)  # Normalize assuming max degree of 50
            }
            
        except Exception as e:
            logger.error(f"Centrality calculation failed: {str(e)}", agent_id=self.agent_id)
            return {"degree_centrality": 0.0, "betweenness_centrality": 0.0, "closeness_centrality": 0.0}
    
    async def _discover_graph_motifs(
        self,
        node_types: List[NodeType],
        min_frequency: int,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Discover common graph motifs."""
        # Simplified motif discovery - in production, use specialized algorithms
        try:
            # Look for triangle motifs (3-node cycles)
            motifs = []
            
            for node_type in node_types:
                results = await (self.query_builder
                    .match_node(node_type, variable="a")
                    .match_relationship("a", "b", relationship_var="rel1")
                    .match_node(variable="b")
                    .match_relationship("b", "c", relationship_var="rel2")
                    .match_node(variable="c")
                    .match_relationship("c", "a", relationship_var="rel3")
                    .return_custom([
                        "rel1.relationship_type as type1",
                        "rel2.relationship_type as type2",
                        "rel3.relationship_type as type3",
                        "count(*) as frequency"
                    ])
                    .limit(max_results)
                    .execute())
                
                for result in results:
                    if result["frequency"] >= min_frequency:
                        motifs.append({
                            "motif_type": "triangle",
                            "node_type": node_type.value,
                            "relationship_pattern": [result["type1"], result["type2"], result["type3"]],
                            "frequency": result["frequency"]
                        })
            
            return motifs[:max_results]
            
        except Exception as e:
            logger.error(f"Motif discovery failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _discover_node_clusters(
        self,
        node_types: List[NodeType],
        min_frequency: int,
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Discover node clusters."""
        # Simplified clustering - in production, use proper clustering algorithms
        try:
            clusters = []
            
            for node_type in node_types:
                # Find highly connected components
                results = await (self.query_builder
                    .match_node(node_type, variable="center")
                    .match_relationship("center", "neighbor", relationship_var="rel")
                    .return_custom([
                        "center.node_id as center_id",
                        "center.title as center_title",
                        "count(neighbor) as neighbor_count"
                    ])
                    .order_by("neighbor_count", "DESC")
                    .limit(max_results)
                    .execute())
                
                for result in results:
                    if result["neighbor_count"] >= min_frequency:
                        clusters.append({
                            "cluster_type": "hub",
                            "center_node": {
                                "node_id": result["center_id"],
                                "title": result["center_title"]
                            },
                            "size": result["neighbor_count"],
                            "node_type": node_type.value
                        })
            
            return clusters
            
        except Exception as e:
            logger.error(f"Cluster discovery failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _discover_anomalies(
        self,
        node_types: List[NodeType],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Discover anomalies in the graph."""
        try:
            anomalies = []
            
            for node_type in node_types:
                # Find nodes with unusual degree (very high or very low)
                results = await (self.query_builder
                    .match_node(node_type, variable="node")
                    .match_relationship("node", "neighbor", relationship_var="rel")
                    .return_custom([
                        "node.node_id as node_id",
                        "node.title as title",
                        "count(neighbor) as degree"
                    ])
                    .execute())
                
                if results:
                    degrees = [r["degree"] for r in results]
                    avg_degree = sum(degrees) / len(degrees)
                    std_dev = (sum([(d - avg_degree) ** 2 for d in degrees]) / len(degrees)) ** 0.5
                    
                    for result in results:
                        degree = result["degree"]
                        z_score = abs(degree - avg_degree) / std_dev if std_dev > 0 else 0
                        
                        if z_score > 2.0:  # More than 2 standard deviations
                            anomalies.append({
                                "anomaly_type": "unusual_degree",
                                "node_id": result["node_id"],
                                "title": result["title"],
                                "degree": degree,
                                "z_score": z_score,
                                "node_type": node_type.value
                            })
            
            # Sort by z_score and limit
            anomalies.sort(key=lambda x: x["z_score"], reverse=True)
            return anomalies[:max_results]
            
        except Exception as e:
            logger.error(f"Anomaly discovery failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _discover_communities(
        self,
        node_types: List[NodeType],
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Discover communities in the graph."""
        # Simplified community detection
        try:
            communities = []
            
            # This is a placeholder for community detection
            # In production, use algorithms like Louvain or Leiden
            
            for node_type in node_types:
                results = await (self.query_builder
                    .match_node(node_type, variable="node")
                    .return_custom([
                        "node.node_id as node_id",
                        "node.title as title",
                        "node.categories as categories"
                    ])
                    .limit(max_results)
                    .execute())
                
                # Group by categories as a simple community proxy
                category_groups = {}
                for result in results:
                    categories = result.get("categories", [])
                    for category in categories:
                        if category not in category_groups:
                            category_groups[category] = []
                        category_groups[category].append({
                            "node_id": result["node_id"],
                            "title": result["title"]
                        })
                
                for category, nodes in category_groups.items():
                    if len(nodes) >= 3:  # Minimum community size
                        communities.append({
                            "community_type": "category_based",
                            "community_id": category,
                            "size": len(nodes),
                            "nodes": nodes[:10],  # Limit nodes shown
                            "node_type": node_type.value
                        })
            
            return communities[:max_results]
            
        except Exception as e:
            logger.error(f"Community discovery failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _calculate_additional_metrics(self) -> Dict[str, Any]:
        """Calculate additional graph metrics."""
        try:
            # Graph density
            node_count_result = await (self.query_builder
                .match_node(variable="n")
                .return_custom(["count(n) as node_count"])
                .execute())
            
            rel_count_result = await (self.query_builder
                .match_path("()-[r]-()")
                .return_custom(["count(r) as rel_count"])
                .execute())
            
            node_count = node_count_result[0]["node_count"] if node_count_result else 0
            rel_count = rel_count_result[0]["rel_count"] if rel_count_result else 0
            
            # Calculate density
            max_possible_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
            density = rel_count / max_possible_edges if max_possible_edges > 0 else 0
            
            # Average degree
            avg_degree = (2 * rel_count) / node_count if node_count > 0 else 0
            
            return {
                "graph_density": density,
                "average_degree": avg_degree,
                "clustering_coefficient": 0.0,  # Placeholder - would need complex calculation
                "diameter": 0,  # Placeholder - would need complex calculation
                "connected_components": 1  # Placeholder - assume single component
            }
            
        except Exception as e:
            logger.error(f"Additional metrics calculation failed: {str(e)}", agent_id=self.agent_id)
            return {
                "graph_density": 0.0,
                "average_degree": 0.0,
                "clustering_coefficient": 0.0,
                "diameter": 0,
                "connected_components": 0
            }