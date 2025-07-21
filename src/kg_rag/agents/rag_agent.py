"""
RAG Agent Implementation for Knowledge Graph-RAG System.

Combines Google ADK with Neo4j vector graph schema and AI Digital Twins
to provide comprehensive Retrieval-Augmented Generation capabilities.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator
import uuid

import structlog
from pydantic import BaseModel, Field
from neo4j import AsyncDriver

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import RAGAgentError, ConfigurationError
from kg_rag.agents.adk_agent import ADKAgent, ADKConfiguration, ADKAgentResponse
from kg_rag.ai_twins.twin_orchestrator import TwinOrchestrator
from kg_rag.graph_schema import (
    GraphSchemaManager, VectorGraphOperations, GraphQueryBuilder
)
from kg_rag.graph_schema.node_models import NodeType

logger = structlog.get_logger(__name__)


class RAGConfiguration(BaseModel):
    """Configuration for RAG agent."""
    
    # Agent Configuration
    agent_name: str = Field(..., description="RAG agent name")
    description: str = Field(default="", description="Agent description")
    
    # ADK Configuration
    adk_config: ADKConfiguration = Field(..., description="Google ADK configuration")
    
    # Knowledge Graph Configuration
    retrieval_strategy: str = Field(default="hybrid", description="Retrieval strategy (vector, graph, hybrid)")
    max_retrieval_docs: int = Field(default=10, description="Maximum documents to retrieve")
    min_similarity_threshold: float = Field(default=0.7, description="Minimum similarity threshold")
    
    # AI Twins Configuration
    enable_expert_consultation: bool = Field(default=True, description="Enable expert twin consultation")
    enable_process_optimization: bool = Field(default=True, description="Enable process twin consultation")
    enable_persona_adaptation: bool = Field(default=True, description="Enable persona-based responses")
    
    # Response Configuration
    max_response_length: int = Field(default=2048, description="Maximum response length")
    include_sources: bool = Field(default=True, description="Include source citations")
    include_confidence: bool = Field(default=True, description="Include confidence scores")
    
    # Performance Configuration
    enable_parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    request_timeout_seconds: int = Field(default=30, description="Request timeout")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class RAGQuery(BaseModel):
    """RAG query with context and preferences."""
    
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique query ID")
    query: str = Field(..., description="User query")
    
    # Context
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    # Preferences
    preferred_sources: List[str] = Field(default_factory=list, description="Preferred source types")
    required_confidence: float = Field(default=0.5, description="Required confidence threshold")
    include_reasoning: bool = Field(default=False, description="Include reasoning in response")
    
    # Filters
    node_type_filters: List[NodeType] = Field(default_factory=list, description="Node types to search")
    date_range: Optional[Dict[str, str]] = Field(None, description="Date range filter")
    domain_filters: List[str] = Field(default_factory=list, description="Domain filters")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class RAGResponse(BaseModel):
    """Comprehensive RAG response."""
    
    query_id: str = Field(..., description="Original query ID")
    agent_id: str = Field(..., description="RAG agent ID")
    
    # Core Response
    response: str = Field(..., description="Generated response")
    confidence_score: float = Field(..., description="Overall confidence score")
    
    # Sources and Evidence
    knowledge_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Knowledge sources used")
    twin_insights: List[Dict[str, Any]] = Field(default_factory=list, description="AI Twin insights")
    reasoning_chain: List[str] = Field(default_factory=list, description="Reasoning steps")
    
    # Metadata
    processing_metrics: Dict[str, float] = Field(default_factory=dict, description="Processing metrics")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model information")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    # Quality Indicators
    source_coverage: float = Field(default=0.0, description="Source coverage score")
    factual_consistency: float = Field(default=0.0, description="Factual consistency score")
    relevance_score: float = Field(default=0.0, description="Relevance score")


class RAGAgent:
    """Comprehensive RAG agent with Knowledge Graph and AI Twins integration."""
    
    def __init__(
        self,
        agent_id: str,
        config: RAGConfiguration,
        neo4j_driver: AsyncDriver,
        twin_orchestrator: Optional[TwinOrchestrator] = None
    ):
        """Initialize RAG agent.
        
        Args:
            agent_id: Unique agent identifier
            config: RAG configuration
            neo4j_driver: Neo4j database driver
            twin_orchestrator: AI Digital Twins orchestrator
        """
        self.agent_id = agent_id
        self.config = config
        self.driver = neo4j_driver
        self.twin_orchestrator = twin_orchestrator
        
        # Initialize components
        self.schema_manager = GraphSchemaManager(neo4j_driver)
        self.vector_ops = VectorGraphOperations(neo4j_driver)
        self.query_builder = GraphQueryBuilder(neo4j_driver)
        
        # Initialize ADK agent
        self.adk_agent = ADKAgent(
            agent_id=f"{agent_id}_adk",
            config=config.adk_config,
            twin_orchestrator=twin_orchestrator,
            vector_operations=self.vector_ops,
            query_builder=self.query_builder
        )
        
        # Internal state
        self._initialized = False
        self._query_cache: Dict[str, RAGResponse] = {}
        
        logger.info(
            "RAG agent initialized",
            agent_id=agent_id,
            agent_name=config.agent_name,
            retrieval_strategy=config.retrieval_strategy
        )
    
    async def initialize(self) -> None:
        """Initialize RAG agent and all components."""
        if self._initialized:
            return
        
        try:
            # Initialize ADK agent
            await self.adk_agent.initialize()
            
            # Validate schema
            schema_validation = await self.schema_manager.validate_schema()
            if not schema_validation["is_valid"]:
                logger.warning(
                    "Schema validation issues detected",
                    agent_id=self.agent_id,
                    missing_constraints=len(schema_validation["constraints"]["missing"]),
                    missing_indexes=len(schema_validation["indexes"]["missing"])
                )
            
            self._initialized = True
            
            logger.info("RAG agent initialized successfully", agent_id=self.agent_id)
            
        except Exception as e:
            error_msg = f"Failed to initialize RAG agent: {str(e)}"
            logger.error(error_msg, agent_id=self.agent_id, error=str(e))
            raise RAGAgentError(error_msg) from e
    
    async def process_query(self, query: RAGQuery) -> RAGResponse:
        """Process a comprehensive RAG query.
        
        Args:
            query: RAG query with context and preferences
            
        Returns:
            Comprehensive RAG response
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        
        logger.info(
            "Processing RAG query",
            agent_id=self.agent_id,
            query_id=query.query_id,
            query_length=len(query.query),
            retrieval_strategy=self.config.retrieval_strategy
        )
        
        try:
            # Step 1: Enhanced knowledge retrieval
            retrieval_start = datetime.utcnow()
            knowledge_sources = await self._enhanced_knowledge_retrieval(query)
            retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000
            
            # Step 2: AI Twins consultation
            twins_start = datetime.utcnow()
            twin_insights = await self._comprehensive_twin_consultation(query, knowledge_sources)
            twins_time = (datetime.utcnow() - twins_start).total_seconds() * 1000
            
            # Step 3: Response generation with enhanced context
            generation_start = datetime.utcnow()
            adk_response = await self._generate_enhanced_response(
                query, knowledge_sources, twin_insights
            )
            generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000
            
            # Step 4: Quality assessment
            quality_start = datetime.utcnow()
            quality_metrics = await self._assess_response_quality(
                query, adk_response, knowledge_sources, twin_insights
            )
            quality_time = (datetime.utcnow() - quality_start).total_seconds() * 1000
            
            # Step 5: Construct comprehensive response
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            rag_response = RAGResponse(
                query_id=query.query_id,
                agent_id=self.agent_id,
                response=adk_response.response,
                confidence_score=quality_metrics["overall_confidence"],
                knowledge_sources=knowledge_sources,
                twin_insights=twin_insights,
                reasoning_chain=self._extract_reasoning_chain(adk_response, quality_metrics),
                processing_metrics={
                    "total_time_ms": total_time,
                    "retrieval_time_ms": retrieval_time,
                    "twins_time_ms": twins_time,
                    "generation_time_ms": generation_time,
                    "quality_time_ms": quality_time
                },
                model_info={
                    "adk_model": self.config.adk_config.model_name,
                    "total_tokens": adk_response.total_tokens,
                    "retrieval_docs": len(knowledge_sources),
                    "twin_consultations": len(twin_insights)
                },
                source_coverage=quality_metrics["source_coverage"],
                factual_consistency=quality_metrics["factual_consistency"],
                relevance_score=quality_metrics["relevance_score"]
            )
            
            # Cache response
            self._query_cache[query.query_id] = rag_response
            
            logger.info(
                "RAG query processed successfully",
                agent_id=self.agent_id,
                query_id=query.query_id,
                processing_time_ms=total_time,
                confidence_score=rag_response.confidence_score,
                knowledge_sources=len(knowledge_sources),
                twin_insights=len(twin_insights)
            )
            
            return rag_response
            
        except Exception as e:
            error_msg = f"Failed to process RAG query: {str(e)}"
            logger.error(
                error_msg,
                agent_id=self.agent_id,
                query_id=query.query_id,
                error=str(e)
            )
            raise RAGAgentError(error_msg) from e
    
    async def _enhanced_knowledge_retrieval(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Enhanced knowledge retrieval with multiple strategies.
        
        Args:
            query: RAG query
            
        Returns:
            List of enhanced knowledge sources
        """
        knowledge_sources = []
        
        try:
            # Generate query embedding (placeholder)
            query_embedding = await self._generate_query_embedding(query.query)
            
            # Strategy 1: Vector similarity search
            if self.config.retrieval_strategy in ["vector", "hybrid"]:
                vector_results = await self.vector_ops.vector_similarity_search(
                    query_vector=query_embedding,
                    node_types=query.node_type_filters or None,
                    limit=self.config.max_retrieval_docs,
                    similarity_threshold=query.required_confidence,
                    filters=self._build_retrieval_filters(query)
                )
                
                for result in vector_results:
                    knowledge_sources.append({
                        "source_id": result["node_id"],
                        "title": result["title"],
                        "content": result.get("content", result.get("description", "")),
                        "node_type": result["node_type"],
                        "similarity_score": result["similarity_score"],
                        "retrieval_method": "vector_similarity",
                        "metadata": result.get("node_properties", {})
                    })
            
            # Strategy 2: Graph-based retrieval
            if self.config.retrieval_strategy in ["graph", "hybrid"]:
                graph_results = await self._graph_based_retrieval(query)
                
                for result in graph_results:
                    # Avoid duplicates
                    if not any(ks["source_id"] == result["source_id"] for ks in knowledge_sources):
                        knowledge_sources.append(result)
            
            # Strategy 3: Semantic expansion
            if len(knowledge_sources) < self.config.max_retrieval_docs // 2:
                expanded_results = await self._semantic_expansion_retrieval(query, knowledge_sources)
                knowledge_sources.extend(expanded_results)
            
            # Sort by relevance and limit
            knowledge_sources.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            knowledge_sources = knowledge_sources[:self.config.max_retrieval_docs]
            
            logger.debug(
                "Enhanced knowledge retrieval completed",
                agent_id=self.agent_id,
                query_id=query.query_id,
                sources_found=len(knowledge_sources),
                strategy=self.config.retrieval_strategy
            )
            
            return knowledge_sources
            
        except Exception as e:
            logger.error(f"Enhanced knowledge retrieval failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _graph_based_retrieval(self, query: RAGQuery) -> List[Dict[str, Any]]:
        """Graph-based knowledge retrieval using relationship traversal.
        
        Args:
            query: RAG query
            
        Returns:
            List of graph-retrieved knowledge sources
        """
        try:
            # Extract entities from query (simplified approach)
            query_entities = await self._extract_query_entities(query.query)
            
            graph_results = []
            
            for entity in query_entities:
                # Find related documents through entity relationships
                results = await (self.query_builder
                    .match_node(NodeType.ENTITY, {"canonical_name": entity}, "entity")
                    .match_relationship("entity", "chunk", relationship_var="mentions")
                    .match_node(NodeType.CHUNK, variable="chunk")
                    .match_relationship("chunk", "doc", relationship_var="contains")
                    .match_node(NodeType.DOCUMENT, variable="doc")
                    .return_custom([
                        "doc.node_id as source_id",
                        "doc.title as title", 
                        "chunk.content as content",
                        "doc.node_type as node_type",
                        "mentions.confidence as relevance_score"
                    ])
                    .limit(5)
                    .execute())
                
                for result in results:
                    graph_results.append({
                        "source_id": result["source_id"],
                        "title": result["title"],
                        "content": result["content"],
                        "node_type": result["node_type"],
                        "similarity_score": result.get("relevance_score", 0.5),
                        "retrieval_method": "graph_traversal",
                        "related_entity": entity,
                        "metadata": {}
                    })
            
            return graph_results
            
        except Exception as e:
            logger.error(f"Graph-based retrieval failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _semantic_expansion_retrieval(
        self, 
        query: RAGQuery, 
        existing_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Semantic expansion retrieval to find related concepts.
        
        Args:
            query: RAG query
            existing_sources: Already retrieved sources
            
        Returns:
            List of semantically expanded knowledge sources
        """
        try:
            # Find similar concepts to expand search
            concept_results = await (self.query_builder
                .match_node(NodeType.CONCEPT, variable="concept")
                .where_property("concept", "domain", query.domain_filters[0] if query.domain_filters else "general")
                .match_relationship("concept", "doc", relationship_var="related")
                .match_node(NodeType.DOCUMENT, variable="doc")
                .return_custom([
                    "doc.node_id as source_id",
                    "doc.title as title",
                    "doc.description as content", 
                    "doc.node_type as node_type",
                    "related.weight as relevance_score"
                ])
                .limit(3)
                .execute())
            
            expanded_results = []
            existing_ids = {source["source_id"] for source in existing_sources}
            
            for result in concept_results:
                if result["source_id"] not in existing_ids:
                    expanded_results.append({
                        "source_id": result["source_id"],
                        "title": result["title"],
                        "content": result["content"],
                        "node_type": result["node_type"],
                        "similarity_score": result.get("relevance_score", 0.3),
                        "retrieval_method": "semantic_expansion",
                        "metadata": {}
                    })
            
            return expanded_results
            
        except Exception as e:
            logger.error(f"Semantic expansion retrieval failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _comprehensive_twin_consultation(
        self,
        query: RAGQuery,
        knowledge_sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Comprehensive AI Twins consultation with context.
        
        Args:
            query: RAG query
            knowledge_sources: Retrieved knowledge sources
            
        Returns:
            List of twin consultation results
        """
        if not self.twin_orchestrator:
            return []
        
        try:
            # Prepare context for twins
            twin_context = {
                "query_type": self._classify_query_type(query.query),
                "domain": query.domain_filters[0] if query.domain_filters else "general",
                "knowledge_sources": len(knowledge_sources),
                "user_context": query.context,
                "confidence_requirement": query.required_confidence
            }
            
            # Get twin recommendations and consultations
            twin_result = await self.twin_orchestrator.process_query(
                query=query.query,
                context=twin_context,
                enable_collaboration=True,
                user_id=query.user_id
            )
            
            # Format comprehensive twin insights
            twin_insights = []
            
            if hasattr(twin_result, 'contributing_twins'):
                for contribution in twin_result.contributing_twins:
                    twin_insights.append({
                        "twin_id": contribution.get("twin_id"),
                        "twin_type": contribution.get("twin_type"),
                        "insight": contribution.get("contribution"),
                        "confidence": contribution.get("confidence", 0.0),
                        "reasoning": contribution.get("reasoning", ""),
                        "consultation_type": "collaborative",
                        "relevant_sources": self._map_twin_to_sources(contribution, knowledge_sources)
                    })
                
                # Add synthesized insight
                twin_insights.append({
                    "twin_id": "orchestrator",
                    "twin_type": "synthesized",
                    "insight": twin_result.synthesized_response,
                    "confidence": twin_result.confidence_score,
                    "reasoning": "Synthesized from multiple expert perspectives",
                    "consultation_type": "synthesis",
                    "relevant_sources": []
                })
            
            return twin_insights
            
        except Exception as e:
            logger.error(f"Twin consultation failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _generate_enhanced_response(
        self,
        query: RAGQuery,
        knowledge_sources: List[Dict[str, Any]],
        twin_insights: List[Dict[str, Any]]
    ) -> ADKAgentResponse:
        """Generate enhanced response using ADK with rich context.
        
        Args:
            query: RAG query
            knowledge_sources: Retrieved knowledge sources
            twin_insights: AI Twin insights
            
        Returns:
            Enhanced ADK agent response
        """
        # Prepare enhanced context for ADK
        enhanced_context = {
            "query_metadata": {
                "query_id": query.query_id,
                "user_id": query.user_id,
                "session_id": query.session_id,
                "required_confidence": query.required_confidence,
                "include_reasoning": query.include_reasoning
            },
            "retrieval_metadata": {
                "strategy": self.config.retrieval_strategy,
                "sources_count": len(knowledge_sources),
                "avg_similarity": sum([ks.get("similarity_score", 0) for ks in knowledge_sources]) / len(knowledge_sources) if knowledge_sources else 0
            },
            "twin_metadata": {
                "consultations_count": len(twin_insights),
                "twin_types": list(set([ti["twin_type"] for ti in twin_insights]))
            },
            "preferences": {
                "max_length": self.config.max_response_length,
                "include_sources": self.config.include_sources,
                "include_confidence": self.config.include_confidence
            }
        }
        
        # Process through ADK agent with enhanced context
        adk_response = await self.adk_agent.process_query(
            query=query.query,
            context=enhanced_context,
            user_id=query.user_id,
            enable_twins=False,  # We've already done comprehensive twin consultation
            enable_retrieval=False  # We've already done enhanced retrieval
        )
        
        # Override retrieved documents and twin consultations with our enhanced versions
        adk_response.retrieved_documents = knowledge_sources
        adk_response.twin_consultations = twin_insights
        
        return adk_response
    
    async def _assess_response_quality(
        self,
        query: RAGQuery,
        adk_response: ADKAgentResponse,
        knowledge_sources: List[Dict[str, Any]],
        twin_insights: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Assess response quality across multiple dimensions.
        
        Args:
            query: Original RAG query
            adk_response: ADK agent response
            knowledge_sources: Knowledge sources used
            twin_insights: Twin insights used
            
        Returns:
            Quality metrics dictionary
        """
        quality_metrics = {}
        
        # 1. Source coverage assessment
        if knowledge_sources:
            high_quality_sources = len([ks for ks in knowledge_sources if ks.get("similarity_score", 0) > 0.8])
            quality_metrics["source_coverage"] = min(high_quality_sources / max(len(knowledge_sources), 1), 1.0)
        else:
            quality_metrics["source_coverage"] = 0.0
        
        # 2. Factual consistency assessment (simplified)
        response_length = len(adk_response.response)
        if response_length > 100:  # Substantial response
            # Check if response references sources appropriately
            source_references = sum([1 for ks in knowledge_sources if ks["title"].lower() in adk_response.response.lower()])
            quality_metrics["factual_consistency"] = min(source_references / max(len(knowledge_sources), 1), 1.0)
        else:
            quality_metrics["factual_consistency"] = 0.5  # Neutral for short responses
        
        # 3. Relevance score assessment
        if twin_insights:
            avg_twin_confidence = sum([ti.get("confidence", 0) for ti in twin_insights]) / len(twin_insights)
            quality_metrics["relevance_score"] = avg_twin_confidence
        else:
            quality_metrics["relevance_score"] = quality_metrics["source_coverage"]
        
        # 4. Overall confidence calculation
        confidence_factors = [
            quality_metrics["source_coverage"] * 0.4,
            quality_metrics["factual_consistency"] * 0.3,
            quality_metrics["relevance_score"] * 0.2,
            min(adk_response.confidence_score, 1.0) * 0.1
        ]
        
        quality_metrics["overall_confidence"] = sum(confidence_factors)
        
        return quality_metrics
    
    def _extract_reasoning_chain(
        self, 
        adk_response: ADKAgentResponse, 
        quality_metrics: Dict[str, float]
    ) -> List[str]:
        """Extract reasoning chain from response and quality assessment.
        
        Args:
            adk_response: ADK agent response
            quality_metrics: Quality metrics
            
        Returns:
            List of reasoning steps
        """
        reasoning_chain = []
        
        # Step 1: Knowledge retrieval reasoning
        if adk_response.retrieved_documents:
            reasoning_chain.append(
                f"Retrieved {len(adk_response.retrieved_documents)} relevant knowledge sources "
                f"with average similarity {sum([doc.get('similarity_score', 0) for doc in adk_response.retrieved_documents]) / len(adk_response.retrieved_documents):.3f}"
            )
        
        # Step 2: Twin consultation reasoning
        if adk_response.twin_consultations:
            twin_types = list(set([tc.get("twin_type", "unknown") for tc in adk_response.twin_consultations]))
            reasoning_chain.append(
                f"Consulted {len(adk_response.twin_consultations)} AI Digital Twins "
                f"including: {', '.join(twin_types)}"
            )
        
        # Step 3: Quality assessment reasoning
        reasoning_chain.append(
            f"Response quality assessment: "
            f"source coverage {quality_metrics['source_coverage']:.3f}, "
            f"factual consistency {quality_metrics['factual_consistency']:.3f}, "
            f"relevance {quality_metrics['relevance_score']:.3f}"
        )
        
        # Step 4: Overall confidence reasoning
        reasoning_chain.append(
            f"Overall confidence score: {quality_metrics['overall_confidence']:.3f} "
            f"based on source quality, twin expertise, and response coherence"
        )
        
        return reasoning_chain
    
    # Helper methods
    
    def _build_retrieval_filters(self, query: RAGQuery) -> Dict[str, Any]:
        """Build retrieval filters from query preferences."""
        filters = {}
        
        if query.preferred_sources:
            filters["source"] = query.preferred_sources
        
        if query.domain_filters:
            filters["domain"] = query.domain_filters
            
        return filters
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query (placeholder)."""
        # Placeholder implementation - use same as ADK agent
        settings = get_settings()
        embedding_dim = settings.ai_models.embedding_dimension
        
        import random
        random.seed(hash(query) % (2**32))
        return [random.random() for _ in range(embedding_dim)]
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract entities from query (simplified)."""
        # Simplified entity extraction - in production, use NLP models
        # Look for capitalized words as potential entities
        words = query.split()
        entities = [word.strip(".,!?") for word in words if word[0].isupper() and len(word) > 2]
        return list(set(entities))[:3]  # Limit to 3 entities
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for twin consultation."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how", "implement", "process", "workflow"]):
            return "process"
        elif any(word in query_lower for word in ["security", "compliance", "control", "audit"]):
            return "security"
        elif any(word in query_lower for word in ["optimize", "improve", "performance", "efficiency"]):
            return "optimization"
        elif any(word in query_lower for word in ["explain", "what", "define", "concept"]):
            return "explanation"
        else:
            return "general"
    
    def _map_twin_to_sources(
        self, 
        twin_contribution: Dict[str, Any], 
        knowledge_sources: List[Dict[str, Any]]
    ) -> List[str]:
        """Map twin contribution to relevant knowledge sources."""
        # Simplified mapping - look for source mentions in twin reasoning
        twin_reasoning = twin_contribution.get("reasoning", "").lower()
        relevant_sources = []
        
        for source in knowledge_sources:
            if source["title"].lower() in twin_reasoning or source["source_id"] in twin_reasoning:
                relevant_sources.append(source["source_id"])
        
        return relevant_sources[:3]  # Limit to 3 most relevant
    
    async def stream_response(self, query: RAGQuery) -> AsyncGenerator[str, None]:
        """Stream RAG response for real-time interactions."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # First, send retrieval status
            yield f"🔍 Retrieving knowledge for: {query.query[:100]}...\n\n"
            
            # Perform knowledge retrieval
            knowledge_sources = await self._enhanced_knowledge_retrieval(query)
            yield f"📚 Found {len(knowledge_sources)} relevant knowledge sources\n\n"
            
            # Consult twins
            if self.config.enable_expert_consultation and self.twin_orchestrator:
                yield "🤖 Consulting AI Digital Twins...\n\n"
                twin_insights = await self._comprehensive_twin_consultation(query, knowledge_sources)
                yield f"💡 Received insights from {len(twin_insights)} AI experts\n\n"
            else:
                twin_insights = []
            
            # Stream response generation
            yield "✨ Generating response...\n\n"
            
            async for chunk in self.adk_agent.stream_response(
                query=query.query,
                context={"knowledge_sources": knowledge_sources, "twin_insights": twin_insights},
                user_id=query.user_id
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}", agent_id=self.agent_id)
            yield f"\n\n❌ Error: {str(e)}"
    
    async def get_agent_metrics(self) -> Dict[str, Any]:
        """Get comprehensive agent metrics and statistics."""
        adk_status = await self.adk_agent.get_agent_status()
        schema_stats = await self.schema_manager.get_schema_statistics()
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.config.agent_name,
            "initialized": self._initialized,
            "queries_cached": len(self._query_cache),
            "adk_status": adk_status,
            "schema_stats": {
                "total_nodes": schema_stats["total_nodes"],
                "total_relationships": schema_stats["total_relationships"],
                "vector_statistics": schema_stats["vector_statistics"]
            },
            "configuration": {
                "retrieval_strategy": self.config.retrieval_strategy,
                "max_retrieval_docs": self.config.max_retrieval_docs,
                "min_similarity_threshold": self.config.min_similarity_threshold,
                "enable_expert_consultation": self.config.enable_expert_consultation
            }
        }