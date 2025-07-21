"""
Google ADK Agent Integration for Knowledge Graph-RAG.

Provides core Google ADK agent functionality with integration to
Neo4j vector graph schema and AI Digital Twins framework.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from pathlib import Path
import json

import structlog
from pydantic import BaseModel, Field, validator
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content
from google.cloud import aiplatform
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import ADKAgentError, ConfigurationError
from kg_rag.ai_twins.twin_orchestrator import TwinOrchestrator
from kg_rag.graph_schema.vector_operations import VectorGraphOperations
from kg_rag.graph_schema.query_builder import GraphQueryBuilder

logger = structlog.get_logger(__name__)


class ADKConfiguration(BaseModel):
    """Configuration for Google ADK agent."""
    
    # Google Cloud Configuration
    project_id: str = Field(..., description="Google Cloud project ID")
    location: str = Field(default="us-central1", description="Google Cloud location")
    
    # Model Configuration
    model_name: str = Field(default="gemini-1.5-pro", description="Generative model name")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Model temperature")
    max_output_tokens: int = Field(default=8192, description="Maximum output tokens")
    top_p: float = Field(default=0.8, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=40, description="Top-k sampling")
    
    # RAG Configuration
    retrieval_limit: int = Field(default=10, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Vector similarity threshold")
    hybrid_search_enabled: bool = Field(default=True, description="Enable hybrid graph-vector search")
    
    # Safety and Security
    safety_settings: Dict[str, str] = Field(default_factory=dict, description="Safety settings")
    enable_content_filtering: bool = Field(default=True, description="Enable content filtering")
    
    # Performance
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class ADKAgentResponse(BaseModel):
    """Response from ADK agent."""
    
    response_id: str = Field(..., description="Unique response ID")
    agent_id: str = Field(..., description="Agent ID that generated response")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response")
    
    # Context and Sources
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    twin_consultations: List[Dict[str, Any]] = Field(default_factory=list, description="AI Twin consultations")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Response confidence")
    
    # Performance Metrics
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    retrieval_time_ms: float = Field(default=0.0, description="Retrieval time in milliseconds")
    generation_time_ms: float = Field(default=0.0, description="Generation time in milliseconds")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    model_used: str = Field(default="", description="Model used for generation")
    total_tokens: int = Field(default=0, description="Total tokens used")


class ADKAgent:
    """Google ADK agent with Knowledge Graph-RAG integration."""
    
    def __init__(
        self,
        agent_id: str,
        config: ADKConfiguration,
        twin_orchestrator: Optional[TwinOrchestrator] = None,
        vector_operations: Optional[VectorGraphOperations] = None,
        query_builder: Optional[GraphQueryBuilder] = None
    ):
        """Initialize ADK agent.
        
        Args:
            agent_id: Unique agent identifier
            config: ADK configuration
            twin_orchestrator: AI Digital Twins orchestrator
            vector_operations: Vector graph operations
            query_builder: Graph query builder
        """
        self.agent_id = agent_id
        self.config = config
        self.twin_orchestrator = twin_orchestrator
        self.vector_operations = vector_operations
        self.query_builder = query_builder
        
        # Internal state
        self._model: Optional[GenerativeModel] = None
        self._cache: Dict[str, Any] = {}
        self._initialized = False
        
        logger.info(
            "ADK agent initialized",
            agent_id=agent_id,
            project_id=config.project_id,
            model_name=config.model_name
        )
    
    async def initialize(self) -> None:
        """Initialize Google ADK and Vertex AI."""
        if self._initialized:
            return
        
        try:
            # Initialize Vertex AI
            vertexai.init(
                project=self.config.project_id,
                location=self.config.location
            )
            
            # Initialize AI Platform
            aiplatform.init(
                project=self.config.project_id,
                location=self.config.location
            )
            
            # Configure generative model
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k
            }
            
            # Initialize model with safety settings
            safety_settings = self._get_safety_settings()
            
            self._model = GenerativeModel(
                model_name=self.config.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self._initialized = True
            
            logger.info(
                "ADK agent initialized successfully",
                agent_id=self.agent_id,
                model_name=self.config.model_name
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize ADK agent: {str(e)}"
            logger.error(error_msg, agent_id=self.agent_id, error=str(e))
            raise ADKAgentError(error_msg) from e
    
    def _get_safety_settings(self) -> List[Dict[str, Any]]:
        """Get safety settings for the model."""
        if not self.config.enable_content_filtering:
            return []
        
        # Default safety settings
        default_settings = [
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        # Merge with custom safety settings
        custom_settings = self.config.safety_settings
        for setting in default_settings:
            category = setting["category"]
            if category in custom_settings:
                setting["threshold"] = custom_settings[category]
        
        return default_settings
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        enable_twins: bool = True,
        enable_retrieval: bool = True
    ) -> ADKAgentResponse:
        """Process a query using RAG with Knowledge Graph integration.
        
        Args:
            query: User query
            context: Additional context
            user_id: User identifier
            enable_twins: Whether to consult AI Digital Twins
            enable_retrieval: Whether to perform knowledge retrieval
            
        Returns:
            ADK agent response
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.utcnow()
        response_id = f"resp_{self.agent_id}_{int(start_time.timestamp())}"
        
        logger.info(
            "Processing query",
            agent_id=self.agent_id,
            response_id=response_id,
            query_length=len(query),
            enable_twins=enable_twins,
            enable_retrieval=enable_retrieval
        )
        
        try:
            # Step 1: Knowledge retrieval
            retrieved_docs = []
            retrieval_start = datetime.utcnow()
            
            if enable_retrieval and self.vector_operations:
                retrieved_docs = await self._retrieve_knowledge(query, context)
            
            retrieval_time = (datetime.utcnow() - retrieval_start).total_seconds() * 1000
            
            # Step 2: AI Twin consultation
            twin_consultations = []
            if enable_twins and self.twin_orchestrator:
                twin_consultations = await self._consult_twins(query, context, user_id)
            
            # Step 3: Generate response using ADK
            generation_start = datetime.utcnow()
            
            response_text, model_metadata = await self._generate_response(
                query=query,
                retrieved_docs=retrieved_docs,
                twin_consultations=twin_consultations,
                context=context
            )
            
            generation_time = (datetime.utcnow() - generation_start).total_seconds() * 1000
            
            # Step 4: Calculate confidence score
            confidence_score = self._calculate_confidence(
                retrieved_docs, twin_consultations, model_metadata
            )
            
            # Step 5: Create response object
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            response = ADKAgentResponse(
                response_id=response_id,
                agent_id=self.agent_id,
                query=query,
                response=response_text,
                retrieved_documents=retrieved_docs,
                twin_consultations=twin_consultations,
                confidence_score=confidence_score,
                processing_time_ms=total_time,
                retrieval_time_ms=retrieval_time,
                generation_time_ms=generation_time,
                model_used=self.config.model_name,
                total_tokens=model_metadata.get("total_tokens", 0)
            )
            
            # Cache response if enabled
            if self.config.enable_caching:
                await self._cache_response(query, response)
            
            logger.info(
                "Query processed successfully",
                agent_id=self.agent_id,
                response_id=response_id,
                processing_time_ms=total_time,
                confidence_score=confidence_score,
                retrieved_docs=len(retrieved_docs),
                twin_consultations=len(twin_consultations)
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(
                error_msg,
                agent_id=self.agent_id,
                response_id=response_id,
                query=query,
                error=str(e)
            )
            raise ADKAgentError(error_msg) from e
    
    async def _retrieve_knowledge(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant knowledge from the graph database.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            List of retrieved documents
        """
        try:
            # Generate query embedding (placeholder - would use actual embedding service)
            query_embedding = await self._generate_query_embedding(query)
            
            if self.config.hybrid_search_enabled:
                # Hybrid graph-vector search
                results = await self.vector_operations.hybrid_search(
                    query_vector=query_embedding,
                    graph_filters=context.get("filters") if context else None,
                    limit=self.config.retrieval_limit,
                    similarity_threshold=self.config.similarity_threshold
                )
            else:
                # Pure vector similarity search
                results = await self.vector_operations.vector_similarity_search(
                    query_vector=query_embedding,
                    limit=self.config.retrieval_limit,
                    similarity_threshold=self.config.similarity_threshold
                )
            
            # Format results for RAG
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "node_id": result["node_id"],
                    "title": result["title"],
                    "content": result.get("content", result.get("description", "")),
                    "similarity_score": result.get("similarity_score", 0.0),
                    "node_type": result["node_type"],
                    "source": result.get("source", "knowledge_graph")
                })
            
            logger.debug(
                "Knowledge retrieval completed",
                agent_id=self.agent_id,
                results_count=len(formatted_results),
                avg_similarity=sum([r["similarity_score"] for r in formatted_results]) / len(formatted_results) if formatted_results else 0
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _consult_twins(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Consult AI Digital Twins for specialized insights.
        
        Args:
            query: User query
            context: Additional context
            user_id: User identifier
            
        Returns:
            List of twin consultation results
        """
        try:
            # Process query through twin orchestrator
            twin_result = await self.twin_orchestrator.process_query(
                query=query,
                context=context,
                enable_collaboration=True,
                user_id=user_id
            )
            
            # Format twin consultation results
            consultations = []
            
            if hasattr(twin_result, 'contributing_twins'):
                # Multiple twins collaborated
                for twin_contribution in twin_result.contributing_twins:
                    consultations.append({
                        "twin_id": twin_contribution.get("twin_id"),
                        "twin_type": twin_contribution.get("twin_type"),
                        "contribution": twin_contribution.get("contribution"),
                        "confidence": twin_contribution.get("confidence", 0.0),
                        "reasoning": twin_contribution.get("reasoning", "")
                    })
                
                # Add synthesized result
                consultations.append({
                    "twin_id": "orchestrator",
                    "twin_type": "synthesized",
                    "contribution": twin_result.synthesized_response,
                    "confidence": twin_result.confidence_score,
                    "reasoning": "Synthesized from multiple twin contributions"
                })
            else:
                # Single twin response
                consultations.append({
                    "twin_id": getattr(twin_result, 'twin_id', 'unknown'),
                    "twin_type": getattr(twin_result, 'twin_type', 'unknown'),
                    "contribution": twin_result.response if hasattr(twin_result, 'response') else str(twin_result),
                    "confidence": getattr(twin_result, 'confidence', 0.0),
                    "reasoning": getattr(twin_result, 'reasoning', '')
                })
            
            logger.debug(
                "Twin consultation completed",
                agent_id=self.agent_id,
                consultations_count=len(consultations)
            )
            
            return consultations
            
        except Exception as e:
            logger.error(f"Twin consultation failed: {str(e)}", agent_id=self.agent_id)
            return []
    
    async def _generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        twin_consultations: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[str, Dict[str, Any]]:
        """Generate response using Google ADK.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            twin_consultations: Twin consultation results
            context: Additional context
            
        Returns:
            Tuple of (response_text, model_metadata)
        """
        # Construct RAG prompt
        prompt = self._construct_rag_prompt(
            query=query,
            retrieved_docs=retrieved_docs,
            twin_consultations=twin_consultations,
            context=context
        )
        
        try:
            # Generate response using Vertex AI
            response = await asyncio.to_thread(
                self._model.generate_content,
                prompt
            )
            
            # Extract response text
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # Extract metadata
            metadata = {
                "total_tokens": getattr(response.usage_metadata, 'total_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                "prompt_tokens": getattr(response.usage_metadata, 'prompt_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                "completion_tokens": getattr(response.usage_metadata, 'candidates_token_count', 0) if hasattr(response, 'usage_metadata') else 0,
                "model": self.config.model_name,
                "finish_reason": getattr(response.candidates[0], 'finish_reason', 'unknown') if hasattr(response, 'candidates') and response.candidates else 'unknown'
            }
            
            return response_text, metadata
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}", agent_id=self.agent_id)
            raise ADKAgentError(f"Failed to generate response: {str(e)}") from e
    
    def _construct_rag_prompt(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        twin_consultations: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Construct RAG prompt with knowledge and twin insights.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            twin_consultations: Twin consultation results
            context: Additional context
            
        Returns:
            Formatted RAG prompt
        """
        prompt_parts = []
        
        # System instruction
        prompt_parts.append("""You are an AI assistant with access to a knowledge graph and AI Digital Twins.
You provide accurate, helpful responses based on retrieved knowledge and expert insights.

Instructions:
- Use the provided knowledge sources to answer the query
- Consider insights from AI Digital Twins when available
- Be precise and cite your sources when relevant
- If information is insufficient, acknowledge the limitations
- Maintain a helpful and professional tone""")
        
        # Add retrieved knowledge
        if retrieved_docs:
            prompt_parts.append("\n## Retrieved Knowledge:")
            for i, doc in enumerate(retrieved_docs, 1):
                prompt_parts.append(f"\n### Source {i} (Similarity: {doc['similarity_score']:.3f})")
                prompt_parts.append(f"**Title:** {doc['title']}")
                prompt_parts.append(f"**Content:** {doc['content'][:1000]}{'...' if len(doc['content']) > 1000 else ''}")
                prompt_parts.append(f"**Type:** {doc['node_type']}")
        
        # Add twin consultations
        if twin_consultations:
            prompt_parts.append("\n## AI Digital Twin Insights:")
            for i, consultation in enumerate(twin_consultations, 1):
                prompt_parts.append(f"\n### {consultation['twin_type'].title()} Twin (Confidence: {consultation['confidence']:.3f})")
                prompt_parts.append(f"**Insight:** {consultation['contribution']}")
                if consultation['reasoning']:
                    prompt_parts.append(f"**Reasoning:** {consultation['reasoning']}")
        
        # Add context if provided
        if context and context.get("additional_info"):
            prompt_parts.append(f"\n## Additional Context:")
            prompt_parts.append(context["additional_info"])
        
        # Add the actual query
        prompt_parts.append(f"\n## User Query:")
        prompt_parts.append(query)
        
        prompt_parts.append(f"\n## Response:")
        prompt_parts.append("Based on the above knowledge sources and expert insights, here is my response:")
        
        return "\n".join(prompt_parts)
    
    def _calculate_confidence(
        self,
        retrieved_docs: List[Dict[str, Any]],
        twin_consultations: List[Dict[str, Any]],
        model_metadata: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the response.
        
        Args:
            retrieved_docs: Retrieved documents
            twin_consultations: Twin consultation results
            model_metadata: Model generation metadata
            
        Returns:
            Confidence score (0.0-1.0)
        """
        confidence_factors = []
        
        # Factor 1: Retrieval quality
        if retrieved_docs:
            avg_similarity = sum([doc["similarity_score"] for doc in retrieved_docs]) / len(retrieved_docs)
            retrieval_confidence = min(avg_similarity * 1.2, 1.0)  # Boost but cap at 1.0
            confidence_factors.append(retrieval_confidence * 0.4)  # 40% weight
        else:
            confidence_factors.append(0.2)  # Low confidence without retrieval
        
        # Factor 2: Twin consultation quality
        if twin_consultations:
            avg_twin_confidence = sum([consultation["confidence"] for consultation in twin_consultations]) / len(twin_consultations)
            confidence_factors.append(avg_twin_confidence * 0.3)  # 30% weight
        else:
            confidence_factors.append(0.1)  # Lower confidence without twin insights
        
        # Factor 3: Model completion quality
        finish_reason = model_metadata.get("finish_reason", "unknown")
        if finish_reason == "STOP":
            model_confidence = 0.3  # 30% weight for clean completion
        elif finish_reason in ["MAX_TOKENS", "LENGTH"]:
            model_confidence = 0.2  # Reduced confidence for truncated responses
        else:
            model_confidence = 0.1  # Low confidence for other reasons
        
        confidence_factors.append(model_confidence)
        
        return sum(confidence_factors)
    
    async def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for query (placeholder implementation).
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        # This is a placeholder - in production, this would use the same
        # embedding service as the knowledge graph ingestion pipeline
        settings = get_settings()
        embedding_dim = settings.ai_models.embedding_dimension
        
        # For now, return a placeholder embedding
        # In production, integrate with sentence-transformers or similar
        import random
        random.seed(hash(query) % (2**32))
        return [random.random() for _ in range(embedding_dim)]
    
    async def _cache_response(self, query: str, response: ADKAgentResponse) -> None:
        """Cache response for future queries.
        
        Args:
            query: Original query
            response: Agent response
        """
        if self.config.enable_caching:
            cache_key = f"agent_{self.agent_id}_query_{hash(query)}"
            self._cache[cache_key] = {
                "response": response.dict(),
                "timestamp": datetime.utcnow(),
                "ttl": self.config.cache_ttl_seconds
            }
    
    async def stream_response(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response generation for real-time interactions.
        
        Args:
            query: User query
            context: Additional context
            user_id: User identifier
            
        Yields:
            Streaming response chunks
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Perform retrieval first
            retrieved_docs = []
            if self.vector_operations:
                retrieved_docs = await self._retrieve_knowledge(query, context)
            
            # Consult twins
            twin_consultations = []
            if self.twin_orchestrator:
                twin_consultations = await self._consult_twins(query, context, user_id)
            
            # Construct prompt
            prompt = self._construct_rag_prompt(
                query=query,
                retrieved_docs=retrieved_docs,
                twin_consultations=twin_consultations,
                context=context
            )
            
            # Stream response using Vertex AI
            response_stream = await asyncio.to_thread(
                self._model.generate_content,
                prompt,
                stream=True
            )
            
            for chunk in response_stream:
                if hasattr(chunk, 'text') and chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            logger.error(f"Streaming response failed: {str(e)}", agent_id=self.agent_id)
            yield f"Error: {str(e)}"
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status and statistics.
        
        Returns:
            Agent status information
        """
        return {
            "agent_id": self.agent_id,
            "initialized": self._initialized,
            "model_name": self.config.model_name,
            "project_id": self.config.project_id,
            "cache_size": len(self._cache),
            "configuration": self.config.dict()
        }