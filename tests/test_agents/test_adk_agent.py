"""
Test suite for Google ADK Agent integration.

Tests the core ADK agent functionality including Google Cloud integration,
query processing, knowledge retrieval, and AI Digital Twins consultation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from kg_rag.agents.adk_agent import (
    ADKAgent, ADKConfiguration, ADKAgentResponse
)
from kg_rag.core.exceptions import ADKAgentError, ConfigurationError
from kg_rag.ai_twins.twin_orchestrator import TwinOrchestrator
from kg_rag.graph_schema.vector_operations import VectorGraphOperations
from kg_rag.graph_schema.query_builder import GraphQueryBuilder


class TestADKConfiguration:
    """Test ADK agent configuration."""
    
    def test_valid_configuration(self):
        """Test creating valid ADK configuration."""
        config = ADKConfiguration(
            project_id="test-project",
            location="us-central1",
            model_name="gemini-1.5-pro",
            temperature=0.3,
            max_output_tokens=4096
        )
        
        assert config.project_id == "test-project"
        assert config.location == "us-central1"
        assert config.model_name == "gemini-1.5-pro"
        assert config.temperature == 0.3
        assert config.max_output_tokens == 4096
        assert config.hybrid_search_enabled is True
        assert config.enable_content_filtering is True
    
    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid temperature
        config = ADKConfiguration(
            project_id="test-project",
            temperature=0.5
        )
        assert config.temperature == 0.5
        
        # Invalid temperature - should fail validation
        with pytest.raises(ValueError):
            ADKConfiguration(
                project_id="test-project",
                temperature=3.0  # > 2.0
            )
        
        with pytest.raises(ValueError):
            ADKConfiguration(
                project_id="test-project",
                temperature=-0.5  # < 0.0
            )
    
    def test_top_p_validation(self):
        """Test top_p parameter validation."""
        # Valid top_p
        config = ADKConfiguration(
            project_id="test-project",
            top_p=0.8
        )
        assert config.top_p == 0.8
        
        # Invalid top_p
        with pytest.raises(ValueError):
            ADKConfiguration(
                project_id="test-project",
                top_p=1.5  # > 1.0
            )
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ADKConfiguration(project_id="test-project")
        
        assert config.location == "us-central1"
        assert config.model_name == "gemini-1.5-pro"
        assert config.temperature == 0.3
        assert config.max_output_tokens == 8192
        assert config.top_p == 0.8
        assert config.top_k == 40
        assert config.retrieval_limit == 10
        assert config.similarity_threshold == 0.7
        assert config.hybrid_search_enabled is True
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 3600


class TestADKAgentResponse:
    """Test ADK agent response model."""
    
    def test_valid_response(self):
        """Test creating valid ADK response."""
        response = ADKAgentResponse(
            response_id="test-123",
            agent_id="adk-agent-1",
            query="What is machine learning?",
            response="Machine learning is a subset of artificial intelligence...",
            confidence_score=0.85,
            processing_time_ms=1500.0,
            model_used="gemini-1.5-pro",
            total_tokens=250
        )
        
        assert response.response_id == "test-123"
        assert response.agent_id == "adk-agent-1"
        assert response.confidence_score == 0.85
        assert response.processing_time_ms == 1500.0
        assert response.model_used == "gemini-1.5-pro"
        assert response.total_tokens == 250
        assert isinstance(response.timestamp, datetime)
        assert response.retrieved_documents == []
        assert response.twin_consultations == []
    
    def test_confidence_score_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        response = ADKAgentResponse(
            response_id="test-123",
            agent_id="adk-agent-1",
            query="test",
            response="test response",
            confidence_score=0.5
        )
        assert response.confidence_score == 0.5
        
        # Invalid confidence scores
        with pytest.raises(ValueError):
            ADKAgentResponse(
                response_id="test-123",
                agent_id="adk-agent-1",
                query="test",
                response="test response",
                confidence_score=1.5  # > 1.0
            )
        
        with pytest.raises(ValueError):
            ADKAgentResponse(
                response_id="test-123",
                agent_id="adk-agent-1",
                query="test",
                response="test response",
                confidence_score=-0.1  # < 0.0
            )


class TestADKAgent:
    """Test ADK agent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock ADK configuration."""
        return ADKConfiguration(
            project_id="test-project",
            location="us-central1",
            model_name="gemini-1.5-pro"
        )
    
    @pytest.fixture
    def mock_twin_orchestrator(self):
        """Create mock twin orchestrator."""
        orchestrator = Mock(spec=TwinOrchestrator)
        orchestrator.process_query = AsyncMock()
        return orchestrator
    
    @pytest.fixture
    def mock_vector_operations(self):
        """Create mock vector operations."""
        ops = Mock(spec=VectorGraphOperations)
        ops.vector_similarity_search = AsyncMock()
        ops.hybrid_search = AsyncMock()
        return ops
    
    @pytest.fixture
    def mock_query_builder(self):
        """Create mock query builder."""
        builder = Mock(spec=GraphQueryBuilder)
        return builder
    
    @pytest.fixture
    def adk_agent(self, mock_config, mock_twin_orchestrator, mock_vector_operations, mock_query_builder):
        """Create ADK agent instance."""
        return ADKAgent(
            agent_id="test-adk-agent",
            config=mock_config,
            twin_orchestrator=mock_twin_orchestrator,
            vector_operations=mock_vector_operations,
            query_builder=mock_query_builder
        )
    
    def test_agent_initialization(self, adk_agent, mock_config):
        """Test ADK agent initialization."""
        assert adk_agent.agent_id == "test-adk-agent"
        assert adk_agent.config == mock_config
        assert adk_agent._initialized is False
        assert adk_agent._model is None
        assert adk_agent._cache == {}
    
    @patch('kg_rag.agents.adk_agent.vertexai')
    @patch('kg_rag.agents.adk_agent.aiplatform')
    @patch('kg_rag.agents.adk_agent.GenerativeModel')
    async def test_initialization_success(
        self, 
        mock_generative_model, 
        mock_aiplatform, 
        mock_vertexai,
        adk_agent
    ):
        """Test successful agent initialization."""
        # Setup mocks
        mock_model_instance = Mock()
        mock_generative_model.return_value = mock_model_instance
        
        # Initialize agent
        await adk_agent.initialize()
        
        # Verify initialization calls
        mock_vertexai.init.assert_called_once_with(
            project="test-project",
            location="us-central1"
        )
        mock_aiplatform.init.assert_called_once_with(
            project="test-project",
            location="us-central1"
        )
        mock_generative_model.assert_called_once()
        
        assert adk_agent._initialized is True
        assert adk_agent._model == mock_model_instance
    
    @patch('kg_rag.agents.adk_agent.vertexai')
    async def test_initialization_failure(self, mock_vertexai, adk_agent):
        """Test initialization failure handling."""
        # Setup mock to raise exception
        mock_vertexai.init.side_effect = Exception("Authentication failed")
        
        # Test initialization failure
        with pytest.raises(ADKAgentError, match="Failed to initialize ADK agent"):
            await adk_agent.initialize()
        
        assert adk_agent._initialized is False
    
    async def test_process_query_not_initialized(self, adk_agent):
        """Test processing query without initialization."""
        with patch.object(adk_agent, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_init.return_value = None
            
            # Mock other methods to avoid further execution
            with patch.object(adk_agent, '_retrieve_knowledge', new_callable=AsyncMock):
                with patch.object(adk_agent, '_consult_twins', new_callable=AsyncMock):
                    with patch.object(adk_agent, '_generate_response', new_callable=AsyncMock) as mock_gen:
                        mock_gen.return_value = ("Test response", {"total_tokens": 100})
                        
                        await adk_agent.process_query("test query")
                        mock_init.assert_called_once()
    
    async def test_retrieve_knowledge_vector_search(self, adk_agent, mock_vector_operations):
        """Test knowledge retrieval with vector search."""
        # Setup mock vector operations
        mock_results = [
            {
                "node_id": "doc1",
                "title": "Document 1",
                "content": "Content 1",
                "similarity_score": 0.85,
                "node_type": "document"
            },
            {
                "node_id": "doc2", 
                "title": "Document 2",
                "description": "Description 2",
                "similarity_score": 0.75,
                "node_type": "chunk"
            }
        ]
        mock_vector_operations.hybrid_search.return_value = mock_results
        
        # Mock query embedding generation
        with patch.object(adk_agent, '_generate_query_embedding', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [0.1, 0.2, 0.3]
            
            # Test knowledge retrieval
            results = await adk_agent._retrieve_knowledge("test query")
            
            assert len(results) == 2
            assert results[0]["node_id"] == "doc1"
            assert results[0]["title"] == "Document 1"
            assert results[0]["content"] == "Content 1"
            assert results[0]["similarity_score"] == 0.85
            
            assert results[1]["node_id"] == "doc2"
            assert results[1]["title"] == "Document 2"
            assert results[1]["content"] == "Description 2"  # Falls back to description
            assert results[1]["similarity_score"] == 0.75
    
    async def test_retrieve_knowledge_no_operations(self, adk_agent):
        """Test knowledge retrieval without vector operations."""
        # Set vector_operations to None
        adk_agent.vector_operations = None
        
        # Should return empty list
        results = await adk_agent._retrieve_knowledge("test query")
        assert results == []
    
    async def test_consult_twins_multiple_contributions(self, adk_agent, mock_twin_orchestrator):
        """Test twin consultation with multiple contributors."""
        # Setup mock twin result with multiple contributions
        mock_twin_result = Mock()
        mock_twin_result.contributing_twins = [
            {
                "twin_id": "expert-1",
                "twin_type": "technical",
                "contribution": "Technical insight 1",
                "confidence": 0.9,
                "reasoning": "Based on experience"
            },
            {
                "twin_id": "expert-2", 
                "twin_type": "business",
                "contribution": "Business insight 1",
                "confidence": 0.8,
                "reasoning": "Market analysis"
            }
        ]
        mock_twin_result.synthesized_response = "Combined expert insights"
        mock_twin_result.confidence_score = 0.85
        
        mock_twin_orchestrator.process_query.return_value = mock_twin_result
        
        # Test twin consultation
        consultations = await adk_agent._consult_twins(
            "test query", 
            {"domain": "technology"}, 
            "user123"
        )
        
        assert len(consultations) == 3  # 2 individual + 1 synthesized
        
        # Check individual contributions
        assert consultations[0]["twin_id"] == "expert-1"
        assert consultations[0]["twin_type"] == "technical"
        assert consultations[0]["contribution"] == "Technical insight 1"
        assert consultations[0]["confidence"] == 0.9
        
        assert consultations[1]["twin_id"] == "expert-2"
        assert consultations[1]["twin_type"] == "business"
        assert consultations[1]["contribution"] == "Business insight 1"
        assert consultations[1]["confidence"] == 0.8
        
        # Check synthesized contribution
        assert consultations[2]["twin_id"] == "orchestrator"
        assert consultations[2]["twin_type"] == "synthesized"
        assert consultations[2]["contribution"] == "Combined expert insights"
        assert consultations[2]["confidence"] == 0.85
    
    async def test_consult_twins_single_response(self, adk_agent, mock_twin_orchestrator):
        """Test twin consultation with single response."""
        # Setup mock twin result without contributing_twins attribute
        mock_twin_result = Mock()
        mock_twin_result.response = "Single expert response"
        mock_twin_result.confidence = 0.8
        mock_twin_result.twin_id = "expert-solo"
        mock_twin_result.twin_type = "general"
        
        # Remove contributing_twins attribute to simulate single response
        delattr(mock_twin_result, 'contributing_twins')
        
        mock_twin_orchestrator.process_query.return_value = mock_twin_result
        
        # Test twin consultation
        consultations = await adk_agent._consult_twins("test query")
        
        assert len(consultations) == 1
        assert consultations[0]["twin_id"] == "expert-solo"
        assert consultations[0]["twin_type"] == "general"
        assert consultations[0]["contribution"] == "Single expert response"
        assert consultations[0]["confidence"] == 0.8
    
    async def test_consult_twins_failure(self, adk_agent, mock_twin_orchestrator):
        """Test twin consultation failure handling."""
        # Setup mock to raise exception
        mock_twin_orchestrator.process_query.side_effect = Exception("Twin orchestrator failed")
        
        # Test should return empty list on failure
        consultations = await adk_agent._consult_twins("test query")
        assert consultations == []
    
    @patch('kg_rag.agents.adk_agent.asyncio.to_thread')
    async def test_generate_response_success(self, mock_to_thread, adk_agent):
        """Test successful response generation."""
        # Setup mock model and response
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = "Generated response from ADK"
        mock_response.usage_metadata.total_token_count = 150
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.candidates = [Mock()]
        mock_response.candidates[0].finish_reason = "STOP"
        
        adk_agent._model = mock_model
        mock_to_thread.return_value = mock_response
        
        # Test response generation
        response_text, metadata = await adk_agent._generate_response(
            query="test query",
            retrieved_docs=[{"title": "Doc 1", "content": "Content 1"}],
            twin_consultations=[{"twin_type": "expert", "contribution": "Expert insight"}],
            context={"additional_info": "Context"}
        )
        
        assert response_text == "Generated response from ADK"
        assert metadata["total_tokens"] == 150
        assert metadata["prompt_tokens"] == 100
        assert metadata["completion_tokens"] == 50
        assert metadata["model"] == "gemini-1.5-pro"
        assert metadata["finish_reason"] == "STOP"
    
    async def test_calculate_confidence_with_all_factors(self, adk_agent):
        """Test confidence calculation with all factors."""
        retrieved_docs = [
            {"similarity_score": 0.9},
            {"similarity_score": 0.8}
        ]
        
        twin_consultations = [
            {"confidence": 0.85},
            {"confidence": 0.75}
        ]
        
        model_metadata = {"finish_reason": "STOP"}
        
        confidence = adk_agent._calculate_confidence(
            retrieved_docs, twin_consultations, model_metadata
        )
        
        # Expected: (0.85 * 1.2 * 0.4) + (0.8 * 0.3) + 0.3 = 0.408 + 0.24 + 0.3 = 0.948
        # But capped at 1.0 for retrieval factor
        expected = (1.0 * 0.4) + (0.8 * 0.3) + 0.3  # 0.4 + 0.24 + 0.3 = 0.94
        assert abs(confidence - expected) < 0.01
    
    async def test_calculate_confidence_no_retrieval(self, adk_agent):
        """Test confidence calculation without retrieval."""
        confidence = adk_agent._calculate_confidence([], [], {"finish_reason": "STOP"})
        
        # Expected: 0.2 (no retrieval) + 0.1 (no twins) + 0.3 (clean completion) = 0.6
        assert confidence == 0.6
    
    async def test_calculate_confidence_truncated_response(self, adk_agent):
        """Test confidence calculation with truncated response."""
        retrieved_docs = [{"similarity_score": 0.8}]
        twin_consultations = [{"confidence": 0.7}]
        model_metadata = {"finish_reason": "MAX_TOKENS"}
        
        confidence = adk_agent._calculate_confidence(
            retrieved_docs, twin_consultations, model_metadata
        )
        
        # Expected: (0.8 * 1.2 * 0.4) + (0.7 * 0.3) + 0.2 = 0.384 + 0.21 + 0.2 = 0.794
        # But retrieval factor is capped at 1.0 * 0.4 = 0.4
        expected = (1.0 * 0.4) + (0.7 * 0.3) + 0.2  # 0.4 + 0.21 + 0.2 = 0.81
        assert abs(confidence - expected) < 0.01
    
    def test_construct_rag_prompt_complete(self, adk_agent):
        """Test RAG prompt construction with all components."""
        retrieved_docs = [
            {
                "title": "Document 1",
                "content": "This is the content of document 1" * 50,  # Long content
                "similarity_score": 0.9,
                "node_type": "document"
            },
            {
                "title": "Document 2", 
                "content": "Short content",
                "similarity_score": 0.7,
                "node_type": "chunk"
            }
        ]
        
        twin_consultations = [
            {
                "twin_type": "technical",
                "contribution": "Technical insight",
                "confidence": 0.8,
                "reasoning": "Based on technical expertise"
            },
            {
                "twin_type": "business",
                "contribution": "Business perspective",
                "confidence": 0.7,
                "reasoning": ""  # Empty reasoning
            }
        ]
        
        context = {"additional_info": "Additional context information"}
        
        prompt = adk_agent._construct_rag_prompt(
            query="What is the best approach?",
            retrieved_docs=retrieved_docs,
            twin_consultations=twin_consultations,
            context=context
        )
        
        # Verify prompt contains all expected sections
        assert "You are an AI assistant" in prompt
        assert "## Retrieved Knowledge:" in prompt
        assert "### Source 1 (Similarity: 0.900)" in prompt
        assert "### Source 2 (Similarity: 0.700)" in prompt
        assert "**Title:** Document 1" in prompt
        assert "**Title:** Document 2" in prompt
        assert "Short content" in prompt
        assert "..." in prompt  # Truncation indicator for long content
        
        assert "## AI Digital Twin Insights:" in prompt
        assert "### Technical Twin (Confidence: 0.800)" in prompt
        assert "### Business Twin (Confidence: 0.700)" in prompt
        assert "**Insight:** Technical insight" in prompt
        assert "**Reasoning:** Based on technical expertise" in prompt
        
        assert "## Additional Context:" in prompt
        assert "Additional context information" in prompt
        
        assert "## User Query:" in prompt
        assert "What is the best approach?" in prompt
        
        assert "## Response:" in prompt
    
    def test_construct_rag_prompt_minimal(self, adk_agent):
        """Test RAG prompt construction with minimal components."""
        prompt = adk_agent._construct_rag_prompt(
            query="Simple query",
            retrieved_docs=[],
            twin_consultations=[],
            context=None
        )
        
        # Should contain basic structure
        assert "You are an AI assistant" in prompt
        assert "## User Query:" in prompt
        assert "Simple query" in prompt
        assert "## Response:" in prompt
        
        # Should not contain optional sections
        assert "## Retrieved Knowledge:" not in prompt
        assert "## AI Digital Twin Insights:" not in prompt
        assert "## Additional Context:" not in prompt
    
    async def test_generate_query_embedding(self, adk_agent):
        """Test query embedding generation."""
        with patch('kg_rag.agents.adk_agent.get_settings') as mock_settings:
            mock_settings.return_value.ai_models.embedding_dimension = 384
            
            embedding = await adk_agent._generate_query_embedding("test query")
            
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)
            assert all(0.0 <= x <= 1.0 for x in embedding)
            
            # Same query should produce same embedding (deterministic)
            embedding2 = await adk_agent._generate_query_embedding("test query")
            assert embedding == embedding2
            
            # Different query should produce different embedding
            embedding3 = await adk_agent._generate_query_embedding("different query")
            assert embedding != embedding3
    
    async def test_cache_response(self, adk_agent):
        """Test response caching."""
        response = ADKAgentResponse(
            response_id="test-123",
            agent_id="test-adk-agent",
            query="test query",
            response="test response",
            confidence_score=0.8
        )
        
        # Test caching enabled
        adk_agent.config.enable_caching = True
        await adk_agent._cache_response("test query", response)
        
        # Verify cache entry exists
        cache_key = f"agent_test-adk-agent_query_{hash('test query')}"
        assert cache_key in adk_agent._cache
        
        cached_data = adk_agent._cache[cache_key]
        assert cached_data["response"]["response_id"] == "test-123"
        assert cached_data["ttl"] == 3600
        assert isinstance(cached_data["timestamp"], datetime)
    
    async def test_get_agent_status(self, adk_agent):
        """Test getting agent status."""
        # Add some cache entries
        adk_agent._cache["test1"] = {"data": "value1"}
        adk_agent._cache["test2"] = {"data": "value2"}
        
        status = await adk_agent.get_agent_status()
        
        assert status["agent_id"] == "test-adk-agent"
        assert status["initialized"] is False
        assert status["model_name"] == "gemini-1.5-pro"
        assert status["project_id"] == "test-project"
        assert status["cache_size"] == 2
        assert "configuration" in status
    
    def test_get_safety_settings_enabled(self, adk_agent):
        """Test safety settings when content filtering is enabled."""
        adk_agent.config.enable_content_filtering = True
        
        settings = adk_agent._get_safety_settings()
        
        assert len(settings) == 4
        categories = [setting["category"] for setting in settings]
        assert "HARM_CATEGORY_HATE_SPEECH" in categories
        assert "HARM_CATEGORY_DANGEROUS_CONTENT" in categories
        assert "HARM_CATEGORY_SEXUALLY_EXPLICIT" in categories
        assert "HARM_CATEGORY_HARASSMENT" in categories
        
        # All should have default threshold
        for setting in settings:
            assert setting["threshold"] == "BLOCK_MEDIUM_AND_ABOVE"
    
    def test_get_safety_settings_disabled(self, adk_agent):
        """Test safety settings when content filtering is disabled."""
        adk_agent.config.enable_content_filtering = False
        
        settings = adk_agent._get_safety_settings()
        
        assert settings == []
    
    def test_get_safety_settings_custom(self, adk_agent):
        """Test safety settings with custom configuration."""
        adk_agent.config.enable_content_filtering = True
        adk_agent.config.safety_settings = {
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }
        
        settings = adk_agent._get_safety_settings()
        
        # Find specific settings
        hate_speech_setting = next(
            s for s in settings if s["category"] == "HARM_CATEGORY_HATE_SPEECH"
        )
        dangerous_content_setting = next(
            s for s in settings if s["category"] == "HARM_CATEGORY_DANGEROUS_CONTENT"
        )
        
        assert hate_speech_setting["threshold"] == "BLOCK_LOW_AND_ABOVE"
        assert dangerous_content_setting["threshold"] == "BLOCK_NONE"


@pytest.mark.integration
class TestADKAgentIntegration:
    """Integration tests for ADK agent."""
    
    @pytest.fixture
    def integration_config(self):
        """Configuration for integration tests."""
        return ADKConfiguration(
            project_id="test-project-integration",
            model_name="gemini-1.5-pro",
            enable_caching=False  # Disable caching for clean tests
        )
    
    async def test_full_query_processing_flow(self, integration_config):
        """Test complete query processing flow."""
        # This would require actual Google Cloud setup for full integration
        # For now, test with extensive mocking
        
        with patch('kg_rag.agents.adk_agent.vertexai'), \
             patch('kg_rag.agents.adk_agent.aiplatform'), \
             patch('kg_rag.agents.adk_agent.GenerativeModel') as mock_model_class:
            
            # Setup comprehensive mocks
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = "This is a comprehensive response to the user's query about machine learning."
            mock_response.usage_metadata.total_token_count = 200
            mock_response.usage_metadata.prompt_token_count = 150
            mock_response.usage_metadata.candidates_token_count = 50
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = "STOP"
            
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            # Create agent with mocked components
            mock_vector_ops = Mock(spec=VectorGraphOperations)
            mock_vector_ops.hybrid_search.return_value = [
                {
                    "node_id": "ml_doc_1",
                    "title": "Introduction to Machine Learning",
                    "content": "Machine learning is a method of data analysis...",
                    "similarity_score": 0.92,
                    "node_type": "document"
                }
            ]
            
            mock_twins = Mock(spec=TwinOrchestrator)
            mock_twin_result = Mock()
            mock_twin_result.contributing_twins = [
                {
                    "twin_id": "ml_expert",
                    "twin_type": "technical",
                    "contribution": "Machine learning involves training algorithms on data...",
                    "confidence": 0.88,
                    "reasoning": "Based on years of ML research experience"
                }
            ]
            mock_twin_result.synthesized_response = "Combined expert knowledge on ML"
            mock_twin_result.confidence_score = 0.87
            mock_twins.process_query.return_value = mock_twin_result
            
            # Create and test agent
            agent = ADKAgent(
                agent_id="integration-test-agent",
                config=integration_config,
                twin_orchestrator=mock_twins,
                vector_operations=mock_vector_ops,
                query_builder=Mock()
            )
            
            # Process query
            with patch('asyncio.to_thread', return_value=mock_response):
                response = await agent.process_query(
                    query="What is machine learning and how does it work?",
                    context={"domain": "technology"},
                    user_id="test_user_123",
                    enable_twins=True,
                    enable_retrieval=True
                )
            
            # Verify complete response
            assert isinstance(response, ADKAgentResponse)
            assert response.agent_id == "integration-test-agent"
            assert response.query == "What is machine learning and how does it work?"
            assert response.response == "This is a comprehensive response to the user's query about machine learning."
            assert response.confidence_score > 0.8  # Should be high with good retrieval and twins
            assert response.total_tokens == 200
            assert response.model_used == "gemini-1.5-pro"
            
            # Verify retrieved documents
            assert len(response.retrieved_documents) == 1
            assert response.retrieved_documents[0]["title"] == "Introduction to Machine Learning"
            assert response.retrieved_documents[0]["similarity_score"] == 0.92
            
            # Verify twin consultations
            assert len(response.twin_consultations) == 2  # Individual + synthesized
            assert response.twin_consultations[0]["twin_type"] == "technical"
            assert response.twin_consultations[0]["confidence"] == 0.88
            assert response.twin_consultations[1]["twin_type"] == "synthesized"
    
    async def test_error_recovery_and_fallbacks(self, integration_config):
        """Test error recovery and fallback mechanisms."""
        with patch('kg_rag.agents.adk_agent.vertexai'), \
             patch('kg_rag.agents.adk_agent.aiplatform'), \
             patch('kg_rag.agents.adk_agent.GenerativeModel'):
            
            # Create agent with failing components
            mock_vector_ops = Mock(spec=VectorGraphOperations)
            mock_vector_ops.hybrid_search.side_effect = Exception("Vector search failed")
            
            mock_twins = Mock(spec=TwinOrchestrator)
            mock_twins.process_query.side_effect = Exception("Twin orchestrator failed")
            
            agent = ADKAgent(
                agent_id="error-test-agent",
                config=integration_config,
                twin_orchestrator=mock_twins,
                vector_operations=mock_vector_ops
            )
            
            # Mock successful model response despite component failures
            mock_response = Mock()
            mock_response.text = "Fallback response without retrieval or twins"
            mock_response.usage_metadata.total_token_count = 100
            mock_response.candidates = [Mock()]
            mock_response.candidates[0].finish_reason = "STOP"
            
            with patch('asyncio.to_thread', return_value=mock_response):
                response = await agent.process_query(
                    query="Test query for error handling",
                    enable_twins=True,
                    enable_retrieval=True
                )
            
            # Should still get response despite component failures
            assert isinstance(response, ADKAgentResponse)
            assert response.response == "Fallback response without retrieval or twins"
            assert response.retrieved_documents == []  # Failed retrieval
            assert response.twin_consultations == []  # Failed twin consultation
            assert response.confidence_score < 0.7  # Lower confidence due to failures