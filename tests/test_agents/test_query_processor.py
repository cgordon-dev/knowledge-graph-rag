"""
Test suite for Query Processor.

Tests query parsing, classification, entity extraction, and processing pipeline
for the Google ADK agent integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any, List

from kg_rag.agents.query_processor import (
    QueryProcessor, ParsedQuery, QueryType, QueryComplexity, QueryIntent,
    QueryMetrics
)
from kg_rag.core.exceptions import QueryProcessingError, ValidationError
from kg_rag.graph_schema.node_models import NodeType


class TestQueryEnums:
    """Test query classification enums."""
    
    def test_query_type_enum(self):
        """Test QueryType enum values."""
        assert QueryType.FACTUAL == "factual"
        assert QueryType.ANALYTICAL == "analytical"
        assert QueryType.PROCEDURAL == "procedural"
        assert QueryType.EXPLORATORY == "exploratory"
        assert QueryType.COMPARATIVE == "comparative"
        assert QueryType.TEMPORAL == "temporal"
        assert QueryType.CAUSAL == "causal"
        assert QueryType.SYNTHESIS == "synthesis"
    
    def test_query_complexity_enum(self):
        """Test QueryComplexity enum values."""
        assert QueryComplexity.SIMPLE == "simple"
        assert QueryComplexity.MODERATE == "moderate"
        assert QueryComplexity.COMPLEX == "complex"
        assert QueryComplexity.EXPERT == "expert"
    
    def test_query_intent_enum(self):
        """Test QueryIntent enum values."""
        assert QueryIntent.SEARCH == "search"
        assert QueryIntent.EXPLANATION == "explanation"
        assert QueryIntent.GUIDANCE == "guidance"
        assert QueryIntent.ANALYSIS == "analysis"
        assert QueryIntent.COMPARISON == "comparison"
        assert QueryIntent.TROUBLESHOOTING == "troubleshooting"
        assert QueryIntent.PLANNING == "planning"
        assert QueryIntent.LEARNING == "learning"


class TestParsedQuery:
    """Test ParsedQuery model."""
    
    def test_valid_parsed_query(self):
        """Test creating valid parsed query."""
        query = ParsedQuery(
            original_query="What is machine learning?",
            normalized_query="what is machine learning",
            query_type=QueryType.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            intent=QueryIntent.EXPLANATION,
            confidence=0.85,
            entities=["machine learning"],
            concepts=["artificial intelligence", "algorithms"],
            keywords=["machine", "learning", "algorithms"]
        )
        
        assert query.original_query == "What is machine learning?"
        assert query.normalized_query == "what is machine learning"
        assert query.query_type == QueryType.FACTUAL
        assert query.complexity == QueryComplexity.SIMPLE
        assert query.intent == QueryIntent.EXPLANATION
        assert query.confidence == 0.85
        assert len(query.entities) == 1
        assert len(query.concepts) == 2
        assert len(query.keywords) == 3
        assert isinstance(query.processing_timestamp, datetime)
        assert len(query.query_id) > 0
    
    def test_default_values(self):
        """Test default values in ParsedQuery."""
        query = ParsedQuery(
            original_query="test",
            normalized_query="test",
            query_type=QueryType.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            intent=QueryIntent.SEARCH,
            confidence=0.5
        )
        
        assert query.entities == []
        assert query.concepts == []
        assert query.keywords == []
        assert query.temporal_references == []
        assert query.domain_hints == []
        assert query.node_type_filters == []
        assert query.required_expertise == []
        assert query.processing_metrics is None
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence
        query = ParsedQuery(
            original_query="test",
            normalized_query="test",
            query_type=QueryType.FACTUAL,
            complexity=QueryComplexity.SIMPLE,
            intent=QueryIntent.SEARCH,
            confidence=0.7
        )
        assert query.confidence == 0.7
        
        # Invalid confidence - too high
        with pytest.raises(ValueError):
            ParsedQuery(
                original_query="test",
                normalized_query="test",
                query_type=QueryType.FACTUAL,
                complexity=QueryComplexity.SIMPLE,
                intent=QueryIntent.SEARCH,
                confidence=1.5
            )
        
        # Invalid confidence - negative
        with pytest.raises(ValueError):
            ParsedQuery(
                original_query="test",
                normalized_query="test",
                query_type=QueryType.FACTUAL,
                complexity=QueryComplexity.SIMPLE,
                intent=QueryIntent.SEARCH,
                confidence=-0.1
            )


class TestQueryProcessor:
    """Test QueryProcessor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create query processor instance."""
        return QueryProcessor()
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert isinstance(processor, QueryProcessor)
        assert hasattr(processor, '_query_patterns')
        assert hasattr(processor, '_entity_patterns')
        assert hasattr(processor, '_complexity_indicators')
        assert processor._processing_stats["total_queries"] == 0
    
    async def test_normalize_query_basic(self, processor):
        """Test basic query normalization."""
        # Test whitespace normalization
        normalized = await processor._normalize_query("  What    is   machine learning?  ")
        assert normalized == "What is machine learning?"
        
        # Test abbreviation expansion
        normalized = await processor._normalize_query("What's machine learning?")
        assert normalized == "What is machine learning?"
        
        normalized = await processor._normalize_query("Can't understand this")
        assert normalized == "cannot understand this"
    
    async def test_normalize_query_contractions(self, processor):
        """Test contraction expansion in normalization."""
        test_cases = [
            ("how's this work", "how is this work"),
            ("where's the file", "where is the file"),
            ("who's responsible", "who is responsible"),
            ("when's the deadline", "when is the deadline"),
            ("why's it failing", "why is it failing"),
            ("won't work", "will not work"),
            ("shouldn't do", "should not do"),
            ("wouldn't recommend", "would not recommend")
        ]
        
        for input_query, expected in test_cases:
            normalized = await processor._normalize_query(input_query)
            assert normalized == expected
    
    async def test_classify_query_type_factual(self, processor):
        """Test factual query type classification."""
        factual_queries = [
            "what is machine learning",
            "define artificial intelligence",
            "who is the CEO of the company",
            "where is the database located"
        ]
        
        for query in factual_queries:
            query_type = await processor._classify_query_type(query)
            assert query_type == QueryType.FACTUAL
    
    async def test_classify_query_type_procedural(self, processor):
        """Test procedural query type classification."""
        procedural_queries = [
            "how to implement machine learning",
            "steps to deploy the application",
            "process for code review",
            "procedure for data backup"
        ]
        
        for query in procedural_queries:
            query_type = await processor._classify_query_type(query)
            assert query_type == QueryType.PROCEDURAL
    
    async def test_classify_query_type_analytical(self, processor):
        """Test analytical query type classification."""
        analytical_queries = [
            "analyze the performance metrics",
            "evaluate the security risks",
            "assess the project timeline",
            "why does the system crash"
        ]
        
        for query in analytical_queries:
            query_type = await processor._classify_query_type(query)
            assert query_type == QueryType.ANALYTICAL
    
    async def test_classify_query_type_comparative(self, processor):
        """Test comparative query type classification."""
        comparative_queries = [
            "compare React and Vue frameworks",
            "difference between SQL and NoSQL",
            "Python versus Java performance",
            "which is better for deployment"
        ]
        
        for query in comparative_queries:
            query_type = await processor._classify_query_type(query)
            assert query_type == QueryType.COMPARATIVE
    
    async def test_classify_complexity_simple(self, processor):
        """Test simple complexity classification."""
        simple_queries = [
            "what is AI",
            "define API",
            "who created Python"
        ]
        
        for query in simple_queries:
            complexity = await processor._classify_complexity(query, QueryType.FACTUAL)
            assert complexity == QueryComplexity.SIMPLE
    
    async def test_classify_complexity_complex(self, processor):
        """Test complex complexity classification."""
        # Long query with multiple complex indicators
        complex_query = (
            "analyze the comprehensive performance metrics of multiple "
            "microservices architectures and provide detailed recommendations "
            "for system optimization and scalability improvements"
        )
        
        complexity = await processor._classify_complexity(complex_query, QueryType.ANALYTICAL)
        assert complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
    
    async def test_classify_complexity_expert(self, processor):
        """Test expert complexity classification."""
        expert_query = (
            "provide a comprehensive analysis of various distributed system "
            "architectures including detailed performance comparisons, "
            "thoroughly evaluate multiple database sharding strategies, "
            "and extensively document the implementation complexities "
            "for large-scale enterprise deployments"
        )
        
        complexity = await processor._classify_complexity(expert_query, QueryType.SYNTHESIS)
        assert complexity == QueryComplexity.EXPERT
    
    async def test_classify_intent_search(self, processor):
        """Test search intent classification."""
        search_queries = [
            "find the configuration file",
            "search for error logs",
            "look for documentation",
            "locate the database schema"
        ]
        
        for query in search_queries:
            intent = await processor._classify_intent(query, QueryType.FACTUAL)
            assert intent == QueryIntent.SEARCH
    
    async def test_classify_intent_guidance(self, processor):
        """Test guidance intent classification."""
        guidance_queries = [
            "how to setup the environment",
            "guide me through deployment",
            "instructions for configuration",
            "tutorial on API usage"
        ]
        
        for query in guidance_queries:
            intent = await processor._classify_intent(query, QueryType.PROCEDURAL)
            assert intent == QueryIntent.GUIDANCE
    
    async def test_classify_intent_troubleshooting(self, processor):
        """Test troubleshooting intent classification."""
        troubleshooting_queries = [
            "fix the connection problem",
            "solve the authentication issue",
            "error in data processing",
            "system performance problem"
        ]
        
        for query in troubleshooting_queries:
            intent = await processor._classify_intent(query, QueryType.ANALYTICAL)
            assert intent == QueryIntent.TROUBLESHOOTING
    
    async def test_extract_elements_entities(self, processor):
        """Test entity extraction."""
        query = "What is the status of Project Alpha and Team Beta collaboration?"
        entities, concepts, keywords = await processor._extract_elements(query)
        
        # Should extract capitalized words as entities
        assert "Project" in entities or "Alpha" in entities
        assert "Team" in entities or "Beta" in entities
    
    async def test_extract_elements_concepts(self, processor):
        """Test concept extraction."""
        query = "analyze the database performance and system architecture design"
        entities, concepts, keywords = await processor._extract_elements(query)
        
        # Should extract domain-specific terms as concepts
        assert "database" in concepts
        assert "performance" in concepts
        assert "system" in concepts
        assert "architecture" in concepts
        assert "design" in concepts
    
    async def test_extract_elements_keywords(self, processor):
        """Test keyword extraction."""
        query = "How to implement machine learning algorithms for data analysis?"
        entities, concepts, keywords = await processor._extract_elements(query)
        
        # Should extract meaningful keywords (excluding stop words)
        meaningful_keywords = ["implement", "machine", "learning", "algorithms", "data", "analysis"]
        for keyword in meaningful_keywords:
            assert keyword in keywords
        
        # Should exclude stop words
        stop_words = ["how", "to", "for"]
        for stop_word in stop_words:
            assert stop_word not in keywords
    
    async def test_extract_temporal_references(self, processor):
        """Test temporal reference extraction."""
        temporal_queries = [
            ("what happened yesterday", ["yesterday"]),
            ("schedule for next week", ["next week"]),
            ("data from last month", ["last month"]),
            ("report for this quarter", ["this quarter"]),
            ("events in 2023", ["in 2023"]),
            ("currently running processes", ["currently"]),
            ("recently updated files", ["recently"])
        ]
        
        for query, expected_refs in temporal_queries:
            temporal_refs = await processor._extract_temporal_references(query)
            for expected in expected_refs:
                assert expected in temporal_refs
    
    async def test_identify_domains(self, processor):
        """Test domain identification."""
        # Technology domain
        tech_query = "analyze the database system architecture"
        concepts = ["database", "system", "architecture"]
        domains = await processor._identify_domains(tech_query, concepts)
        assert "technology" in domains
        
        # Security domain
        security_query = "audit security compliance and vulnerability assessment"
        concepts = ["security", "compliance", "vulnerability"]
        domains = await processor._identify_domains(security_query, concepts)
        assert "security" in domains
        
        # Business domain
        business_query = "optimize business process workflow analysis"
        concepts = ["process", "workflow", "analysis", "optimization"]
        domains = await processor._identify_domains(business_query, concepts)
        assert "business" in domains
    
    async def test_suggest_node_types(self, processor):
        """Test node type suggestions."""
        # Test with entities
        entities = ["Company", "Project"]
        concepts = ["database"]
        node_types = await processor._suggest_node_types(
            QueryType.FACTUAL, entities, concepts
        )
        
        # Should always include documents and chunks
        assert NodeType.DOCUMENT in node_types
        assert NodeType.CHUNK in node_types
        
        # Should include entities and concepts based on extraction
        assert NodeType.ENTITY in node_types
        assert NodeType.CONCEPT in node_types
    
    async def test_suggest_node_types_procedural(self, processor):
        """Test node type suggestions for procedural queries."""
        node_types = await processor._suggest_node_types(
            QueryType.PROCEDURAL, [], ["process"]
        )
        
        assert NodeType.DOCUMENT in node_types
        assert NodeType.CHUNK in node_types
        assert NodeType.PROCESS in node_types
        assert NodeType.WORKFLOW in node_types
    
    async def test_suggest_node_types_temporal(self, processor):
        """Test node type suggestions for temporal queries."""
        node_types = await processor._suggest_node_types(
            QueryType.TEMPORAL, ["Event"], []
        )
        
        assert NodeType.EVENT in node_types
    
    async def test_identify_required_expertise(self, processor):
        """Test required expertise identification."""
        # Complex analytical query in security domain
        expertise = await processor._identify_required_expertise(
            QueryType.ANALYTICAL,
            QueryComplexity.COMPLEX,
            ["security", "technology"]
        )
        
        assert "expert" in expertise
        assert "analyst" in expertise
        assert "security" in expertise
        assert "technical" in expertise
    
    async def test_identify_required_expertise_procedural(self, processor):
        """Test expertise for procedural queries."""
        expertise = await processor._identify_required_expertise(
            QueryType.PROCEDURAL,
            QueryComplexity.MODERATE,
            ["business"]
        )
        
        assert "process" in expertise
        assert "business" in expertise
    
    async def test_calculate_complexity_score(self, processor):
        """Test complexity score calculation."""
        # Simple query
        simple_score = await processor._calculate_complexity_score(
            "what is AI",
            [],
            [],
            QueryType.FACTUAL,
            QueryComplexity.SIMPLE
        )
        assert 0.1 <= simple_score <= 0.3
        
        # Complex analytical query with entities and concepts
        complex_score = await processor._calculate_complexity_score(
            "comprehensive analysis of distributed systems",
            ["System1", "System2"],
            ["architecture", "performance", "scalability"],
            QueryType.ANALYTICAL,
            QueryComplexity.COMPLEX
        )
        assert complex_score > simple_score
        assert 0.8 <= complex_score <= 1.0
    
    async def test_calculate_classification_confidence(self, processor):
        """Test classification confidence calculation."""
        # Clear factual query
        high_confidence = await processor._calculate_classification_confidence(
            "what is machine learning",
            QueryType.FACTUAL,
            QueryComplexity.SIMPLE,
            QueryIntent.EXPLANATION
        )
        assert high_confidence >= 0.8
        
        # Ambiguous short query
        low_confidence = await processor._calculate_classification_confidence(
            "test",
            QueryType.FACTUAL,
            QueryComplexity.SIMPLE,
            QueryIntent.SEARCH
        )
        assert low_confidence < high_confidence
    
    async def test_validate_parsed_query_valid(self, processor):
        """Test validation of valid queries."""
        # Should not raise exception for valid query
        await processor._validate_parsed_query(
            "This is a valid query with reasonable length",
            ["Entity1"],
            ["concept1"]
        )
    
    async def test_validate_parsed_query_too_short(self, processor):
        """Test validation failure for too short query."""
        with pytest.raises(ValidationError, match="Query too short"):
            await processor._validate_parsed_query("ab", [], [])
    
    async def test_validate_parsed_query_too_long(self, processor):
        """Test validation failure for too long query."""
        long_query = "a" * 1001  # Exceeds 1000 character limit
        with pytest.raises(ValidationError, match="Query too long"):
            await processor._validate_parsed_query(long_query, [], [])
    
    async def test_validate_parsed_query_harmful_content(self, processor):
        """Test validation failure for potentially harmful content."""
        harmful_queries = [
            "check this <script>alert('xss')</script>",
            "run javascript:alert('test')",
            "execute eval(malicious_code)",
            "process exec(dangerous_command)"
        ]
        
        for harmful_query in harmful_queries:
            with pytest.raises(ValidationError, match="potentially harmful content"):
                await processor._validate_parsed_query(harmful_query, [], [])
    
    async def test_process_query_complete_flow(self, processor):
        """Test complete query processing flow."""
        query = "How to implement secure authentication in a distributed microservices architecture?"
        
        parsed = await processor.process_query(query)
        
        # Verify basic structure
        assert isinstance(parsed, ParsedQuery)
        assert parsed.original_query == query
        assert len(parsed.normalized_query) > 0
        assert parsed.query_type in QueryType
        assert parsed.complexity in QueryComplexity
        assert parsed.intent in QueryIntent
        assert 0.0 <= parsed.confidence <= 1.0
        
        # Verify processing metrics
        assert parsed.processing_metrics is not None
        assert parsed.processing_metrics.processing_time_ms > 0
        assert parsed.processing_metrics.total_tokens > 0
        assert 0.0 <= parsed.processing_metrics.complexity_score <= 1.0
        
        # Verify extracted elements
        assert len(parsed.keywords) > 0
        assert len(parsed.domain_hints) > 0
        assert len(parsed.node_type_filters) > 0
        
        # Should identify security and technology domains
        assert any(domain in ["security", "technology"] for domain in parsed.domain_hints)
        
        # Should be classified as procedural with moderate/complex complexity
        assert parsed.query_type == QueryType.PROCEDURAL
        assert parsed.complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX, QueryComplexity.EXPERT]
        assert parsed.intent == QueryIntent.GUIDANCE
    
    async def test_process_query_with_context(self, processor):
        """Test query processing with user context."""
        query = "analyze system performance"
        context = {
            "user_domain": "devops",
            "previous_queries": ["deployment issues"],
            "user_expertise": "expert"
        }
        
        parsed = await processor.process_query(query, context)
        
        assert isinstance(parsed, ParsedQuery)
        assert parsed.query_type == QueryType.ANALYTICAL
        # Context should influence processing but doesn't change core logic in current implementation
    
    async def test_process_query_error_handling(self, processor):
        """Test error handling in query processing."""
        # Test with invalid input that might cause processing errors
        with patch.object(processor, '_normalize_query', side_effect=Exception("Normalization failed")):
            with pytest.raises(QueryProcessingError, match="Failed to process query"):
                await processor.process_query("test query")
    
    async def test_batch_process_queries(self, processor):
        """Test batch query processing."""
        queries = [
            "what is machine learning",
            "how to deploy applications",
            "compare databases",
            "analyze system performance",
            "invalid query that might fail"
        ]
        
        # Mock one query to fail
        original_process = processor.process_query
        
        async def mock_process_query(query, context=None, options=None):
            if "invalid query" in query:
                raise Exception("Processing failed")
            return await original_process(query, context, options)
        
        with patch.object(processor, 'process_query', side_effect=mock_process_query):
            results = await processor.batch_process_queries(queries)
        
        # Should get results for successful queries only
        assert len(results) == 4  # 5 queries - 1 failed
        
        for result in results:
            assert isinstance(result, ParsedQuery)
    
    async def test_get_processing_stats(self, processor):
        """Test getting processing statistics."""
        # Process a few queries to generate stats
        await processor.process_query("test query 1")
        await processor.process_query("test query 2")
        
        stats = await processor.get_processing_stats()
        
        assert stats["total_queries_processed"] == 2
        assert stats["average_processing_time_ms"] > 0
        assert "classification_accuracy" in stats
    
    def test_update_processing_stats(self, processor):
        """Test processing statistics updates."""
        # Create mock metrics
        metrics = QueryMetrics(
            processing_time_ms=100.0,
            parsing_time_ms=20.0,
            classification_time_ms=30.0,
            validation_time_ms=10.0,
            entity_extraction_time_ms=25.0,
            total_tokens=50,
            complexity_score=0.6,
            confidence_score=0.8
        )
        
        # Update stats
        processor._update_processing_stats(metrics)
        
        assert processor._processing_stats["total_queries"] == 1
        assert processor._processing_stats["avg_processing_time"] == 100.0
        
        # Update with another query
        metrics2 = QueryMetrics(
            processing_time_ms=200.0,
            parsing_time_ms=40.0,
            classification_time_ms=60.0,
            validation_time_ms=20.0,
            entity_extraction_time_ms=50.0,
            total_tokens=100,
            complexity_score=0.8,
            confidence_score=0.9
        )
        
        processor._update_processing_stats(metrics2)
        
        assert processor._processing_stats["total_queries"] == 2
        assert processor._processing_stats["avg_processing_time"] == 150.0  # (100 + 200) / 2


@pytest.mark.integration
class TestQueryProcessorIntegration:
    """Integration tests for query processor."""
    
    @pytest.fixture
    def processor(self):
        """Create processor for integration tests."""
        return QueryProcessor()
    
    async def test_real_world_queries(self, processor):
        """Test processing of real-world queries."""
        real_queries = [
            # Factual queries
            "What is the current version of our API?",
            "Who is responsible for the user authentication module?",
            
            # Procedural queries
            "How do I configure SSL certificates for the web server?",
            "What are the steps to deploy a new microservice?",
            
            # Analytical queries
            "Why is the database query performance degrading?",
            "What are the security implications of using third-party libraries?",
            
            # Comparative queries
            "Should we use Redis or Memcached for caching?",
            "What's the difference between REST and GraphQL APIs?",
            
            # Complex queries
            "Analyze the impact of implementing a new caching layer on system performance, considering both memory usage and response times across different microservices, and provide recommendations for optimal configuration."
        ]
        
        for query in real_queries:
            parsed = await processor.process_query(query)
            
            # All queries should be processed successfully
            assert isinstance(parsed, ParsedQuery)
            assert len(parsed.normalized_query) > 0
            assert parsed.confidence > 0.0
            assert len(parsed.keywords) > 0
            
            # Processing should be reasonably fast (< 1 second)
            if parsed.processing_metrics:
                assert parsed.processing_metrics.processing_time_ms < 1000
    
    async def test_edge_case_queries(self, processor):
        """Test edge cases in query processing."""
        edge_cases = [
            # Very short query
            "API?",
            
            # Query with special characters
            "How to use @decorator in Python?",
            
            # Query with numbers and symbols
            "Configure port 8080 for HTTP/2 protocol",
            
            # Mixed case query
            "WhAt Is ThE dIfFeReNcE bEtWeEn SQL aNd NoSQL?",
            
            # Query with multiple questions
            "How to setup Redis? What about configuration? Any security concerns?",
            
            # Technical jargon heavy query
            "Implement OAuth2 PKCE flow with JWT tokens for SPA authentication"
        ]
        
        for query in edge_cases:
            parsed = await processor.process_query(query)
            
            # Should handle all edge cases gracefully
            assert isinstance(parsed, ParsedQuery)
            assert len(parsed.normalized_query) > 0
    
    async def test_multilingual_support_preparation(self, processor):
        """Test preparation for multilingual support."""
        # Currently English only, but structure should support expansion
        
        # Test with some common non-English words that might appear in technical contexts
        mixed_queries = [
            "How to configure nginx server?",  # nginx is a proper noun
            "Implement OAuth authentication",   # OAuth is an acronym
            "Setup PostgreSQL database"        # PostgreSQL is a proper noun
        ]
        
        for query in mixed_queries:
            parsed = await processor.process_query(query)
            assert isinstance(parsed, ParsedQuery)
            
            # Should extract proper nouns as entities
            technical_terms = ["nginx", "OAuth", "PostgreSQL"]
            found_terms = [term for term in technical_terms if any(
                term.lower() in entity.lower() for entity in parsed.entities
            ) or any(
                term.lower() in keyword.lower() for keyword in parsed.keywords
            )]
            
            # Should find at least some technical terms
            assert len(found_terms) > 0
    
    async def test_performance_benchmarking(self, processor):
        """Test performance benchmarking of query processing."""
        # Generate various query types for performance testing
        test_queries = []
        
        # Simple queries
        test_queries.extend([
            f"what is {concept}" for concept in ["API", "database", "security", "performance"]
        ])
        
        # Medium complexity
        test_queries.extend([
            f"how to implement {feature}" for feature in ["authentication", "caching", "logging", "monitoring"]
        ])
        
        # Complex queries
        test_queries.extend([
            f"analyze the {aspect} of distributed systems considering {factor}" 
            for aspect in ["performance", "security", "reliability"]
            for factor in ["scalability", "maintainability", "cost"]
        ])
        
        # Process all queries and collect timing data
        processing_times = []
        
        for query in test_queries:
            parsed = await processor.process_query(query)
            if parsed.processing_metrics:
                processing_times.append(parsed.processing_metrics.processing_time_ms)
        
        # Verify performance characteristics
        assert len(processing_times) > 0
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        # Most queries should process quickly
        assert avg_time < 100  # Average under 100ms
        assert max_time < 500  # No query should take more than 500ms
        
        # 95th percentile should be reasonable
        processing_times.sort()
        percentile_95 = processing_times[int(len(processing_times) * 0.95)]
        assert percentile_95 < 200  # 95% of queries under 200ms