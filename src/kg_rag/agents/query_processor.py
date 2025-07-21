"""
Query Processing Pipeline for Google ADK Agent Integration.

Handles query parsing, routing, preprocessing, and result formatting
for the Knowledge Graph-RAG system with ADK agents.
"""

import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from enum import Enum
from dataclasses import dataclass
import uuid

import structlog
from pydantic import BaseModel, Field, validator

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import QueryProcessingError, ValidationError
from kg_rag.graph_schema.node_models import NodeType

logger = structlog.get_logger(__name__)


class QueryType(str, Enum):
    """Types of queries the system can handle."""
    
    FACTUAL = "factual"           # Direct information retrieval
    ANALYTICAL = "analytical"     # Analysis and reasoning
    PROCEDURAL = "procedural"     # How-to and process queries
    EXPLORATORY = "exploratory"   # Open-ended exploration
    COMPARATIVE = "comparative"   # Comparison between concepts
    TEMPORAL = "temporal"         # Time-based queries
    CAUSAL = "causal"            # Cause-and-effect queries
    SYNTHESIS = "synthesis"       # Combining multiple sources


class QueryComplexity(str, Enum):
    """Query complexity levels."""
    
    SIMPLE = "simple"         # Single concept, direct answer
    MODERATE = "moderate"     # Multiple concepts, some reasoning
    COMPLEX = "complex"       # Multi-step reasoning, synthesis
    EXPERT = "expert"         # Domain expertise required


class QueryIntent(str, Enum):
    """User intent classification."""
    
    SEARCH = "search"              # Information seeking
    EXPLANATION = "explanation"    # Understanding concepts
    GUIDANCE = "guidance"          # Step-by-step instructions
    ANALYSIS = "analysis"          # Deep analysis needed
    COMPARISON = "comparison"      # Compare options
    TROUBLESHOOTING = "troubleshooting"  # Problem solving
    PLANNING = "planning"          # Strategic planning
    LEARNING = "learning"          # Educational content


@dataclass
class QueryMetrics:
    """Metrics for query processing performance."""
    
    processing_time_ms: float
    parsing_time_ms: float
    classification_time_ms: float
    validation_time_ms: float
    entity_extraction_time_ms: float
    total_tokens: int
    complexity_score: float
    confidence_score: float


class ParsedQuery(BaseModel):
    """Parsed and enriched query structure."""
    
    # Core Query Information
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = Field(..., description="Original user query")
    normalized_query: str = Field(..., description="Normalized query text")
    
    # Classification
    query_type: QueryType = Field(..., description="Type of query")
    complexity: QueryComplexity = Field(..., description="Query complexity level")
    intent: QueryIntent = Field(..., description="User intent")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    
    # Extracted Elements
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    concepts: List[str] = Field(default_factory=list, description="Extracted concepts")
    keywords: List[str] = Field(default_factory=list, description="Key terms")
    temporal_references: List[str] = Field(default_factory=list, description="Time references")
    
    # Context and Filters
    domain_hints: List[str] = Field(default_factory=list, description="Domain indicators")
    node_type_filters: List[NodeType] = Field(default_factory=list, description="Suggested node types")
    required_expertise: List[str] = Field(default_factory=list, description="Required expert types")
    
    # Processing Metadata
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_metrics: Optional[QueryMetrics] = Field(None, description="Processing metrics")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class QueryProcessor:
    """Advanced query processing pipeline for ADK agents."""
    
    def __init__(self):
        """Initialize query processor."""
        self.settings = get_settings()
        
        # Initialize classification patterns
        self._query_patterns = self._load_query_patterns()
        self._entity_patterns = self._load_entity_patterns()
        self._complexity_indicators = self._load_complexity_indicators()
        
        # Performance tracking
        self._processing_stats = {
            "total_queries": 0,
            "avg_processing_time": 0.0,
            "classification_accuracy": 0.0
        }
        
        logger.info("Query processor initialized")
    
    async def process_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> ParsedQuery:
        """Process and enrich a user query.
        
        Args:
            query: Raw user query
            user_context: Additional user context
            processing_options: Processing configuration
            
        Returns:
            Parsed and enriched query
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(
                "Processing query",
                query_length=len(query),
                has_context=bool(user_context)
            )
            
            # Step 1: Query normalization and cleaning
            parsing_start = datetime.utcnow()
            normalized_query = await self._normalize_query(query)
            parsing_time = (datetime.utcnow() - parsing_start).total_seconds() * 1000
            
            # Step 2: Query classification
            classification_start = datetime.utcnow()
            query_type, complexity, intent, confidence = await self._classify_query(
                normalized_query, user_context
            )
            classification_time = (datetime.utcnow() - classification_start).total_seconds() * 1000
            
            # Step 3: Entity and concept extraction
            extraction_start = datetime.utcnow()
            entities, concepts, keywords = await self._extract_elements(normalized_query)
            temporal_refs = await self._extract_temporal_references(normalized_query)
            extraction_time = (datetime.utcnow() - extraction_start).total_seconds() * 1000
            
            # Step 4: Context enrichment
            domain_hints = await self._identify_domains(normalized_query, concepts)
            node_type_filters = await self._suggest_node_types(query_type, entities, concepts)
            required_expertise = await self._identify_required_expertise(
                query_type, complexity, domain_hints
            )
            
            # Step 5: Validation
            validation_start = datetime.utcnow()
            await self._validate_parsed_query(normalized_query, entities, concepts)
            validation_time = (datetime.utcnow() - validation_start).total_seconds() * 1000
            
            # Step 6: Calculate metrics
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            complexity_score = await self._calculate_complexity_score(
                normalized_query, entities, concepts, query_type, complexity
            )
            
            metrics = QueryMetrics(
                processing_time_ms=total_time,
                parsing_time_ms=parsing_time,
                classification_time_ms=classification_time,
                validation_time_ms=validation_time,
                entity_extraction_time_ms=extraction_time,
                total_tokens=len(normalized_query.split()),
                complexity_score=complexity_score,
                confidence_score=confidence
            )
            
            # Create parsed query
            parsed_query = ParsedQuery(
                original_query=query,
                normalized_query=normalized_query,
                query_type=query_type,
                complexity=complexity,
                intent=intent,
                confidence=confidence,
                entities=entities,
                concepts=concepts,
                keywords=keywords,
                temporal_references=temporal_refs,
                domain_hints=domain_hints,
                node_type_filters=node_type_filters,
                required_expertise=required_expertise,
                processing_metrics=metrics
            )
            
            # Update statistics
            self._update_processing_stats(metrics)
            
            logger.info(
                "Query processed successfully",
                query_id=parsed_query.query_id,
                query_type=query_type.value,
                complexity=complexity.value,
                intent=intent.value,
                processing_time_ms=total_time,
                entities_found=len(entities),
                concepts_found=len(concepts)
            )
            
            return parsed_query
            
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(error_msg, query=query, error=str(e))
            raise QueryProcessingError(error_msg) from e
    
    async def _normalize_query(self, query: str) -> str:
        """Normalize and clean query text.
        
        Args:
            query: Raw query text
            
        Returns:
            Normalized query
        """
        # Basic cleaning
        normalized = query.strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common abbreviations
        abbreviations = {
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "who's": "who is",
            "when's": "when is",
            "why's": "why is",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "wouldn't": "would not"
        }
        
        for abbrev, expansion in abbreviations.items():
            normalized = re.sub(
                rf'\b{re.escape(abbrev)}\b', 
                expansion, 
                normalized, 
                flags=re.IGNORECASE
            )
        
        return normalized
    
    async def _classify_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[QueryType, QueryComplexity, QueryIntent, float]:
        """Classify query type, complexity, and intent.
        
        Args:
            query: Normalized query
            context: User context
            
        Returns:
            Tuple of (query_type, complexity, intent, confidence)
        """
        query_lower = query.lower()
        
        # Classify query type
        query_type = await self._classify_query_type(query_lower)
        
        # Classify complexity
        complexity = await self._classify_complexity(query_lower, query_type)
        
        # Classify intent
        intent = await self._classify_intent(query_lower, query_type)
        
        # Calculate overall classification confidence
        confidence = await self._calculate_classification_confidence(
            query_lower, query_type, complexity, intent
        )
        
        return query_type, complexity, intent, confidence
    
    async def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query."""
        
        # Factual queries
        if any(pattern in query for pattern in ["what is", "define", "who is", "where is"]):
            return QueryType.FACTUAL
        
        # Analytical queries
        if any(pattern in query for pattern in ["analyze", "evaluate", "assess", "why does"]):
            return QueryType.ANALYTICAL
        
        # Procedural queries
        if any(pattern in query for pattern in ["how to", "steps to", "process for", "procedure"]):
            return QueryType.PROCEDURAL
        
        # Exploratory queries
        if any(pattern in query for pattern in ["explore", "discover", "find out", "tell me about"]):
            return QueryType.EXPLORATORY
        
        # Comparative queries
        if any(pattern in query for pattern in ["compare", "difference", "versus", "vs", "better"]):
            return QueryType.COMPARATIVE
        
        # Temporal queries
        if any(pattern in query for pattern in ["when", "timeline", "history", "evolution"]):
            return QueryType.TEMPORAL
        
        # Causal queries
        if any(pattern in query for pattern in ["why", "cause", "reason", "because", "due to"]):
            return QueryType.CAUSAL
        
        # Synthesis queries
        if any(pattern in query for pattern in ["synthesize", "combine", "integrate", "overall"]):
            return QueryType.SYNTHESIS
        
        # Default to factual
        return QueryType.FACTUAL
    
    async def _classify_complexity(self, query: str, query_type: QueryType) -> QueryComplexity:
        """Classify query complexity."""
        
        complexity_score = 0
        
        # Length indicators
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Complexity indicators
        complex_patterns = [
            "multiple", "various", "several", "complex", "comprehensive",
            "detailed", "in-depth", "thoroughly", "extensively"
        ]
        complexity_score += sum(1 for pattern in complex_patterns if pattern in query)
        
        # Question type complexity
        multi_part_indicators = ["and", "also", "additionally", "furthermore", "moreover"]
        complexity_score += sum(1 for indicator in multi_part_indicators if indicator in query)
        
        # Technical terms (simplified heuristic)
        technical_indicators = ["system", "architecture", "implementation", "configuration"]
        complexity_score += sum(1 for indicator in technical_indicators if indicator in query)
        
        # Map to complexity levels
        if complexity_score >= 4:
            return QueryComplexity.EXPERT
        elif complexity_score >= 2:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 1:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    async def _classify_intent(self, query: str, query_type: QueryType) -> QueryIntent:
        """Classify user intent."""
        
        # Search intent
        if any(pattern in query for pattern in ["find", "search", "look for", "locate"]):
            return QueryIntent.SEARCH
        
        # Explanation intent
        if any(pattern in query for pattern in ["explain", "understand", "clarify", "meaning"]):
            return QueryIntent.EXPLANATION
        
        # Guidance intent
        if any(pattern in query for pattern in ["how", "guide", "instructions", "tutorial"]):
            return QueryIntent.GUIDANCE
        
        # Analysis intent
        if any(pattern in query for pattern in ["analyze", "examine", "investigate", "study"]):
            return QueryIntent.ANALYSIS
        
        # Comparison intent
        if any(pattern in query for pattern in ["compare", "contrast", "difference", "versus"]):
            return QueryIntent.COMPARISON
        
        # Troubleshooting intent
        if any(pattern in query for pattern in ["problem", "issue", "error", "fix", "solve"]):
            return QueryIntent.TROUBLESHOOTING
        
        # Planning intent
        if any(pattern in query for pattern in ["plan", "strategy", "approach", "design"]):
            return QueryIntent.PLANNING
        
        # Learning intent
        if any(pattern in query for pattern in ["learn", "teach", "understand", "master"]):
            return QueryIntent.LEARNING
        
        # Default based on query type
        if query_type == QueryType.PROCEDURAL:
            return QueryIntent.GUIDANCE
        elif query_type == QueryType.ANALYTICAL:
            return QueryIntent.ANALYSIS
        elif query_type == QueryType.COMPARATIVE:
            return QueryIntent.COMPARISON
        else:
            return QueryIntent.SEARCH
    
    async def _extract_elements(self, query: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract entities, concepts, and keywords from query.
        
        Args:
            query: Normalized query
            
        Returns:
            Tuple of (entities, concepts, keywords)
        """
        # Simplified extraction - in production, use NLP models
        words = query.split()
        
        # Extract potential entities (capitalized words)
        entities = []
        for word in words:
            clean_word = word.strip(".,!?()[]{}\"';:")
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append(clean_word)
        
        # Extract concepts (domain-specific terms)
        concept_indicators = [
            "system", "architecture", "framework", "model", "algorithm",
            "database", "network", "security", "performance", "optimization",
            "analysis", "design", "implementation", "configuration", "deployment"
        ]
        
        concepts = []
        query_lower = query.lower()
        for indicator in concept_indicators:
            if indicator in query_lower:
                concepts.append(indicator)
        
        # Extract keywords (important terms)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "must", "shall", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they"
        }
        
        keywords = []
        for word in words:
            clean_word = word.lower().strip(".,!?()[]{}\"';:")
            if (clean_word and 
                len(clean_word) > 2 and 
                clean_word not in stop_words and
                clean_word not in keywords):
                keywords.append(clean_word)
        
        return entities[:5], concepts[:5], keywords[:10]  # Limit results
    
    async def _extract_temporal_references(self, query: str) -> List[str]:
        """Extract temporal references from query."""
        temporal_patterns = [
            r'\b(yesterday|today|tomorrow)\b',
            r'\b(last|next|this)\s+(week|month|year|quarter)\b',
            r'\b(in|during|before|after)\s+\d{4}\b',
            r'\b(recently|soon|later|earlier)\b',
            r'\b(now|currently|presently)\b'
        ]
        
        temporal_refs = []
        for pattern in temporal_patterns:
            matches = re.findall(pattern, query.lower())
            temporal_refs.extend([match if isinstance(match, str) else ' '.join(match) for match in matches])
        
        return list(set(temporal_refs))
    
    async def _identify_domains(self, query: str, concepts: List[str]) -> List[str]:
        """Identify domain hints from query and concepts."""
        domain_mapping = {
            "technology": ["system", "software", "algorithm", "database", "network"],
            "business": ["strategy", "process", "workflow", "optimization", "analysis"],
            "security": ["security", "compliance", "audit", "risk", "vulnerability"],
            "data": ["data", "analytics", "model", "prediction", "insights"],
            "infrastructure": ["deployment", "configuration", "infrastructure", "architecture"]
        }
        
        query_lower = query.lower()
        domains = []
        
        for domain, indicators in domain_mapping.items():
            if any(indicator in query_lower or indicator in concepts for indicator in indicators):
                domains.append(domain)
        
        return domains
    
    async def _suggest_node_types(
        self,
        query_type: QueryType,
        entities: List[str],
        concepts: List[str]
    ) -> List[NodeType]:
        """Suggest relevant node types for retrieval."""
        suggested_types = []
        
        # Always include documents and chunks for comprehensive search
        suggested_types.extend([NodeType.DOCUMENT, NodeType.CHUNK])
        
        # Add entities if entities were extracted
        if entities:
            suggested_types.append(NodeType.ENTITY)
        
        # Add concepts if concepts were identified
        if concepts:
            suggested_types.append(NodeType.CONCEPT)
        
        # Add based on query type
        if query_type in [QueryType.PROCEDURAL, QueryType.ANALYTICAL]:
            suggested_types.extend([NodeType.PROCESS, NodeType.WORKFLOW])
        
        if query_type == QueryType.TEMPORAL:
            suggested_types.append(NodeType.EVENT)
        
        return list(set(suggested_types))  # Remove duplicates
    
    async def _identify_required_expertise(
        self,
        query_type: QueryType,
        complexity: QueryComplexity,
        domains: List[str]
    ) -> List[str]:
        """Identify required AI Twin expertise."""
        required_expertise = []
        
        # Based on complexity
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.EXPERT]:
            required_expertise.append("expert")
        
        # Based on query type
        if query_type == QueryType.ANALYTICAL:
            required_expertise.append("analyst")
        elif query_type == QueryType.PROCEDURAL:
            required_expertise.append("process")
        elif query_type == QueryType.COMPARATIVE:
            required_expertise.append("analyst")
        
        # Based on domains
        domain_expert_mapping = {
            "technology": "technical",
            "business": "business",
            "security": "security",
            "data": "data_scientist",
            "infrastructure": "devops"
        }
        
        for domain in domains:
            if domain in domain_expert_mapping:
                required_expertise.append(domain_expert_mapping[domain])
        
        return list(set(required_expertise))
    
    async def _validate_parsed_query(
        self,
        query: str,
        entities: List[str],
        concepts: List[str]
    ) -> None:
        """Validate parsed query components."""
        
        # Minimum length check
        if len(query.strip()) < 3:
            raise ValidationError("Query too short for meaningful processing")
        
        # Maximum length check
        if len(query) > 1000:
            raise ValidationError("Query too long for efficient processing")
        
        # Check for potentially harmful content (basic)
        harmful_patterns = ["<script", "javascript:", "eval(", "exec("]
        query_lower = query.lower()
        if any(pattern in query_lower for pattern in harmful_patterns):
            raise ValidationError("Query contains potentially harmful content")
    
    async def _calculate_complexity_score(
        self,
        query: str,
        entities: List[str],
        concepts: List[str],
        query_type: QueryType,
        complexity: QueryComplexity
    ) -> float:
        """Calculate numerical complexity score."""
        
        score = 0.0
        
        # Base score from complexity classification
        complexity_scores = {
            QueryComplexity.SIMPLE: 0.2,
            QueryComplexity.MODERATE: 0.4,
            QueryComplexity.COMPLEX: 0.7,
            QueryComplexity.EXPERT: 0.9
        }
        score += complexity_scores[complexity]
        
        # Adjust based on extracted elements
        score += min(len(entities) * 0.05, 0.2)  # Max 0.2 from entities
        score += min(len(concepts) * 0.05, 0.2)  # Max 0.2 from concepts
        
        # Adjust based on query type
        type_multipliers = {
            QueryType.FACTUAL: 0.8,
            QueryType.EXPLORATORY: 0.9,
            QueryType.PROCEDURAL: 1.0,
            QueryType.ANALYTICAL: 1.2,
            QueryType.COMPARATIVE: 1.1,
            QueryType.TEMPORAL: 1.0,
            QueryType.CAUSAL: 1.3,
            QueryType.SYNTHESIS: 1.4
        }
        score *= type_multipliers.get(query_type, 1.0)
        
        return min(score, 1.0)  # Cap at 1.0
    
    async def _calculate_classification_confidence(
        self,
        query: str,
        query_type: QueryType,
        complexity: QueryComplexity,
        intent: QueryIntent
    ) -> float:
        """Calculate confidence in classification."""
        
        # Simplified confidence calculation
        # In production, this would use ML models
        
        base_confidence = 0.7
        
        # Adjust based on query clarity
        if len(query.split()) < 5:
            base_confidence -= 0.1
        elif len(query.split()) > 15:
            base_confidence -= 0.05
        
        # Adjust based on keyword matches
        clear_indicators = {
            "what is": QueryType.FACTUAL,
            "how to": QueryType.PROCEDURAL,
            "compare": QueryType.COMPARATIVE,
            "analyze": QueryType.ANALYTICAL
        }
        
        for indicator, expected_type in clear_indicators.items():
            if indicator in query and query_type == expected_type:
                base_confidence += 0.2
                break
        
        return min(base_confidence, 1.0)
    
    def _update_processing_stats(self, metrics: QueryMetrics) -> None:
        """Update processing statistics."""
        self._processing_stats["total_queries"] += 1
        
        # Update average processing time
        current_avg = self._processing_stats["avg_processing_time"]
        total_queries = self._processing_stats["total_queries"]
        new_avg = ((current_avg * (total_queries - 1)) + metrics.processing_time_ms) / total_queries
        self._processing_stats["avg_processing_time"] = new_avg
    
    def _load_query_patterns(self) -> Dict[str, List[str]]:
        """Load query classification patterns."""
        # In production, load from configuration files
        return {
            "factual": ["what is", "define", "who is", "where is"],
            "procedural": ["how to", "steps to", "process for"],
            "analytical": ["analyze", "evaluate", "assess"],
            "comparative": ["compare", "difference", "versus"]
        }
    
    def _load_entity_patterns(self) -> Dict[str, str]:
        """Load entity recognition patterns."""
        # In production, load trained NER models
        return {}
    
    def _load_complexity_indicators(self) -> List[str]:
        """Load complexity indicator patterns."""
        return [
            "multiple", "various", "complex", "comprehensive",
            "detailed", "in-depth", "thoroughly"
        ]
    
    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get query processing statistics."""
        return {
            "total_queries_processed": self._processing_stats["total_queries"],
            "average_processing_time_ms": self._processing_stats["avg_processing_time"],
            "classification_accuracy": self._processing_stats["classification_accuracy"]
        }
    
    async def batch_process_queries(
        self,
        queries: List[str],
        batch_context: Optional[Dict[str, Any]] = None
    ) -> List[ParsedQuery]:
        """Process multiple queries in batch.
        
        Args:
            queries: List of queries to process
            batch_context: Shared context for batch
            
        Returns:
            List of parsed queries
        """
        results = []
        
        # Process queries with limited concurrency
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent processing
        
        async def process_single(query: str) -> ParsedQuery:
            async with semaphore:
                return await self.process_query(query, batch_context)
        
        # Execute batch processing
        tasks = [process_single(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log errors
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process query {i}: {str(result)}")
            else:
                successful_results.append(result)
        
        logger.info(
            "Batch query processing completed",
            total_queries=len(queries),
            successful=len(successful_results),
            failed=len(queries) - len(successful_results)
        )
        
        return successful_results