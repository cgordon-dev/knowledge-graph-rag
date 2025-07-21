# Agents API Reference

## Overview

The Knowledge Graph-RAG system provides a comprehensive agent framework with Google ADK integration, intelligent orchestration, and hybrid knowledge retrieval capabilities.

## Core Agents

### ADK Agent

Google ADK agent providing direct integration with Vertex AI and Generative AI models.

#### Configuration

```python
from kg_rag.agents import ADKAgent, ADKConfiguration

config = ADKConfiguration(
    project_id="your-gcp-project",
    location="us-central1", 
    model_name="gemini-1.5-pro",
    temperature=0.3,
    max_output_tokens=8192,
    hybrid_search_enabled=True,
    enable_content_filtering=True
)

agent = ADKAgent(
    agent_id="production-adk",
    config=config,
    twin_orchestrator=twin_orchestrator,
    vector_operations=vector_ops,
    query_builder=query_builder
)
```

#### Methods

##### `process_query(query, context=None, user_id=None, enable_twins=True, enable_retrieval=True)`

Process a query using Google ADK with knowledge graph integration.

**Parameters:**
- `query` (str): User query text
- `context` (Dict[str, Any], optional): Additional context information
- `user_id` (str, optional): User identifier for personalization
- `enable_twins` (bool): Whether to consult AI Digital Twins
- `enable_retrieval` (bool): Whether to perform knowledge retrieval

**Returns:** `ADKAgentResponse`

**Example:**
```python
response = await agent.process_query(
    query="Explain machine learning concepts",
    context={"domain": "technology", "level": "intermediate"},
    user_id="user123",
    enable_twins=True,
    enable_retrieval=True
)

print(f"Response: {response.response}")
print(f"Confidence: {response.confidence_score}")
print(f"Sources: {len(response.retrieved_documents)}")
```

##### `stream_response(query, context=None, user_id=None)`

Stream response generation for real-time interactions.

**Parameters:**
- `query` (str): User query text
- `context` (Dict[str, Any], optional): Additional context
- `user_id` (str, optional): User identifier

**Returns:** `AsyncGenerator[str, None]`

**Example:**
```python
async for chunk in agent.stream_response(
    query="Explain distributed systems architecture",
    context={"domain": "technology"}
):
    print(chunk, end="", flush=True)
```

##### `get_agent_status()`

Get agent status and statistics.

**Returns:** `Dict[str, Any]`

**Example:**
```python
status = await agent.get_agent_status()
print(f"Initialized: {status['initialized']}")
print(f"Cache size: {status['cache_size']}")
```

### RAG Agent

Comprehensive RAG agent combining Google ADK with advanced knowledge retrieval.

#### Configuration

```python
from kg_rag.agents import RAGAgent, RAGConfiguration, ADKConfiguration

adk_config = ADKConfiguration(
    project_id="your-project",
    model_name="gemini-1.5-pro"
)

rag_config = RAGConfiguration(
    agent_name="Production RAG Agent",
    adk_config=adk_config,
    retrieval_strategy="hybrid",
    max_retrieval_docs=10,
    min_similarity_threshold=0.7,
    enable_expert_consultation=True
)

agent = RAGAgent(
    agent_id="production-rag",
    config=rag_config,
    neo4j_driver=driver,
    twin_orchestrator=twin_orchestrator
)
```

#### Methods

##### `process_query(query)`

Process a comprehensive RAG query with multiple retrieval strategies.

**Parameters:**
- `query` (RAGQuery): Structured RAG query object

**Returns:** `RAGResponse`

**Example:**
```python
from kg_rag.agents import RAGQuery

rag_query = RAGQuery(
    query="Analyze microservices security architecture",
    context={"domain": "security", "urgency": "high"},
    user_id="security_engineer",
    required_confidence=0.8,
    include_reasoning=True,
    domain_filters=["security", "architecture"]
)

response = await agent.process_query(rag_query)

print(f"Response: {response.response}")
print(f"Confidence: {response.confidence_score}")
print(f"Knowledge Sources: {len(response.knowledge_sources)}")
print(f"Twin Insights: {len(response.twin_insights)}")
print(f"Reasoning: {response.reasoning_chain}")
```

##### `stream_response(query)`

Stream RAG response with retrieval status updates.

**Parameters:**
- `query` (RAGQuery): Structured RAG query

**Returns:** `AsyncGenerator[str, None]`

**Example:**
```python
async for chunk in agent.stream_response(rag_query):
    print(chunk, end="", flush=True)
```

##### `get_agent_metrics()`

Get comprehensive agent metrics and statistics.

**Returns:** `Dict[str, Any]`

## Query Processing

### Query Processor

Advanced query processing pipeline with intelligent classification.

#### Usage

```python
from kg_rag.agents import QueryProcessor

processor = QueryProcessor()

parsed_query = await processor.process_query(
    query="How to implement secure authentication in microservices?",
    user_context={"domain": "security", "experience": "intermediate"}
)

print(f"Query Type: {parsed_query.query_type}")
print(f"Complexity: {parsed_query.complexity}")
print(f"Intent: {parsed_query.intent}")
print(f"Entities: {parsed_query.entities}")
print(f"Concepts: {parsed_query.concepts}")
print(f"Required Expertise: {parsed_query.required_expertise}")
```

#### Query Types

- **FACTUAL**: Direct information retrieval
- **ANALYTICAL**: Analysis and reasoning
- **PROCEDURAL**: How-to and process queries
- **EXPLORATORY**: Open-ended exploration
- **COMPARATIVE**: Comparison between concepts
- **TEMPORAL**: Time-based queries
- **CAUSAL**: Cause-and-effect queries
- **SYNTHESIS**: Combining multiple sources

#### Complexity Levels

- **SIMPLE**: Single concept, direct answer
- **MODERATE**: Multiple concepts, some reasoning
- **COMPLEX**: Multi-step reasoning, synthesis
- **EXPERT**: Domain expertise required

#### Intent Categories

- **SEARCH**: Information seeking
- **EXPLANATION**: Understanding concepts
- **GUIDANCE**: Step-by-step instructions
- **ANALYSIS**: Deep analysis needed
- **COMPARISON**: Compare options
- **TROUBLESHOOTING**: Problem solving
- **PLANNING**: Strategic planning
- **LEARNING**: Educational content

## Agent Orchestration

### Agent Orchestrator

Intelligent multi-agent coordination with automatic routing.

#### Configuration

```python
from kg_rag.agents import AgentOrchestrator, OrchestrationConfiguration

config = OrchestrationConfiguration(
    routing_strategy="automatic",
    orchestration_mode="parallel",
    max_concurrent_agents=3,
    min_confidence_threshold=0.7,
    enable_validation=True,
    enable_fallback=True
)

orchestrator = AgentOrchestrator(
    orchestrator_id="production",
    config=config,
    adk_agent=adk_agent,
    rag_agent=rag_agent,
    kg_agent=kg_agent
)
```

#### Routing Strategies

- **AUTOMATIC**: ML-based routing using query analysis
- **BEST_MATCH**: Route to highest-scoring agent
- **LOAD_BALANCED**: Consider agent load and performance
- **COLLABORATIVE**: Multi-agent collaboration
- **ROUND_ROBIN**: Simple distribution

#### Orchestration Modes

- **SINGLE_AGENT**: One agent handles the query
- **SEQUENTIAL**: Agents process in sequence
- **PARALLEL**: Concurrent agent processing
- **HIERARCHICAL**: Structured agent coordination
- **CONSENSUS**: Multi-agent consensus building

#### Methods

##### `process_query(query, user_context=None, user_id=None, force_agent=None, force_mode=None)`

Process query through intelligent agent orchestration.

**Parameters:**
- `query` (str): User query
- `user_context` (Dict[str, Any], optional): Additional context
- `user_id` (str, optional): User identifier
- `force_agent` (AgentType, optional): Force specific agent
- `force_mode` (OrchestrationMode, optional): Force orchestration mode

**Returns:** `OrchestrationResult`

**Example:**
```python
result = await orchestrator.process_query(
    query="Provide comprehensive security analysis of microservices",
    user_context={"domain": "security", "urgency": "high"},
    user_id="security_analyst"
)

print(f"Primary Response: {result.primary_response}")
print(f"Agents Used: {result.agents_used}")
print(f"Confidence: {result.confidence_score}")
print(f"Routing Decision: {result.routing_decision}")
print(f"Processing Time: {result.processing_time_ms}ms")
```

##### `stream_response(query, user_context=None, user_id=None)`

Stream orchestrated response with routing information.

**Parameters:**
- `query` (str): User query
- `user_context` (Dict[str, Any], optional): Additional context
- `user_id` (str, optional): User identifier

**Returns:** `AsyncGenerator[str, None]`

**Example:**
```python
async for chunk in orchestrator.stream_response(
    query="Explain cloud security best practices",
    user_context={"domain": "security"}
):
    print(chunk, end="", flush=True)
```

##### `get_orchestration_stats()`

Get orchestration statistics and performance metrics.

**Returns:** `Dict[str, Any]`

**Example:**
```python
stats = await orchestrator.get_orchestration_stats()
print(f"Total Requests: {stats['total_requests']}")
print(f"Routing Decisions: {stats['routing_decisions']}")
print(f"Agent Performance: {stats['agent_performance']}")
```

## Data Models

### ADKAgentResponse

Response from Google ADK agent.

```python
class ADKAgentResponse(BaseModel):
    response_id: str
    agent_id: str
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    twin_consultations: List[Dict[str, Any]]
    confidence_score: float
    processing_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    timestamp: datetime
    model_used: str
    total_tokens: int
```

### RAGResponse

Comprehensive RAG response with sources and reasoning.

```python
class RAGResponse(BaseModel):
    query_id: str
    agent_id: str
    response: str
    confidence_score: float
    knowledge_sources: List[Dict[str, Any]]
    twin_insights: List[Dict[str, Any]]
    reasoning_chain: List[str]
    processing_metrics: Dict[str, float]
    model_info: Dict[str, Any]
    timestamp: datetime
    source_coverage: float
    factual_consistency: float
    relevance_score: float
```

### ParsedQuery

Parsed and enriched query structure.

```python
class ParsedQuery(BaseModel):
    query_id: str
    original_query: str
    normalized_query: str
    query_type: QueryType
    complexity: QueryComplexity
    intent: QueryIntent
    confidence: float
    entities: List[str]
    concepts: List[str]
    keywords: List[str]
    temporal_references: List[str]
    domain_hints: List[str]
    node_type_filters: List[NodeType]
    required_expertise: List[str]
    processing_timestamp: datetime
    processing_metrics: Optional[QueryMetrics]
```

### OrchestrationResult

Result from agent orchestration.

```python
class OrchestrationResult(BaseModel):
    orchestration_id: str
    query_id: str
    primary_response: str
    confidence_score: float
    agents_used: List[str]
    primary_agent: str
    routing_decision: str
    agent_responses: List[Dict[str, Any]]
    consensus_data: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    processing_time_ms: float
    orchestration_mode: OrchestrationMode
    timestamp: datetime
```

## Error Handling

### Exception Types

```python
from kg_rag.core.exceptions import (
    ADKAgentError,
    RAGAgentError,
    QueryProcessingError,
    OrchestrationError,
    ValidationError
)

try:
    response = await agent.process_query("test query")
except ADKAgentError as e:
    print(f"ADK Agent failed: {e}")
except QueryProcessingError as e:
    print(f"Query processing failed: {e}")
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### Error Recovery

```python
# Configure with fallback
config = OrchestrationConfiguration(
    enable_fallback=True,
    fallback_agent=AgentType.RAG
)

# Graceful degradation
try:
    result = await orchestrator.process_query(query)
except OrchestrationError:
    # Fallback to single agent
    result = await rag_agent.process_query(rag_query)
```

## Performance Optimization

### Caching

```python
# Enable response caching
adk_config = ADKConfiguration(
    enable_caching=True,
    cache_ttl_seconds=3600
)

orch_config = OrchestrationConfiguration(
    enable_caching=True
)
```

### Parallel Processing

```python
# Configure for high throughput
config = OrchestrationConfiguration(
    orchestration_mode="parallel",
    max_concurrent_agents=5,
    enable_parallel_processing=True
)
```

### Streaming

```python
# Use streaming for real-time responses
async for chunk in orchestrator.stream_response(query):
    # Process chunk immediately
    await process_chunk(chunk)
```

## Monitoring and Metrics

### Health Checks

```python
# Check agent health
adk_status = await adk_agent.get_agent_status()
rag_metrics = await rag_agent.get_agent_metrics()
orch_stats = await orchestrator.get_orchestration_stats()

health = {
    "adk_agent": adk_status["initialized"],
    "rag_agent": rag_metrics["initialized"],
    "orchestrator": orch_stats["total_requests"] > 0
}
```

### Performance Metrics

```python
# Collect performance data
processing_metrics = {
    "avg_response_time": stats["avg_processing_time"],
    "agent_performance": stats["agent_performance"],
    "routing_decisions": stats["routing_decisions"],
    "cache_hit_rate": cache_stats["hit_rate"]
}
```

## Best Practices

### Configuration

1. **Use environment variables** for sensitive configuration
2. **Enable caching** for production deployments
3. **Configure timeouts** appropriately for your use case
4. **Set confidence thresholds** based on quality requirements

### Error Handling

1. **Always handle exceptions** from agent operations
2. **Implement retry logic** for transient failures
3. **Use fallback strategies** for high availability
4. **Log errors with context** for debugging

### Performance

1. **Use streaming** for long-running queries
2. **Enable parallel processing** for multi-agent scenarios
3. **Monitor response times** and optimize as needed
4. **Implement circuit breakers** for external dependencies

### Security

1. **Validate all inputs** before processing
2. **Use secure authentication** for Google Cloud
3. **Enable content filtering** for safety
4. **Avoid logging sensitive information**