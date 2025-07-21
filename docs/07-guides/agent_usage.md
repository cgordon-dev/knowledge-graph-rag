# Agent Usage Examples

This document provides comprehensive examples for using the Google ADK agent integration in the Knowledge Graph-RAG system.

## Basic Setup

### Environment Configuration

```bash
# Set up Google Cloud credentials
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Configure Neo4j
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
```

### Initialize Components

```python
import asyncio
from neo4j import AsyncGraphDatabase
from kg_rag.agents import (
    ADKAgent, RAGAgent, AgentOrchestrator,
    ADKConfiguration, RAGConfiguration, OrchestrationConfiguration
)
from kg_rag.ai_twins.twin_orchestrator import TwinOrchestrator
from kg_rag.graph_schema import VectorGraphOperations, GraphQueryBuilder

async def setup_agents():
    # Initialize Neo4j driver
    driver = AsyncGraphDatabase.driver(
        "neo4j://localhost:7687",
        auth=("neo4j", "password")
    )
    
    # Initialize graph operations
    vector_ops = VectorGraphOperations(driver)
    query_builder = GraphQueryBuilder(driver)
    
    # Initialize AI Digital Twins
    twin_orchestrator = TwinOrchestrator()
    
    # Configure ADK Agent
    adk_config = ADKConfiguration(
        project_id="your-gcp-project",
        location="us-central1",
        model_name="gemini-1.5-pro",
        temperature=0.3,
        enable_content_filtering=True
    )
    
    adk_agent = ADKAgent(
        agent_id="production-adk",
        config=adk_config,
        twin_orchestrator=twin_orchestrator,
        vector_operations=vector_ops,
        query_builder=query_builder
    )
    
    # Configure RAG Agent
    rag_config = RAGConfiguration(
        agent_name="Production RAG Agent",
        adk_config=adk_config,
        retrieval_strategy="hybrid",
        max_retrieval_docs=10,
        enable_expert_consultation=True
    )
    
    rag_agent = RAGAgent(
        agent_id="production-rag",
        config=rag_config,
        neo4j_driver=driver,
        twin_orchestrator=twin_orchestrator
    )
    
    # Configure Orchestrator
    orch_config = OrchestrationConfiguration(
        routing_strategy="automatic",
        orchestration_mode="single_agent",
        enable_validation=True,
        min_confidence_threshold=0.7
    )
    
    orchestrator = AgentOrchestrator(
        orchestrator_id="production",
        config=orch_config,
        adk_agent=adk_agent,
        rag_agent=rag_agent
    )
    
    return orchestrator, adk_agent, rag_agent

# Initialize agents
orchestrator, adk_agent, rag_agent = await setup_agents()
```

## Basic Query Processing

### Simple Factual Query

```python
async def simple_query_example():
    """Example of processing a simple factual query."""
    
    result = await orchestrator.process_query(
        query="What is machine learning?",
        user_context={"domain": "technology", "level": "beginner"},
        user_id="student_001"
    )
    
    print(f"Query: What is machine learning?")
    print(f"Agent Used: {result.primary_agent}")
    print(f"Response: {result.primary_response}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print(f"Processing Time: {result.processing_time_ms:.0f}ms")
    
    return result

# Run example
result = await simple_query_example()
```

### Direct ADK Agent Usage

```python
async def direct_adk_example():
    """Example of using ADK agent directly."""
    
    response = await adk_agent.process_query(
        query="Explain the concept of distributed systems",
        context={
            "domain": "computer_science",
            "detail_level": "intermediate",
            "focus_areas": ["architecture", "scalability"]
        },
        user_id="engineer_001",
        enable_twins=True,
        enable_retrieval=True
    )
    
    print(f"ADK Agent Response:")
    print(f"Response: {response.response}")
    print(f"Confidence: {response.confidence_score:.3f}")
    print(f"Model Used: {response.model_used}")
    print(f"Total Tokens: {response.total_tokens}")
    print(f"Retrieved Documents: {len(response.retrieved_documents)}")
    print(f"Twin Consultations: {len(response.twin_consultations)}")
    
    # Access retrieved documents
    for i, doc in enumerate(response.retrieved_documents):
        print(f"Document {i+1}:")
        print(f"  Title: {doc['title']}")
        print(f"  Similarity: {doc['similarity_score']:.3f}")
        print(f"  Type: {doc['node_type']}")
    
    return response

# Run example
adk_response = await direct_adk_example()
```

## Advanced RAG Queries

### Complex Analysis Query

```python
from kg_rag.agents import RAGQuery
from kg_rag.graph_schema.node_models import NodeType

async def complex_rag_example():
    """Example of complex RAG query with detailed configuration."""
    
    # Create structured RAG query
    rag_query = RAGQuery(
        query="Analyze the security implications of implementing microservices architecture in a cloud environment, considering data protection, network security, and compliance requirements",
        context={
            "domain": "security",
            "urgency": "high",
            "environment": "cloud",
            "compliance_frameworks": ["SOC2", "GDPR", "ISO27001"]
        },
        user_id="security_architect",
        required_confidence=0.8,
        include_reasoning=True,
        node_type_filters=[NodeType.DOCUMENT, NodeType.CONCEPT, NodeType.PROCESS],
        domain_filters=["security", "cloud", "architecture"],
        preferred_sources=["security_guidelines", "compliance_docs"]
    )
    
    # Process through RAG agent
    response = await rag_agent.process_query(rag_query)
    
    print(f"Complex RAG Analysis:")
    print(f"Query ID: {response.query_id}")
    print(f"Response: {response.response}")
    print(f"Confidence: {response.confidence_score:.3f}")
    print(f"Source Coverage: {response.source_coverage:.3f}")
    print(f"Factual Consistency: {response.factual_consistency:.3f}")
    print(f"Relevance Score: {response.relevance_score:.3f}")
    
    # Display knowledge sources
    print(f"\nKnowledge Sources ({len(response.knowledge_sources)}):")
    for source in response.knowledge_sources:
        print(f"- {source['title']} (similarity: {source['similarity_score']:.3f})")
    
    # Display AI Twin insights
    print(f"\nAI Twin Insights ({len(response.twin_insights)}):")
    for insight in response.twin_insights:
        print(f"- {insight['twin_type']}: {insight['insight'][:100]}...")
        print(f"  Confidence: {insight['confidence']:.3f}")
    
    # Display reasoning chain
    print(f"\nReasoning Chain:")
    for i, step in enumerate(response.reasoning_chain, 1):
        print(f"{i}. {step}")
    
    return response

# Run example
complex_response = await complex_rag_example()
```

### Streaming RAG Response

```python
async def streaming_rag_example():
    """Example of streaming RAG response for real-time interaction."""
    
    rag_query = RAGQuery(
        query="Provide a comprehensive guide for implementing zero-trust security architecture",
        context={"domain": "security", "format": "implementation_guide"},
        user_id="security_engineer",
        include_reasoning=True
    )
    
    print("Streaming RAG Response:")
    print("=" * 50)
    
    # Stream response with real-time updates
    async for chunk in rag_agent.stream_response(rag_query):
        print(chunk, end="", flush=True)
    
    print("\n" + "=" * 50)
    print("Stream completed")

# Run example
await streaming_rag_example()
```

## Multi-Agent Orchestration

### Automatic Routing

```python
async def automatic_routing_example():
    """Example demonstrating automatic agent routing."""
    
    queries = [
        "What is the current status of Project Alpha?",  # Simple factual
        "How do I configure SSL certificates for nginx?",  # Procedural
        "Analyze the performance impact of database indexing strategies",  # Complex analytical
        "Compare OAuth 2.0 vs SAML for enterprise authentication"  # Comparative
    ]
    
    results = []
    
    for query in queries:
        result = await orchestrator.process_query(
            query=query,
            user_context={"domain": "technology"},
            user_id="tech_lead"
        )
        
        results.append({
            "query": query,
            "agent": result.primary_agent,
            "confidence": result.confidence_score,
            "processing_time": result.processing_time_ms,
            "routing_reason": result.routing_decision
        })
        
        print(f"Query: {query[:50]}...")
        print(f"Routed to: {result.primary_agent}")
        print(f"Confidence: {result.confidence_score:.3f}")
        print(f"Reason: {result.routing_decision}")
        print("-" * 40)
    
    return results

# Run example
routing_results = await automatic_routing_example()
```

### Collaborative Multi-Agent Processing

```python
async def collaborative_orchestration_example():
    """Example of collaborative multi-agent processing."""
    
    # Configure for collaboration
    collab_config = OrchestrationConfiguration(
        routing_strategy="collaborative",
        orchestration_mode="parallel",
        require_consensus=True,
        max_concurrent_agents=3,
        min_confidence_threshold=0.8
    )
    
    collab_orchestrator = AgentOrchestrator(
        orchestrator_id="collaborative",
        config=collab_config,
        adk_agent=adk_agent,
        rag_agent=rag_agent
    )
    
    # Complex query requiring multiple perspectives
    result = await collab_orchestrator.process_query(
        query="Design a comprehensive cloud migration strategy for a legacy monolithic application, considering security, performance, cost, and compliance requirements",
        user_context={
            "domain": "enterprise_architecture",
            "stakeholders": ["security", "operations", "finance"],
            "timeline": "6_months",
            "budget": "500k"
        },
        user_id="enterprise_architect"
    )
    
    print(f"Collaborative Orchestration Result:")
    print(f"Agents Used: {result.agents_used}")
    print(f"Primary Agent: {result.primary_agent}")
    print(f"Overall Confidence: {result.confidence_score:.3f}")
    print(f"Processing Time: {result.processing_time_ms:.0f}ms")
    
    # Display individual agent responses
    print(f"\nIndividual Agent Responses:")
    for response in result.agent_responses:
        print(f"- {response['agent_type']}: confidence {response['confidence']:.3f}")
        print(f"  {response['response'][:100]}...")
    
    # Display consensus analysis
    if result.consensus_data:
        print(f"\nConsensus Analysis:")
        print(f"Agreement Level: {result.consensus_data['agreement_level']:.3f}")
        print(f"Agent Count: {result.consensus_data['agent_count']}")
    
    return result

# Run example
collab_result = await collaborative_orchestration_example()
```

## Specialized Use Cases

### Security Analysis

```python
async def security_analysis_example():
    """Example of comprehensive security analysis."""
    
    # Force security-focused routing
    result = await orchestrator.process_query(
        query="Conduct a security audit of our API gateway configuration, identifying vulnerabilities and providing remediation steps",
        user_context={
            "domain": "security",
            "urgency": "critical",
            "system": "api_gateway",
            "environment": "production"
        },
        user_id="security_analyst",
        force_agent=AgentType.RAG  # Force RAG for comprehensive analysis
    )
    
    print(f"Security Analysis:")
    print(f"Primary Response: {result.primary_response}")
    
    # Check validation results
    if result.validation_results:
        print(f"\nValidation Results:")
        print(f"Valid: {result.validation_results['is_valid']}")
        print(f"Confidence: {result.validation_results['confidence']:.3f}")
        if result.validation_results['issues']:
            print(f"Issues: {result.validation_results['issues']}")
    
    return result

# Run example
security_result = await security_analysis_example()
```

### Technical Documentation Generation

```python
async def documentation_generation_example():
    """Example of generating technical documentation."""
    
    result = await orchestrator.process_query(
        query="Generate comprehensive API documentation for our user authentication endpoints, including examples, error codes, and security considerations",
        user_context={
            "domain": "documentation",
            "format": "api_docs",
            "target_audience": "developers",
            "include_examples": True
        },
        user_id="technical_writer"
    )
    
    print(f"Generated Documentation:")
    print(f"Length: {len(result.primary_response)} characters")
    print(f"Agent: {result.primary_agent}")
    print(f"Confidence: {result.confidence_score:.3f}")
    print(f"\nDocumentation Preview:")
    print(result.primary_response[:500] + "...")
    
    return result

# Run example
docs_result = await documentation_generation_example()
```

### Performance Optimization

```python
async def performance_optimization_example():
    """Example of performance-focused query processing."""
    
    # Configure for high performance
    perf_config = OrchestrationConfiguration(
        routing_strategy="load_balanced",
        orchestration_mode="single_agent",
        enable_caching=True,
        request_timeout_seconds=30
    )
    
    perf_orchestrator = AgentOrchestrator(
        orchestrator_id="performance",
        config=perf_config,
        adk_agent=adk_agent,
        rag_agent=rag_agent
    )
    
    # Batch process multiple queries
    queries = [
        "What is containerization?",
        "How does load balancing work?",
        "Explain database connection pooling",
        "What are the benefits of CDNs?",
        "How do you implement caching strategies?"
    ]
    
    start_time = asyncio.get_event_loop().time()
    
    # Process queries concurrently
    tasks = [
        perf_orchestrator.process_query(
            query=query,
            user_context={"domain": "performance"},
            user_id=f"user_{i}"
        )
        for i, query in enumerate(queries)
    ]
    
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    total_time = (end_time - start_time) * 1000
    
    print(f"Performance Test Results:")
    print(f"Queries Processed: {len(queries)}")
    print(f"Total Time: {total_time:.0f}ms")
    print(f"Average per Query: {total_time/len(queries):.0f}ms")
    
    for i, result in enumerate(results):
        print(f"Query {i+1}: {result.processing_time_ms:.0f}ms ({result.primary_agent})")
    
    return results

# Run example
perf_results = await performance_optimization_example()
```

## Streaming and Real-Time Interactions

### Real-Time Chat Interface

```python
async def chat_interface_example():
    """Example of implementing a real-time chat interface."""
    
    conversation_context = {
        "domain": "general",
        "conversation_id": "chat_001",
        "user_preferences": {
            "detail_level": "moderate",
            "technical_depth": "intermediate"
        }
    }
    
    print("Chat Interface Demo (type 'quit' to exit):")
    print("=" * 50)
    
    while True:
        # Get user input (in real implementation, this would come from UI)
        user_query = input("\nYou: ")
        
        if user_query.lower() == 'quit':
            break
        
        print("Assistant: ", end="", flush=True)
        
        # Stream response
        async for chunk in orchestrator.stream_response(
            query=user_query,
            user_context=conversation_context,
            user_id="chat_user"
        ):
            print(chunk, end="", flush=True)
        
        print()  # New line after response

# Run example (interactive)
# await chat_interface_example()
```

### Progressive Response Building

```python
async def progressive_response_example():
    """Example of building progressive responses with status updates."""
    
    query = "Provide a comprehensive analysis of cloud security best practices"
    
    print(f"Processing: {query}")
    print("=" * 60)
    
    # Simulate progressive response with status tracking
    status_messages = []
    response_chunks = []
    
    async for chunk in orchestrator.stream_response(
        query=query,
        user_context={"domain": "security"},
        user_id="security_engineer"
    ):
        # Check for status messages (would be formatted in real implementation)
        if chunk.startswith("🔍") or chunk.startswith("📚") or chunk.startswith("🤖"):
            status_messages.append(chunk.strip())
            print(f"Status: {chunk.strip()}")
        else:
            response_chunks.append(chunk)
            print(chunk, end="", flush=True)
    
    print("\n" + "=" * 60)
    print(f"Status Updates: {len(status_messages)}")
    print(f"Response Length: {len(''.join(response_chunks))} characters")
    
    return {
        "status_messages": status_messages,
        "response": "".join(response_chunks)
    }

# Run example
progressive_result = await progressive_response_example()
```

## Error Handling and Recovery

### Graceful Error Handling

```python
from kg_rag.core.exceptions import (
    ADKAgentError, RAGAgentError, OrchestrationError
)

async def error_handling_example():
    """Example of comprehensive error handling."""
    
    # Test queries that might cause different types of errors
    test_cases = [
        ("", "Empty query"),
        ("a" * 2000, "Very long query"),
        ("What is the meaning of life?", "Normal query"),
        ("<script>alert('xss')</script>", "Potentially harmful content")
    ]
    
    results = []
    
    for query, description in test_cases:
        print(f"Testing: {description}")
        
        try:
            result = await orchestrator.process_query(
                query=query,
                user_context={"domain": "test"},
                user_id="test_user"
            )
            
            print(f"✅ Success: {result.confidence_score:.3f} confidence")
            results.append({"status": "success", "confidence": result.confidence_score})
            
        except OrchestrationError as e:
            print(f"🔄 Orchestration Error: {str(e)}")
            
            # Try fallback to direct agent
            try:
                fallback_response = await adk_agent.process_query(
                    query=query if len(query) < 1000 and query.strip() else "Help with general information",
                    context={"fallback": True}
                )
                print(f"✅ Fallback Success: {fallback_response.confidence_score:.3f}")
                results.append({"status": "fallback_success", "confidence": fallback_response.confidence_score})
                
            except Exception as e2:
                print(f"❌ Fallback Failed: {str(e2)}")
                results.append({"status": "failed", "error": str(e2)})
                
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")
            results.append({"status": "unexpected_error", "error": str(e)})
        
        print("-" * 40)
    
    return results

# Run example
error_results = await error_handling_example()
```

### Circuit Breaker Pattern

```python
import time
from typing import Dict, Any

class AgentCircuitBreaker:
    """Simple circuit breaker for agent operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
    
    async def call_with_breaker(self, operation, *args, **kwargs):
        """Execute operation with circuit breaker protection."""
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await operation(*args, **kwargs)
            
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0
            
            return result
            
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

async def circuit_breaker_example():
    """Example using circuit breaker pattern."""
    
    breaker = AgentCircuitBreaker(failure_threshold=3, recovery_timeout=30)
    
    # Test circuit breaker with potential failures
    for i in range(10):
        try:
            result = await breaker.call_with_breaker(
                orchestrator.process_query,
                query=f"Test query {i}",
                user_context={"test": True},
                user_id="circuit_test"
            )
            
            print(f"Query {i}: Success")
            
        except Exception as e:
            print(f"Query {i}: Failed - {str(e)}")
            
        # Small delay between requests
        await asyncio.sleep(1)

# Run example
await circuit_breaker_example()
```

## Monitoring and Analytics

### Performance Monitoring

```python
async def monitoring_example():
    """Example of comprehensive monitoring and analytics."""
    
    # Collect baseline metrics
    adk_status = await adk_agent.get_agent_status()
    rag_metrics = await rag_agent.get_agent_metrics()
    orch_stats = await orchestrator.get_orchestration_stats()
    
    print("System Health Dashboard:")
    print("=" * 50)
    
    print(f"ADK Agent:")
    print(f"  Status: {'✅ Healthy' if adk_status['initialized'] else '❌ Down'}")
    print(f"  Model: {adk_status['model_name']}")
    print(f"  Cache Size: {adk_status['cache_size']}")
    
    print(f"\nRAG Agent:")
    print(f"  Status: {'✅ Healthy' if rag_metrics['initialized'] else '❌ Down'}")
    print(f"  Cached Queries: {rag_metrics['queries_cached']}")
    print(f"  Schema Stats: {rag_metrics['schema_stats']['total_nodes']} nodes")
    
    print(f"\nOrchestrator:")
    print(f"  Total Requests: {orch_stats['total_requests']}")
    print(f"  Routing Decisions: {orch_stats['routing_decisions']}")
    print(f"  Agent Performance: {len(orch_stats['agent_performance'])} agents tracked")
    
    # Test performance with sample queries
    print(f"\nPerformance Test:")
    
    test_queries = [
        "What is cloud computing?",
        "How do microservices communicate?",
        "Explain database sharding"
    ]
    
    performance_data = []
    
    for query in test_queries:
        start_time = time.time()
        
        result = await orchestrator.process_query(
            query=query,
            user_context={"domain": "technology"},
            user_id="perf_test"
        )
        
        end_time = time.time()
        response_time = (end_time - start_time) * 1000
        
        performance_data.append({
            "query": query,
            "response_time_ms": response_time,
            "confidence": result.confidence_score,
            "agent": result.primary_agent
        })
        
        print(f"  {query[:30]}... -> {response_time:.0f}ms ({result.primary_agent})")
    
    # Calculate averages
    avg_response_time = sum(p["response_time_ms"] for p in performance_data) / len(performance_data)
    avg_confidence = sum(p["confidence"] for p in performance_data) / len(performance_data)
    
    print(f"\nPerformance Summary:")
    print(f"  Average Response Time: {avg_response_time:.0f}ms")
    print(f"  Average Confidence: {avg_confidence:.3f}")
    
    return {
        "health": {
            "adk": adk_status,
            "rag": rag_metrics,
            "orchestrator": orch_stats
        },
        "performance": performance_data
    }

# Run example
monitoring_data = await monitoring_example()
```

## Integration Patterns

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Knowledge Graph RAG API")

class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    force_agent: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    confidence: float
    agent_used: str
    processing_time_ms: float
    sources: List[Dict[str, Any]]

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query through the agent orchestrator."""
    
    try:
        result = await orchestrator.process_query(
            query=request.query,
            user_context=request.context,
            user_id=request.user_id,
            force_agent=AgentType(request.force_agent) if request.force_agent else None
        )
        
        return QueryResponse(
            response=result.primary_response,
            confidence=result.confidence_score,
            agent_used=result.primary_agent,
            processing_time_ms=result.processing_time_ms,
            sources=result.agent_responses
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    try:
        adk_status = await adk_agent.get_agent_status()
        orch_stats = await orchestrator.get_orchestration_stats()
        
        return {
            "status": "healthy",
            "adk_initialized": adk_status["initialized"],
            "total_requests": orch_stats["total_requests"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

# Example usage:
# uvicorn main:app --host 0.0.0.0 --port 8000
```

This comprehensive set of examples demonstrates the full range of capabilities available in the Google ADK agent integration, from basic query processing to advanced multi-agent orchestration and production deployment patterns.