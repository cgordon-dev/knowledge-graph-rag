"""
Agent Orchestrator for Google ADK Integration.

Coordinates multiple agents (ADK, RAG, Knowledge Graph) to provide
comprehensive responses through intelligent routing and collaboration.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from enum import Enum
import uuid

import structlog
from pydantic import BaseModel, Field

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import OrchestrationError, AgentError
from kg_rag.agents.adk_agent import ADKAgent, ADKConfiguration, ADKAgentResponse
from kg_rag.agents.rag_agent import RAGAgent, RAGConfiguration, RAGQuery, RAGResponse
from kg_rag.agents.knowledge_graph_agent import KnowledgeGraphAgent
from kg_rag.agents.query_processor import (
    QueryProcessor, ParsedQuery, QueryType, QueryComplexity, QueryIntent
)
from kg_rag.ai_twins.twin_orchestrator import TwinOrchestrator
from kg_rag.graph_schema import VectorGraphOperations, GraphQueryBuilder

logger = structlog.get_logger(__name__)


class AgentType(str, Enum):
    """Available agent types."""
    
    ADK = "adk"                           # Google ADK agent
    RAG = "rag"                           # Comprehensive RAG agent
    KNOWLEDGE_GRAPH = "knowledge_graph"   # Knowledge graph specialist
    HYBRID = "hybrid"                     # Multi-agent collaboration


class RoutingStrategy(str, Enum):
    """Agent routing strategies."""
    
    AUTOMATIC = "automatic"       # Automatic routing based on query analysis
    ROUND_ROBIN = "round_robin"   # Round-robin distribution
    LOAD_BALANCED = "load_balanced"  # Load-based routing
    BEST_MATCH = "best_match"     # Route to best-matching agent
    COLLABORATIVE = "collaborative"  # Multi-agent collaboration


class OrchestrationMode(str, Enum):
    """Orchestration modes."""
    
    SINGLE_AGENT = "single_agent"         # Single agent handles query
    SEQUENTIAL = "sequential"             # Agents process sequentially
    PARALLEL = "parallel"                 # Agents process in parallel
    HIERARCHICAL = "hierarchical"         # Hierarchical agent coordination
    CONSENSUS = "consensus"               # Multi-agent consensus building


class OrchestrationConfiguration(BaseModel):
    """Configuration for agent orchestration."""
    
    # Routing Configuration
    routing_strategy: RoutingStrategy = Field(
        default=RoutingStrategy.AUTOMATIC,
        description="Agent routing strategy"
    )
    orchestration_mode: OrchestrationMode = Field(
        default=OrchestrationMode.SINGLE_AGENT,
        description="Orchestration mode"
    )
    
    # Performance Configuration
    max_concurrent_agents: int = Field(default=3, description="Maximum concurrent agents")
    request_timeout_seconds: int = Field(default=60, description="Request timeout")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    
    # Quality Configuration
    require_consensus: bool = Field(default=False, description="Require multi-agent consensus")
    min_confidence_threshold: float = Field(default=0.6, description="Minimum confidence threshold")
    enable_validation: bool = Field(default=True, description="Enable response validation")
    
    # Fallback Configuration
    enable_fallback: bool = Field(default=True, description="Enable agent fallback")
    fallback_agent: AgentType = Field(default=AgentType.RAG, description="Fallback agent type")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class OrchestrationResult(BaseModel):
    """Result from agent orchestration."""
    
    # Core Response
    orchestration_id: str = Field(..., description="Unique orchestration ID")
    query_id: str = Field(..., description="Original query ID")
    primary_response: str = Field(..., description="Primary response")
    confidence_score: float = Field(..., description="Overall confidence score")
    
    # Agent Information
    agents_used: List[str] = Field(default_factory=list, description="Agent IDs used")
    primary_agent: str = Field(..., description="Primary agent used")
    routing_decision: str = Field(..., description="Routing decision rationale")
    
    # Response Details
    agent_responses: List[Dict[str, Any]] = Field(default_factory=list, description="Individual agent responses")
    consensus_data: Optional[Dict[str, Any]] = Field(None, description="Consensus analysis")
    validation_results: Optional[Dict[str, Any]] = Field(None, description="Validation results")
    
    # Metadata
    processing_time_ms: float = Field(..., description="Total processing time")
    orchestration_mode: OrchestrationMode = Field(..., description="Mode used")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


class AgentOrchestrator:
    """Intelligent agent orchestrator for multi-agent coordination."""
    
    def __init__(
        self,
        orchestrator_id: str,
        config: OrchestrationConfiguration,
        adk_agent: Optional[ADKAgent] = None,
        rag_agent: Optional[RAGAgent] = None,
        kg_agent: Optional[KnowledgeGraphAgent] = None,
        twin_orchestrator: Optional[TwinOrchestrator] = None
    ):
        """Initialize agent orchestrator.
        
        Args:
            orchestrator_id: Unique orchestrator identifier
            config: Orchestration configuration
            adk_agent: Google ADK agent instance
            rag_agent: RAG agent instance
            kg_agent: Knowledge graph agent instance
            twin_orchestrator: AI Digital Twins orchestrator
        """
        self.orchestrator_id = orchestrator_id
        self.config = config
        self.twin_orchestrator = twin_orchestrator
        
        # Initialize agents
        self.agents = {}
        if adk_agent:
            self.agents[AgentType.ADK] = adk_agent
        if rag_agent:
            self.agents[AgentType.RAG] = rag_agent
        if kg_agent:
            self.agents[AgentType.KNOWLEDGE_GRAPH] = kg_agent
        
        # Initialize query processor
        self.query_processor = QueryProcessor()
        
        # Internal state
        self._routing_stats = {
            "total_requests": 0,
            "routing_decisions": {},
            "agent_performance": {},
            "fallback_activations": 0
        }
        self._response_cache = {}
        
        # Agent load tracking
        self._agent_loads = {agent_type: 0 for agent_type in self.agents.keys()}
        
        logger.info(
            "Agent orchestrator initialized",
            orchestrator_id=orchestrator_id,
            available_agents=list(self.agents.keys()),
            routing_strategy=config.routing_strategy.value
        )
    
    async def process_query(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        force_agent: Optional[AgentType] = None,
        force_mode: Optional[OrchestrationMode] = None
    ) -> OrchestrationResult:
        """Process query through intelligent agent orchestration.
        
        Args:
            query: User query
            user_context: Additional user context
            user_id: User identifier
            force_agent: Force specific agent (override routing)
            force_mode: Force specific orchestration mode
            
        Returns:
            Orchestration result with response and metadata
        """
        start_time = datetime.utcnow()
        orchestration_id = f"orch_{self.orchestrator_id}_{int(start_time.timestamp())}"
        
        logger.info(
            "Processing query through orchestration",
            orchestration_id=orchestration_id,
            query_length=len(query),
            has_context=bool(user_context),
            force_agent=force_agent.value if force_agent else None,
            force_mode=force_mode.value if force_mode else None
        )
        
        try:
            # Step 1: Parse and analyze query
            parsed_query = await self.query_processor.process_query(
                query, user_context
            )
            
            # Step 2: Determine routing strategy
            routing_decision = await self._determine_routing(
                parsed_query, force_agent, force_mode
            )
            
            # Step 3: Execute orchestration based on mode
            orchestration_mode = force_mode or self._determine_orchestration_mode(
                parsed_query, routing_decision
            )
            
            # Step 4: Process query through selected strategy
            agent_responses, primary_response, confidence = await self._execute_orchestration(
                parsed_query, routing_decision, orchestration_mode, user_id
            )
            
            # Step 5: Validate response if enabled
            validation_results = None
            if self.config.enable_validation:
                validation_results = await self._validate_response(
                    primary_response, agent_responses, parsed_query
                )
            
            # Step 6: Build consensus data if multiple agents used
            consensus_data = None
            if len(agent_responses) > 1:
                consensus_data = await self._build_consensus_analysis(agent_responses)
            
            # Step 7: Create orchestration result
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            result = OrchestrationResult(
                orchestration_id=orchestration_id,
                query_id=parsed_query.query_id,
                primary_response=primary_response,
                confidence_score=confidence,
                agents_used=[resp["agent_id"] for resp in agent_responses],
                primary_agent=routing_decision["primary_agent"],
                routing_decision=routing_decision["rationale"],
                agent_responses=agent_responses,
                consensus_data=consensus_data,
                validation_results=validation_results,
                processing_time_ms=total_time,
                orchestration_mode=orchestration_mode
            )
            
            # Step 8: Update statistics and cache
            await self._update_orchestration_stats(result, routing_decision)
            
            if self.config.enable_caching:
                await self._cache_response(query, result)
            
            logger.info(
                "Query orchestration completed successfully",
                orchestration_id=orchestration_id,
                query_id=parsed_query.query_id,
                primary_agent=routing_decision["primary_agent"],
                agents_used=len(agent_responses),
                confidence_score=confidence,
                processing_time_ms=total_time
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to orchestrate query processing: {str(e)}"
            logger.error(
                error_msg,
                orchestration_id=orchestration_id,
                query=query,
                error=str(e)
            )
            raise OrchestrationError(error_msg) from e
    
    async def _determine_routing(
        self,
        parsed_query: ParsedQuery,
        force_agent: Optional[AgentType] = None,
        force_mode: Optional[OrchestrationMode] = None
    ) -> Dict[str, Any]:
        """Determine agent routing based on query analysis.
        
        Args:
            parsed_query: Parsed query with metadata
            force_agent: Forced agent selection
            force_mode: Forced orchestration mode
            
        Returns:
            Routing decision with rationale
        """
        if force_agent:
            return {
                "primary_agent": force_agent.value,
                "secondary_agents": [],
                "rationale": f"Forced routing to {force_agent.value}",
                "confidence": 1.0
            }
        
        # Analyze query characteristics for routing
        routing_factors = await self._analyze_routing_factors(parsed_query)
        
        # Apply routing strategy
        if self.config.routing_strategy == RoutingStrategy.AUTOMATIC:
            return await self._automatic_routing(parsed_query, routing_factors)
        elif self.config.routing_strategy == RoutingStrategy.BEST_MATCH:
            return await self._best_match_routing(parsed_query, routing_factors)
        elif self.config.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._load_balanced_routing(parsed_query, routing_factors)
        elif self.config.routing_strategy == RoutingStrategy.COLLABORATIVE:
            return await self._collaborative_routing(parsed_query, routing_factors)
        else:  # ROUND_ROBIN
            return await self._round_robin_routing(parsed_query)
    
    async def _analyze_routing_factors(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Analyze factors for routing decisions."""
        
        factors = {
            "complexity_score": parsed_query.processing_metrics.complexity_score if parsed_query.processing_metrics else 0.5,
            "query_type": parsed_query.query_type,
            "intent": parsed_query.intent,
            "entities_count": len(parsed_query.entities),
            "concepts_count": len(parsed_query.concepts),
            "domain_hints": parsed_query.domain_hints,
            "required_expertise": parsed_query.required_expertise
        }
        
        # Agent capability scores
        factors["agent_scores"] = {
            AgentType.ADK: await self._score_agent_suitability(AgentType.ADK, parsed_query),
            AgentType.RAG: await self._score_agent_suitability(AgentType.RAG, parsed_query),
            AgentType.KNOWLEDGE_GRAPH: await self._score_agent_suitability(AgentType.KNOWLEDGE_GRAPH, parsed_query)
        }
        
        return factors
    
    async def _score_agent_suitability(self, agent_type: AgentType, parsed_query: ParsedQuery) -> float:
        """Score agent suitability for the parsed query."""
        
        score = 0.5  # Base score
        
        if agent_type == AgentType.ADK:
            # ADK excels at general queries with moderate complexity
            if parsed_query.complexity in ["simple", "moderate"]:
                score += 0.2
            if parsed_query.query_type in ["factual", "explanatory"]:
                score += 0.2
            if len(parsed_query.entities) <= 3:
                score += 0.1
        
        elif agent_type == AgentType.RAG:
            # RAG excels at complex queries requiring comprehensive analysis
            if parsed_query.complexity in ["complex", "expert"]:
                score += 0.3
            if parsed_query.query_type in ["analytical", "synthesis", "comparative"]:
                score += 0.2
            if len(parsed_query.concepts) >= 2:
                score += 0.1
            if parsed_query.required_expertise:
                score += 0.1
        
        elif agent_type == AgentType.KNOWLEDGE_GRAPH:
            # KG agent excels at entity-relationship queries
            if len(parsed_query.entities) >= 2:
                score += 0.3
            if parsed_query.query_type in ["exploratory", "temporal", "causal"]:
                score += 0.2
            if "relationship" in parsed_query.normalized_query.lower():
                score += 0.2
        
        # Check agent availability
        if agent_type in self.agents:
            score += 0.1
        else:
            score = 0.0  # Agent not available
        
        return min(score, 1.0)
    
    async def _automatic_routing(
        self,
        parsed_query: ParsedQuery,
        routing_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Automatic routing based on query analysis."""
        
        agent_scores = routing_factors["agent_scores"]
        
        # Find best primary agent
        available_agents = {k: v for k, v in agent_scores.items() if k in self.agents}
        
        if not available_agents:
            raise OrchestrationError("No agents available for routing")
        
        primary_agent = max(available_agents, key=available_agents.get)
        primary_score = available_agents[primary_agent]
        
        # Determine if collaboration is needed
        secondary_agents = []
        if (routing_factors["complexity_score"] > 0.7 or 
            len(routing_factors["required_expertise"]) > 1):
            
            # Add complementary agents
            for agent_type, score in available_agents.items():
                if agent_type != primary_agent and score > 0.6:
                    secondary_agents.append(agent_type.value)
        
        rationale = (
            f"Automatic routing selected {primary_agent.value} "
            f"(score: {primary_score:.3f}) based on query complexity "
            f"({routing_factors['complexity_score']:.3f}) and type "
            f"({parsed_query.query_type})"
        )
        
        if secondary_agents:
            rationale += f". Collaboration with: {', '.join(secondary_agents)}"
        
        return {
            "primary_agent": primary_agent.value,
            "secondary_agents": secondary_agents,
            "rationale": rationale,
            "confidence": primary_score
        }
    
    async def _best_match_routing(
        self,
        parsed_query: ParsedQuery,
        routing_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route to best-matching agent."""
        
        agent_scores = routing_factors["agent_scores"]
        available_agents = {k: v for k, v in agent_scores.items() if k in self.agents}
        
        if not available_agents:
            raise OrchestrationError("No agents available for routing")
        
        best_agent = max(available_agents, key=available_agents.get)
        best_score = available_agents[best_agent]
        
        return {
            "primary_agent": best_agent.value,
            "secondary_agents": [],
            "rationale": f"Best match routing to {best_agent.value} with score {best_score:.3f}",
            "confidence": best_score
        }
    
    async def _load_balanced_routing(
        self,
        parsed_query: ParsedQuery,
        routing_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route based on agent load balancing."""
        
        agent_scores = routing_factors["agent_scores"]
        available_agents = {k: v for k, v in agent_scores.items() if k in self.agents}
        
        # Adjust scores based on current load
        load_adjusted_scores = {}
        for agent_type, score in available_agents.items():
            load_factor = 1.0 - (self._agent_loads[agent_type] / 10.0)  # Normalize load
            load_adjusted_scores[agent_type] = score * max(load_factor, 0.1)
        
        best_agent = max(load_adjusted_scores, key=load_adjusted_scores.get)
        
        return {
            "primary_agent": best_agent.value,
            "secondary_agents": [],
            "rationale": f"Load-balanced routing to {best_agent.value}",
            "confidence": available_agents[best_agent]
        }
    
    async def _collaborative_routing(
        self,
        parsed_query: ParsedQuery,
        routing_factors: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route for collaborative multi-agent processing."""
        
        agent_scores = routing_factors["agent_scores"]
        available_agents = {k: v for k, v in agent_scores.items() if k in self.agents}
        
        # Select primary agent
        primary_agent = max(available_agents, key=available_agents.get)
        
        # Select secondary agents with good scores
        secondary_agents = []
        for agent_type, score in available_agents.items():
            if agent_type != primary_agent and score > 0.5:
                secondary_agents.append(agent_type.value)
        
        return {
            "primary_agent": primary_agent.value,
            "secondary_agents": secondary_agents,
            "rationale": f"Collaborative routing with {primary_agent.value} as primary",
            "confidence": available_agents[primary_agent]
        }
    
    async def _round_robin_routing(self, parsed_query: ParsedQuery) -> Dict[str, Any]:
        """Simple round-robin routing."""
        
        available_agents = list(self.agents.keys())
        if not available_agents:
            raise OrchestrationError("No agents available for routing")
        
        # Simple round-robin based on request count
        agent_index = self._routing_stats["total_requests"] % len(available_agents)
        selected_agent = available_agents[agent_index]
        
        return {
            "primary_agent": selected_agent.value,
            "secondary_agents": [],
            "rationale": f"Round-robin routing to {selected_agent.value}",
            "confidence": 0.5
        }
    
    def _determine_orchestration_mode(
        self,
        parsed_query: ParsedQuery,
        routing_decision: Dict[str, Any]
    ) -> OrchestrationMode:
        """Determine orchestration mode based on routing decision."""
        
        if len(routing_decision["secondary_agents"]) == 0:
            return OrchestrationMode.SINGLE_AGENT
        elif self.config.require_consensus:
            return OrchestrationMode.CONSENSUS
        elif parsed_query.processing_metrics and parsed_query.processing_metrics.complexity_score > 0.8:
            return OrchestrationMode.HIERARCHICAL
        else:
            return OrchestrationMode.PARALLEL
    
    async def _execute_orchestration(
        self,
        parsed_query: ParsedQuery,
        routing_decision: Dict[str, Any],
        mode: OrchestrationMode,
        user_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """Execute orchestration based on mode and routing."""
        
        primary_agent_type = AgentType(routing_decision["primary_agent"])
        secondary_agent_types = [AgentType(a) for a in routing_decision["secondary_agents"]]
        
        if mode == OrchestrationMode.SINGLE_AGENT:
            return await self._single_agent_execution(
                parsed_query, primary_agent_type, user_id
            )
        elif mode == OrchestrationMode.PARALLEL:
            return await self._parallel_execution(
                parsed_query, primary_agent_type, secondary_agent_types, user_id
            )
        elif mode == OrchestrationMode.SEQUENTIAL:
            return await self._sequential_execution(
                parsed_query, primary_agent_type, secondary_agent_types, user_id
            )
        elif mode == OrchestrationMode.HIERARCHICAL:
            return await self._hierarchical_execution(
                parsed_query, primary_agent_type, secondary_agent_types, user_id
            )
        elif mode == OrchestrationMode.CONSENSUS:
            return await self._consensus_execution(
                parsed_query, primary_agent_type, secondary_agent_types, user_id
            )
        else:
            raise OrchestrationError(f"Unsupported orchestration mode: {mode}")
    
    async def _single_agent_execution(
        self,
        parsed_query: ParsedQuery,
        agent_type: AgentType,
        user_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """Execute query with single agent."""
        
        agent = self.agents[agent_type]
        
        # Increment agent load
        self._agent_loads[agent_type] += 1
        
        try:
            if agent_type == AgentType.RAG:
                # Convert to RAG query
                rag_query = self._convert_to_rag_query(parsed_query, user_id)
                response = await agent.process_query(rag_query)
                
                agent_response = {
                    "agent_id": agent.agent_id,
                    "agent_type": agent_type.value,
                    "response": response.response,
                    "confidence": response.confidence_score,
                    "metadata": response.model_info
                }
                
                return [agent_response], response.response, response.confidence_score
                
            elif agent_type == AgentType.ADK:
                response = await agent.process_query(
                    query=parsed_query.normalized_query,
                    context={"parsed_query": parsed_query.dict()},
                    user_id=user_id
                )
                
                agent_response = {
                    "agent_id": agent.agent_id,
                    "agent_type": agent_type.value,
                    "response": response.response,
                    "confidence": response.confidence_score,
                    "metadata": {"model": response.model_used, "tokens": response.total_tokens}
                }
                
                return [agent_response], response.response, response.confidence_score
                
            else:  # Knowledge Graph Agent
                # For now, return a placeholder response
                # In full implementation, would have proper query processing
                agent_response = {
                    "agent_id": getattr(agent, 'agent_id', 'kg_agent'),
                    "agent_type": agent_type.value,
                    "response": f"Knowledge graph analysis for: {parsed_query.normalized_query}",
                    "confidence": 0.7,
                    "metadata": {}
                }
                
                return [agent_response], agent_response["response"], 0.7
        
        finally:
            # Decrement agent load
            self._agent_loads[agent_type] -= 1
    
    async def _parallel_execution(
        self,
        parsed_query: ParsedQuery,
        primary_agent_type: AgentType,
        secondary_agent_types: List[AgentType],
        user_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """Execute query with multiple agents in parallel."""
        
        all_agent_types = [primary_agent_type] + secondary_agent_types
        
        # Execute all agents in parallel
        tasks = []
        for agent_type in all_agent_types:
            task = self._single_agent_execution(parsed_query, agent_type, user_id)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {all_agent_types[i]} failed: {str(result)}")
            else:
                agent_responses, _, _ = result
                all_responses.extend(agent_responses)
        
        if not all_responses:
            raise OrchestrationError("All agents failed to process query")
        
        # Select primary response (from primary agent if available)
        primary_response = all_responses[0]["response"]
        primary_confidence = all_responses[0]["confidence"]
        
        for response in all_responses:
            if response["agent_type"] == primary_agent_type.value:
                primary_response = response["response"]
                primary_confidence = response["confidence"]
                break
        
        return all_responses, primary_response, primary_confidence
    
    async def _sequential_execution(
        self,
        parsed_query: ParsedQuery,
        primary_agent_type: AgentType,
        secondary_agent_types: List[AgentType],
        user_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """Execute query with agents sequentially."""
        
        all_responses = []
        
        # Execute primary agent first
        agent_responses, primary_response, primary_confidence = await self._single_agent_execution(
            parsed_query, primary_agent_type, user_id
        )
        all_responses.extend(agent_responses)
        
        # Execute secondary agents with enriched context
        enriched_context = {
            "primary_response": primary_response,
            "primary_confidence": primary_confidence
        }
        
        for agent_type in secondary_agent_types:
            try:
                agent_responses, _, _ = await self._single_agent_execution(
                    parsed_query, agent_type, user_id
                )
                all_responses.extend(agent_responses)
            except Exception as e:
                logger.error(f"Secondary agent {agent_type} failed: {str(e)}")
        
        return all_responses, primary_response, primary_confidence
    
    async def _hierarchical_execution(
        self,
        parsed_query: ParsedQuery,
        primary_agent_type: AgentType,
        secondary_agent_types: List[AgentType],
        user_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """Execute query with hierarchical agent coordination."""
        
        # For now, implement as sequential with improved coordination
        # In full implementation, would have sophisticated hierarchy
        return await self._sequential_execution(
            parsed_query, primary_agent_type, secondary_agent_types, user_id
        )
    
    async def _consensus_execution(
        self,
        parsed_query: ParsedQuery,
        primary_agent_type: AgentType,
        secondary_agent_types: List[AgentType],
        user_id: Optional[str]
    ) -> tuple[List[Dict[str, Any]], str, float]:
        """Execute query with consensus building."""
        
        # Execute all agents in parallel
        all_responses, _, _ = await self._parallel_execution(
            parsed_query, primary_agent_type, secondary_agent_types, user_id
        )
        
        # Build consensus response
        consensus_response = await self._build_consensus_response(all_responses)
        consensus_confidence = await self._calculate_consensus_confidence(all_responses)
        
        return all_responses, consensus_response, consensus_confidence
    
    def _convert_to_rag_query(self, parsed_query: ParsedQuery, user_id: Optional[str]) -> RAGQuery:
        """Convert parsed query to RAG query format."""
        
        return RAGQuery(
            query_id=parsed_query.query_id,
            query=parsed_query.normalized_query,
            user_id=user_id,
            context={
                "query_type": parsed_query.query_type,
                "complexity": parsed_query.complexity,
                "intent": parsed_query.intent,
                "entities": parsed_query.entities,
                "concepts": parsed_query.concepts
            },
            required_confidence=self.config.min_confidence_threshold,
            include_reasoning=True,
            node_type_filters=parsed_query.node_type_filters,
            domain_filters=parsed_query.domain_hints
        )
    
    async def _validate_response(
        self,
        response: str,
        agent_responses: List[Dict[str, Any]],
        parsed_query: ParsedQuery
    ) -> Dict[str, Any]:
        """Validate orchestrated response."""
        
        validation_results = {
            "is_valid": True,
            "confidence": 1.0,
            "issues": [],
            "recommendations": []
        }
        
        # Basic validation checks
        if len(response.strip()) < 10:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Response too short")
        
        if len(agent_responses) == 0:
            validation_results["is_valid"] = False
            validation_results["issues"].append("No agent responses available")
        
        # Check confidence threshold
        avg_confidence = sum([ar["confidence"] for ar in agent_responses]) / len(agent_responses)
        if avg_confidence < self.config.min_confidence_threshold:
            validation_results["issues"].append(f"Low confidence: {avg_confidence:.3f}")
            validation_results["confidence"] = avg_confidence
        
        return validation_results
    
    async def _build_consensus_analysis(self, agent_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build consensus analysis from multiple agent responses."""
        
        if len(agent_responses) <= 1:
            return {"type": "single_agent", "agreement_level": 1.0}
        
        # Calculate agreement metrics
        confidences = [ar["confidence"] for ar in agent_responses]
        avg_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum([(c - avg_confidence) ** 2 for c in confidences]) / len(confidences)
        
        # Simple agreement based on confidence similarity
        agreement_level = max(0.0, 1.0 - confidence_variance)
        
        return {
            "type": "multi_agent",
            "agent_count": len(agent_responses),
            "agreement_level": agreement_level,
            "avg_confidence": avg_confidence,
            "confidence_variance": confidence_variance,
            "participating_agents": [ar["agent_type"] for ar in agent_responses]
        }
    
    async def _build_consensus_response(self, agent_responses: List[Dict[str, Any]]) -> str:
        """Build consensus response from multiple agent responses."""
        
        if len(agent_responses) == 1:
            return agent_responses[0]["response"]
        
        # For now, return the highest confidence response
        # In full implementation, would do sophisticated consensus building
        best_response = max(agent_responses, key=lambda x: x["confidence"])
        
        return best_response["response"]
    
    async def _calculate_consensus_confidence(self, agent_responses: List[Dict[str, Any]]) -> float:
        """Calculate consensus confidence score."""
        
        if len(agent_responses) == 1:
            return agent_responses[0]["confidence"]
        
        # Average confidence with bonus for agreement
        avg_confidence = sum([ar["confidence"] for ar in agent_responses]) / len(agent_responses)
        
        # Agreement bonus (simplified)
        agreement_bonus = 0.1 if len(agent_responses) > 1 else 0.0
        
        return min(avg_confidence + agreement_bonus, 1.0)
    
    async def _update_orchestration_stats(
        self,
        result: OrchestrationResult,
        routing_decision: Dict[str, Any]
    ) -> None:
        """Update orchestration statistics."""
        
        self._routing_stats["total_requests"] += 1
        
        # Update routing decision stats
        primary_agent = routing_decision["primary_agent"]
        if primary_agent not in self._routing_stats["routing_decisions"]:
            self._routing_stats["routing_decisions"][primary_agent] = 0
        self._routing_stats["routing_decisions"][primary_agent] += 1
        
        # Update agent performance stats
        for agent_id in result.agents_used:
            if agent_id not in self._routing_stats["agent_performance"]:
                self._routing_stats["agent_performance"][agent_id] = {
                    "requests": 0,
                    "avg_confidence": 0.0,
                    "avg_time": 0.0
                }
            
            stats = self._routing_stats["agent_performance"][agent_id]
            stats["requests"] += 1
            
            # Update running averages
            new_confidence = result.confidence_score
            new_time = result.processing_time_ms
            
            stats["avg_confidence"] = (
                (stats["avg_confidence"] * (stats["requests"] - 1) + new_confidence) / 
                stats["requests"]
            )
            stats["avg_time"] = (
                (stats["avg_time"] * (stats["requests"] - 1) + new_time) / 
                stats["requests"]
            )
    
    async def _cache_response(self, query: str, result: OrchestrationResult) -> None:
        """Cache orchestration result."""
        
        cache_key = f"orch_{hash(query)}_{self.orchestrator_id}"
        self._response_cache[cache_key] = {
            "result": result.dict(),
            "timestamp": datetime.utcnow(),
            "ttl": 3600  # 1 hour TTL
        }
    
    async def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics."""
        
        return {
            "orchestrator_id": self.orchestrator_id,
            "total_requests": self._routing_stats["total_requests"],
            "routing_decisions": self._routing_stats["routing_decisions"],
            "agent_performance": self._routing_stats["agent_performance"],
            "fallback_activations": self._routing_stats["fallback_activations"],
            "available_agents": list(self.agents.keys()),
            "current_loads": self._agent_loads,
            "cache_size": len(self._response_cache)
        }
    
    async def stream_response(
        self,
        query: str,
        user_context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream orchestrated response for real-time interactions."""
        
        try:
            # Parse query
            parsed_query = await self.query_processor.process_query(query, user_context)
            
            # Determine routing
            routing_decision = await self._determine_routing(parsed_query)
            primary_agent_type = AgentType(routing_decision["primary_agent"])
            
            yield f"🎯 Routing to {primary_agent_type.value} agent...\n\n"
            
            # Stream from primary agent
            agent = self.agents[primary_agent_type]
            
            if hasattr(agent, 'stream_response'):
                async for chunk in agent.stream_response(
                    parsed_query.normalized_query, user_context, user_id
                ):
                    yield chunk
            else:
                # Fallback to regular processing
                result = await self.process_query(query, user_context, user_id)
                yield result.primary_response
                
        except Exception as e:
            logger.error(f"Streaming orchestration failed: {str(e)}")
            yield f"\n\n❌ Error: {str(e)}"