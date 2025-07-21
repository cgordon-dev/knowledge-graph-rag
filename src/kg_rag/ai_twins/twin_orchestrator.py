"""
Twin Orchestrator for managing and coordinating AI Digital Twins.

Provides centralized management, routing, and coordination for all digital twins
with intelligent query routing and collaborative twin interactions.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import structlog
from pydantic import BaseModel, Field

from kg_rag.ai_twins.base_twin import BaseTwin
from kg_rag.ai_twins.expert_twin import ExpertTwin, ExpertDomain
from kg_rag.ai_twins.user_journey_twin import UserJourneyTwin, UserPersona, UserJourneyStep
from kg_rag.ai_twins.process_automation_twin import ProcessAutomationTwin, ProcessStep, ProcessMetrics
from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import PersonaTwinError, PersonaValidationError
from kg_rag.core.logger import get_performance_logger
from kg_rag.mcp_servers.orchestrator import get_orchestrator


class TwinRegistration(BaseModel):
    """Registration information for a digital twin."""
    
    twin_id: str = Field(..., description="Unique twin identifier")
    twin_type: str = Field(..., description="Type of twin")
    twin_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Twin description")
    capabilities: List[str] = Field(default_factory=list, description="Twin capabilities")
    domains: List[str] = Field(default_factory=list, description="Domain expertise")
    registered_at: datetime = Field(default_factory=datetime.utcnow, description="Registration timestamp")
    is_active: bool = Field(default=True, description="Whether twin is active")
    priority: int = Field(default=5, description="Priority level (1-10)")


class TwinInteractionResult(BaseModel):
    """Result of a twin interaction."""
    
    twin_id: str = Field(..., description="Twin that processed the query")
    interaction_id: str = Field(..., description="Interaction identifier")
    response: str = Field(..., description="Twin response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class CollaborativeResult(BaseModel):
    """Result of collaborative twin processing."""
    
    query: str = Field(..., description="Original query")
    primary_twin: str = Field(..., description="Primary twin ID")
    contributing_twins: List[str] = Field(default_factory=list, description="Contributing twin IDs")
    synthesized_response: str = Field(..., description="Synthesized response")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    collaboration_metadata: Dict[str, Any] = Field(default_factory=dict, description="Collaboration details")


class QueryRoutingRule(BaseModel):
    """Rule for routing queries to appropriate twins."""
    
    rule_id: str = Field(..., description="Rule identifier")
    keywords: List[str] = Field(default_factory=list, description="Trigger keywords")
    twin_types: List[str] = Field(default_factory=list, description="Target twin types")
    domains: List[str] = Field(default_factory=list, description="Target domains")
    priority: int = Field(default=5, description="Rule priority")
    is_active: bool = Field(default=True, description="Whether rule is active")


class TwinOrchestrator:
    """
    Orchestrates multiple AI Digital Twins for collaborative intelligence.
    
    Manages twin lifecycle, routes queries intelligently, enables collaboration,
    and provides unified interface for twin interactions.
    """
    
    def __init__(self):
        """Initialize Twin Orchestrator."""
        self.settings = get_settings()
        self.logger = structlog.get_logger("twin_orchestrator")
        self.performance_logger = get_performance_logger()
        
        # Twin management
        self.twins: Dict[str, BaseTwin] = {}
        self.twin_registrations: Dict[str, TwinRegistration] = {}
        
        # Routing and coordination
        self.routing_rules: List[QueryRoutingRule] = []
        self.interaction_history: List[TwinInteractionResult] = []
        self.collaboration_cache: Dict[str, Any] = {}
        
        # Performance tracking
        self.total_interactions = 0
        self.successful_interactions = 0
        self.average_response_time = 0.0
        self.twin_performance: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default routing rules
        self._initialize_default_routing_rules()
        
        self.logger.info("Twin orchestrator initialized")
    
    async def register_twin(
        self,
        twin: BaseTwin,
        capabilities: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
        priority: int = 5
    ) -> TwinRegistration:
        """
        Register a digital twin with the orchestrator.
        
        Args:
            twin: Digital twin instance
            capabilities: List of twin capabilities
            domains: List of domain expertise
            priority: Twin priority level (1-10)
            
        Returns:
            Twin registration information
        """
        try:
            if twin.twin_id in self.twins:
                raise PersonaValidationError(
                    f"Twin '{twin.twin_id}' already registered",
                    persona_id=twin.twin_id,
                    validation_errors={"duplicate_registration": True}
                )
            
            # Create registration
            registration = TwinRegistration(
                twin_id=twin.twin_id,
                twin_type=twin.twin_type,
                twin_name=twin.name,
                description=twin.description,
                capabilities=capabilities or self._infer_twin_capabilities(twin),
                domains=domains or self._infer_twin_domains(twin),
                priority=priority
            )
            
            # Register twin
            self.twins[twin.twin_id] = twin
            self.twin_registrations[twin.twin_id] = registration
            
            # Initialize performance tracking
            self.twin_performance[twin.twin_id] = {
                "interaction_count": 0,
                "average_confidence": 0.0,
                "average_response_time": 0.0,
                "success_rate": 1.0,
                "last_interaction": None
            }
            
            self.logger.info(
                "Twin registered successfully",
                twin_id=twin.twin_id,
                twin_type=twin.twin_type,
                capabilities=len(registration.capabilities),
                domains=len(registration.domains)
            )
            
            return registration
            
        except Exception as e:
            self.logger.error(
                "Twin registration failed",
                twin_id=getattr(twin, 'twin_id', 'unknown'),
                error=str(e)
            )
            raise PersonaTwinError(f"Twin registration failed: {e}", getattr(twin, 'twin_id', 'unknown'))
    
    async def unregister_twin(self, twin_id: str) -> bool:
        """
        Unregister a digital twin.
        
        Args:
            twin_id: Twin identifier
            
        Returns:
            Success status
        """
        try:
            if twin_id not in self.twins:
                self.logger.warning(f"Twin '{twin_id}' not found for unregistration")
                return False
            
            # Remove twin and registration
            del self.twins[twin_id]
            del self.twin_registrations[twin_id]
            
            # Clean up performance tracking
            if twin_id in self.twin_performance:
                del self.twin_performance[twin_id]
            
            self.logger.info(f"Twin unregistered successfully", twin_id=twin_id)
            return True
            
        except Exception as e:
            self.logger.error(f"Twin unregistration failed", twin_id=twin_id, error=str(e))
            return False
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        preferred_twin_type: Optional[str] = None,
        enable_collaboration: bool = True,
        user_id: Optional[str] = None
    ) -> Union[TwinInteractionResult, CollaborativeResult]:
        """
        Process a query through the appropriate digital twin(s).
        
        Args:
            query: User query
            context: Additional context
            preferred_twin_type: Preferred twin type for processing
            enable_collaboration: Whether to enable multi-twin collaboration
            user_id: User identifier
            
        Returns:
            Processing result from single twin or collaborative processing
        """
        start_time = datetime.utcnow()
        self.total_interactions += 1
        
        try:
            # Route query to appropriate twin(s)
            if preferred_twin_type:
                target_twins = self._get_twins_by_type(preferred_twin_type)
            else:
                target_twins = await self._route_query(query, context)
            
            if not target_twins:
                raise PersonaTwinError("No suitable twins found for query", query[:50])
            
            # Determine processing strategy
            if len(target_twins) == 1 or not enable_collaboration:
                # Single twin processing
                twin = target_twins[0]
                result = await self._process_single_twin(twin, query, context, user_id)
                
            else:
                # Collaborative processing
                result = await self._process_collaborative(target_twins, query, context, user_id)
            
            # Update performance metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_performance_metrics(result, processing_time)
            
            self.successful_interactions += 1
            
            # Log performance
            self.performance_logger.log_query_performance(
                query_type="twin_orchestrator_query",
                duration_ms=processing_time,
                result_count=1,
                user_id=user_id,
                additional_metrics={
                    "twins_involved": len(target_twins),
                    "collaboration_enabled": enable_collaboration,
                    "query_length": len(query)
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Query processing failed",
                query_preview=query[:100],
                error=str(e)
            )
            raise PersonaTwinError(f"Query processing failed: {e}", query[:50])
    
    async def get_twin_recommendations(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_recommendations: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get twin recommendations for a query without executing.
        
        Args:
            query: User query
            context: Additional context
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of twin recommendations with scores
        """
        try:
            recommendations = []
            
            # Score all active twins for the query
            for twin_id, twin in self.twins.items():
                if not self.twin_registrations[twin_id].is_active:
                    continue
                
                score = await self._calculate_twin_relevance_score(twin, query, context)
                
                if score > 0.1:  # Minimum relevance threshold
                    recommendations.append({
                        "twin_id": twin_id,
                        "twin_name": twin.name,
                        "twin_type": twin.twin_type,
                        "relevance_score": score,
                        "capabilities": self.twin_registrations[twin_id].capabilities,
                        "domains": self.twin_registrations[twin_id].domains,
                        "performance": self.twin_performance.get(twin_id, {}),
                        "confidence_estimate": self._estimate_response_confidence(twin, query)
                    })
            
            # Sort by relevance score and limit results
            recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
            return recommendations[:max_recommendations]
            
        except Exception as e:
            self.logger.error("Twin recommendation generation failed", error=str(e))
            return []
    
    async def enable_twin_collaboration(
        self,
        primary_twin_id: str,
        supporting_twin_ids: List[str],
        collaboration_strategy: str = "consensus"
    ) -> Dict[str, Any]:
        """
        Enable collaboration between specific twins.
        
        Args:
            primary_twin_id: Primary twin identifier
            supporting_twin_ids: Supporting twin identifiers
            collaboration_strategy: Collaboration strategy (consensus, expert_review, synthesis)
            
        Returns:
            Collaboration configuration
        """
        try:
            # Validate twins exist and are active
            all_twin_ids = [primary_twin_id] + supporting_twin_ids
            for twin_id in all_twin_ids:
                if twin_id not in self.twins:
                    raise PersonaValidationError(
                        f"Twin '{twin_id}' not found",
                        persona_id=twin_id,
                        validation_errors={"twin_not_found": True}
                    )
                
                if not self.twin_registrations[twin_id].is_active:
                    raise PersonaValidationError(
                        f"Twin '{twin_id}' is not active",
                        persona_id=twin_id,
                        validation_errors={"twin_inactive": True}
                    )
            
            # Create collaboration configuration
            collaboration_config = {
                "collaboration_id": f"collab_{int(datetime.utcnow().timestamp())}",
                "primary_twin": primary_twin_id,
                "supporting_twins": supporting_twin_ids,
                "strategy": collaboration_strategy,
                "created_at": datetime.utcnow().isoformat(),
                "is_active": True
            }
            
            # Store in collaboration cache
            cache_key = f"{primary_twin_id}:{'_'.join(sorted(supporting_twin_ids))}"
            self.collaboration_cache[cache_key] = collaboration_config
            
            self.logger.info(
                "Twin collaboration enabled",
                primary_twin=primary_twin_id,
                supporting_twins=len(supporting_twin_ids),
                strategy=collaboration_strategy
            )
            
            return collaboration_config
            
        except Exception as e:
            self.logger.error(
                "Collaboration setup failed",
                primary_twin=primary_twin_id,
                error=str(e)
            )
            raise PersonaTwinError(f"Collaboration setup failed: {e}", primary_twin_id)
    
    async def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator statistics."""
        active_twins = sum(1 for reg in self.twin_registrations.values() if reg.is_active)
        
        # Calculate performance metrics
        success_rate = (self.successful_interactions / max(self.total_interactions, 1)) * 100
        avg_confidence = np.mean([
            perf.get("average_confidence", 0) for perf in self.twin_performance.values()
        ]) if self.twin_performance else 0
        
        # Twin type distribution
        twin_types = {}
        for registration in self.twin_registrations.values():
            twin_type = registration.twin_type
            twin_types[twin_type] = twin_types.get(twin_type, 0) + 1
        
        # Recent interaction summary
        recent_interactions = [
            result for result in self.interaction_history
            if datetime.fromisoformat(result.metadata.get("timestamp", datetime.utcnow().isoformat())) 
            > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "orchestrator_status": "active",
            "total_twins": len(self.twins),
            "active_twins": active_twins,
            "twin_types": twin_types,
            "performance_metrics": {
                "total_interactions": self.total_interactions,
                "successful_interactions": self.successful_interactions,
                "success_rate": success_rate,
                "average_confidence": avg_confidence,
                "average_response_time_ms": self.average_response_time
            },
            "routing_rules": len(self.routing_rules),
            "collaboration_configs": len(self.collaboration_cache),
            "recent_activity": {
                "interactions_24h": len(recent_interactions),
                "active_twins_24h": len(set(r.twin_id for r in recent_interactions))
            },
            "twin_performance": self.twin_performance.copy()
        }
    
    def get_twin_by_id(self, twin_id: str) -> Optional[BaseTwin]:
        """Get a twin by its identifier."""
        return self.twins.get(twin_id)
    
    def get_twins_by_type(self, twin_type: str) -> List[BaseTwin]:
        """Get all twins of a specific type."""
        return [
            twin for twin in self.twins.values()
            if twin.twin_type == twin_type and 
            self.twin_registrations[twin.twin_id].is_active
        ]
    
    def get_twins_by_domain(self, domain: str) -> List[BaseTwin]:
        """Get all twins with expertise in a specific domain."""
        matching_twins = []
        for twin_id, twin in self.twins.items():
            registration = self.twin_registrations[twin_id]
            if registration.is_active and domain.lower() in [d.lower() for d in registration.domains]:
                matching_twins.append(twin)
        return matching_twins
    
    async def validate_all_twins(self) -> Dict[str, Any]:
        """Validate all registered twins."""
        validation_results = {}
        
        for twin_id, twin in self.twins.items():
            try:
                validation_result = await twin.validate_consistency()
                validation_results[twin_id] = {
                    "status": "valid" if validation_result["consistency_score"] > 0.8 else "issues",
                    "consistency_score": validation_result["consistency_score"],
                    "issues": validation_result.get("issues", []),
                    "last_validated": datetime.utcnow().isoformat()
                }
            except Exception as e:
                validation_results[twin_id] = {
                    "status": "error",
                    "error": str(e),
                    "last_validated": datetime.utcnow().isoformat()
                }
        
        return {
            "validation_summary": {
                "total_twins": len(validation_results),
                "valid_twins": sum(1 for r in validation_results.values() if r["status"] == "valid"),
                "twins_with_issues": sum(1 for r in validation_results.values() if r["status"] == "issues"),
                "error_twins": sum(1 for r in validation_results.values() if r["status"] == "error")
            },
            "twin_validations": validation_results
        }
    
    def _initialize_default_routing_rules(self) -> None:
        """Initialize default routing rules."""
        default_rules = [
            QueryRoutingRule(
                rule_id="expert_consultation",
                keywords=["expert", "validate", "review", "assess", "professional"],
                twin_types=["expert"],
                priority=8
            ),
            QueryRoutingRule(
                rule_id="user_journey_analysis",
                keywords=["journey", "user", "experience", "persona", "pain point"],
                twin_types=["user_journey"],
                priority=7
            ),
            QueryRoutingRule(
                rule_id="process_automation",
                keywords=["process", "automate", "workflow", "efficiency", "bottleneck"],
                twin_types=["process_automation"],
                priority=7
            ),
            QueryRoutingRule(
                rule_id="compliance_security",
                keywords=["compliance", "security", "control", "audit", "policy"],
                twin_types=["expert"],
                domains=["compliance", "security"],
                priority=9
            )
        ]
        
        self.routing_rules.extend(default_rules)
    
    async def _route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[BaseTwin]:
        """Route query to appropriate twins based on routing rules."""
        query_lower = query.lower()
        twin_scores = {}
        
        # Apply routing rules
        for rule in self.routing_rules:
            if not rule.is_active:
                continue
            
            # Check keyword matches
            keyword_matches = sum(1 for keyword in rule.keywords if keyword.lower() in query_lower)
            if keyword_matches == 0:
                continue
            
            # Score twins matching this rule
            for twin_id, twin in self.twins.items():
                if not self.twin_registrations[twin_id].is_active:
                    continue
                
                score = 0
                
                # Twin type match
                if not rule.twin_types or twin.twin_type in rule.twin_types:
                    score += rule.priority * 0.1
                
                # Domain match
                twin_domains = self.twin_registrations[twin_id].domains
                if rule.domains:
                    domain_matches = sum(1 for domain in rule.domains if domain in twin_domains)
                    score += domain_matches * rule.priority * 0.05
                elif twin_domains:  # Bonus for domain expertise
                    score += rule.priority * 0.02
                
                # Keyword match bonus
                score += keyword_matches * rule.priority * 0.03
                
                twin_scores[twin_id] = twin_scores.get(twin_id, 0) + score
        
        # Calculate relevance scores for all twins
        for twin_id, twin in self.twins.items():
            if twin_id not in twin_scores:
                relevance_score = await self._calculate_twin_relevance_score(twin, query, context)
                twin_scores[twin_id] = relevance_score * 10  # Scale to match rule scores
        
        # Select top twins
        sorted_twins = sorted(twin_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 3 twins with score > threshold
        threshold = 1.0
        selected_twins = []
        for twin_id, score in sorted_twins[:3]:
            if score > threshold:
                selected_twins.append(self.twins[twin_id])
        
        return selected_twins if selected_twins else [sorted_twins[0][1]] if sorted_twins else []
    
    async def _calculate_twin_relevance_score(
        self,
        twin: BaseTwin,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate relevance score for a twin given a query."""
        score = 0.0
        query_lower = query.lower()
        
        # Twin type relevance
        type_keywords = {
            "expert": ["expert", "validate", "review", "professional", "specialist"],
            "user_journey": ["journey", "user", "experience", "persona", "customer"],
            "process_automation": ["process", "automate", "workflow", "efficiency", "optimize"]
        }
        
        if twin.twin_type in type_keywords:
            matches = sum(1 for keyword in type_keywords[twin.twin_type] if keyword in query_lower)
            score += matches * 0.2
        
        # Domain relevance
        twin_domains = self.twin_registrations[twin.twin_id].domains
        for domain in twin_domains:
            if domain.lower() in query_lower:
                score += 0.3
        
        # Performance history
        performance = self.twin_performance.get(twin.twin_id, {})
        success_rate = performance.get("success_rate", 1.0)
        avg_confidence = performance.get("average_confidence", 0.5)
        
        score += (success_rate * 0.1) + (avg_confidence * 0.1)
        
        # Priority bonus
        priority = self.twin_registrations[twin.twin_id].priority
        score += (priority / 10) * 0.1
        
        return min(score, 1.0)
    
    async def _process_single_twin(
        self,
        twin: BaseTwin,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> TwinInteractionResult:
        """Process query through a single twin."""
        start_time = datetime.utcnow()
        
        try:
            # Execute twin interaction
            interaction_result = await twin.interact(query, context, user_id)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create result
            result = TwinInteractionResult(
                twin_id=twin.twin_id,
                interaction_id=interaction_result["interaction_id"],
                response=interaction_result["response"],
                confidence=interaction_result["confidence"],
                processing_time_ms=processing_time,
                metadata={
                    "twin_type": twin.twin_type,
                    "twin_name": twin.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_metadata": interaction_result.get("processing_metadata", {}),
                    "twin_metadata": interaction_result.get("twin_metadata", {})
                }
            )
            
            # Store in history
            self.interaction_history.append(result)
            
            # Update twin performance
            self._update_twin_performance(twin.twin_id, result)
            
            return result
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            self.logger.error(
                "Single twin processing failed",
                twin_id=twin.twin_id,
                error=str(e)
            )
            
            # Return error result
            return TwinInteractionResult(
                twin_id=twin.twin_id,
                interaction_id=f"error_{twin.twin_id}_{int(datetime.utcnow().timestamp())}",
                response=f"Processing failed: {str(e)}",
                confidence=0.0,
                processing_time_ms=processing_time,
                metadata={
                    "error": True,
                    "error_message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def _process_collaborative(
        self,
        twins: List[BaseTwin],
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> CollaborativeResult:
        """Process query through collaborative twin interaction."""
        start_time = datetime.utcnow()
        
        try:
            # Execute twins in parallel
            tasks = [
                self._process_single_twin(twin, query, context, user_id)
                for twin in twins
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = [
                result for result in results
                if isinstance(result, TwinInteractionResult) and not result.metadata.get("error", False)
            ]
            
            if not successful_results:
                raise PersonaTwinError("All twin interactions failed", query[:50])
            
            # Synthesize responses
            synthesized_response = await self._synthesize_responses(successful_results, query)
            
            # Calculate overall confidence
            confidence_scores = [result.confidence for result in successful_results]
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            # Create collaborative result
            collaborative_result = CollaborativeResult(
                query=query,
                primary_twin=successful_results[0].twin_id,  # Highest scoring twin
                contributing_twins=[result.twin_id for result in successful_results[1:]],
                synthesized_response=synthesized_response,
                confidence_score=overall_confidence,
                collaboration_metadata={
                    "twins_involved": len(successful_results),
                    "synthesis_method": "weighted_consensus",
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                    "individual_results": [
                        {
                            "twin_id": result.twin_id,
                            "confidence": result.confidence,
                            "processing_time_ms": result.processing_time_ms
                        }
                        for result in successful_results
                    ]
                }
            )
            
            return collaborative_result
            
        except Exception as e:
            self.logger.error(
                "Collaborative processing failed",
                twins_count=len(twins),
                error=str(e)
            )
            raise PersonaTwinError(f"Collaborative processing failed: {e}", query[:50])
    
    async def _synthesize_responses(
        self,
        results: List[TwinInteractionResult],
        query: str
    ) -> str:
        """Synthesize multiple twin responses into a unified response."""
        if len(results) == 1:
            return results[0].response
        
        # Weight responses by confidence
        weighted_responses = []
        total_weight = 0
        
        for result in results:
            weight = result.confidence
            total_weight += weight
            weighted_responses.append({
                "response": result.response,
                "weight": weight,
                "twin_type": result.metadata.get("twin_type", "unknown"),
                "twin_name": result.metadata.get("twin_name", "unknown")
            })
        
        # Sort by weight (confidence)
        weighted_responses.sort(key=lambda x: x["weight"], reverse=True)
        
        # Create synthesized response
        synthesis = "**Collaborative Twin Analysis:**\n\n"
        
        # Primary response (highest confidence)
        primary = weighted_responses[0]
        synthesis += f"**Primary Analysis** ({primary['twin_name']}):\n"
        synthesis += f"{primary['response']}\n\n"
        
        # Supporting insights
        if len(weighted_responses) > 1:
            synthesis += "**Supporting Insights:**\n\n"
            
            for supporting in weighted_responses[1:]:
                synthesis += f"**{supporting['twin_name']} Perspective:**\n"
                
                # Extract key insights (first 2-3 sentences or bullet points)
                response_lines = supporting['response'].split('\n')
                key_insights = []
                
                for line in response_lines[:10]:  # Look at first 10 lines
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('*') or line.endswith('.')):
                        key_insights.append(line)
                        if len(key_insights) >= 3:
                            break
                
                if key_insights:
                    synthesis += '\n'.join(key_insights[:3]) + "\n\n"
                else:
                    # Fallback: use first sentence
                    first_sentence = supporting['response'].split('.')[0] + '.'
                    synthesis += f"{first_sentence}\n\n"
        
        # Consensus summary
        synthesis += "**Synthesis:**\n"
        synthesis += f"This analysis combines insights from {len(results)} specialized twins, "
        synthesis += f"providing a comprehensive perspective on your query. "
        synthesis += f"The recommendations reflect consensus across multiple domains of expertise."
        
        return synthesis
    
    def _update_performance_metrics(
        self,
        result: Union[TwinInteractionResult, CollaborativeResult],
        processing_time: float
    ) -> None:
        """Update orchestrator performance metrics."""
        # Update average response time
        if self.total_interactions == 1:
            self.average_response_time = processing_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_interactions - 1) + processing_time) 
                / self.total_interactions
            )
    
    def _update_twin_performance(
        self,
        twin_id: str,
        result: TwinInteractionResult
    ) -> None:
        """Update performance metrics for a specific twin."""
        if twin_id not in self.twin_performance:
            self.twin_performance[twin_id] = {
                "interaction_count": 0,
                "average_confidence": 0.0,
                "average_response_time": 0.0,
                "success_rate": 1.0,
                "last_interaction": None
            }
        
        perf = self.twin_performance[twin_id]
        
        # Update interaction count
        perf["interaction_count"] += 1
        count = perf["interaction_count"]
        
        # Update average confidence
        perf["average_confidence"] = (
            (perf["average_confidence"] * (count - 1) + result.confidence) / count
        )
        
        # Update average response time
        perf["average_response_time"] = (
            (perf["average_response_time"] * (count - 1) + result.processing_time_ms) / count
        )
        
        # Update success rate (based on confidence > 0.5)
        successes = (perf["success_rate"] * (count - 1)) + (1 if result.confidence > 0.5 else 0)
        perf["success_rate"] = successes / count
        
        # Update last interaction
        perf["last_interaction"] = result.metadata.get("timestamp")
    
    def _infer_twin_capabilities(self, twin: BaseTwin) -> List[str]:
        """Infer capabilities from twin type and characteristics."""
        capabilities = []
        
        if twin.twin_type == "expert":
            capabilities.extend(["validation", "consultation", "expert_analysis", "domain_expertise"])
        elif twin.twin_type == "user_journey":
            capabilities.extend(["journey_mapping", "pain_point_analysis", "persona_analysis", "optimization"])
        elif twin.twin_type == "process_automation":
            capabilities.extend(["process_analysis", "automation_recommendations", "bottleneck_detection", "roi_calculation"])
        
        # Add common capabilities
        capabilities.extend(["query_processing", "adaptive_learning", "performance_tracking"])
        
        return capabilities
    
    def _infer_twin_domains(self, twin: BaseTwin) -> List[str]:
        """Infer domain expertise from twin type and attributes."""
        domains = []
        
        if twin.twin_type == "expert" and hasattr(twin, 'domain'):
            domains.append(twin.domain.domain_name)
            domains.extend(twin.domain.specializations)
        elif twin.twin_type == "user_journey":
            domains.extend(["user_experience", "customer_journey", "persona_development"])
        elif twin.twin_type == "process_automation":
            domains.extend(["business_process", "automation", "efficiency_optimization"])
        
        return domains
    
    def _get_twins_by_type(self, twin_type: str) -> List[BaseTwin]:
        """Get active twins of a specific type."""
        return [
            twin for twin in self.twins.values()
            if twin.twin_type == twin_type and 
            self.twin_registrations[twin.twin_id].is_active
        ]
    
    def _estimate_response_confidence(self, twin: BaseTwin, query: str) -> float:
        """Estimate response confidence for a twin given a query."""
        # Base estimate on twin performance and query relevance
        performance = self.twin_performance.get(twin.twin_id, {})
        base_confidence = performance.get("average_confidence", 0.5)
        
        # Adjust based on query relevance
        relevance_adjustment = 0.1  # Small adjustment for now
        
        return min(base_confidence + relevance_adjustment, 1.0)


# Global orchestrator instance
_orchestrator: Optional[TwinOrchestrator] = None


def get_twin_orchestrator() -> TwinOrchestrator:
    """
    Get the global twin orchestrator instance.
    
    Returns:
        TwinOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = TwinOrchestrator()
    return _orchestrator


async def initialize_default_twins() -> TwinOrchestrator:
    """
    Initialize orchestrator with default twins.
    
    Returns:
        Initialized orchestrator
    """
    orchestrator = get_twin_orchestrator()
    
    # Initialize sample expert twin
    compliance_domain = ExpertDomain(
        domain_name="compliance",
        expertise_level=0.9,
        specializations=["FedRAMP", "NIST", "security controls", "audit"],
        experience_years=10,
        knowledge_sources=["NIST 800-53", "FedRAMP guidelines", "industry standards"],
        validation_criteria={"min_score": 0.8, "evidence_required": True}
    )
    
    compliance_expert = ExpertTwin(
        expert_id="compliance_expert_001",
        name="Dr. Sarah Mitchell",
        domain=compliance_domain,
        description="Senior compliance expert specializing in FedRAMP and NIST frameworks"
    )
    
    await orchestrator.register_twin(
        compliance_expert,
        capabilities=["compliance_validation", "security_assessment", "audit_preparation"],
        domains=["compliance", "security", "government"]
    )
    
    # Initialize sample user journey twin
    user_persona = UserPersona(
        persona_id="enterprise_admin",
        persona_name="Enterprise System Administrator",
        digital_literacy=0.9,
        patience_level=0.6,
        primary_goals=["system_efficiency", "security_compliance", "user_satisfaction"],
        preferred_channels=["web_interface", "api", "documentation"]
    )
    
    journey_steps = [
        UserJourneyStep(
            step_id="discovery",
            step_name="System Discovery",
            step_type="discovery",
            description="User discovers and evaluates the system",
            completion_rate=0.85,
            average_duration_minutes=15
        ),
        UserJourneyStep(
            step_id="onboarding",
            step_name="User Onboarding",
            step_type="action",
            description="User completes initial setup and configuration",
            completion_rate=0.75,
            average_duration_minutes=45
        )
    ]
    
    user_journey = UserJourneyTwin(
        journey_id="enterprise_admin_journey",
        name="Enterprise Admin User Journey",
        persona=user_persona,
        journey_steps=journey_steps,
        description="User journey for enterprise system administrators"
    )
    
    await orchestrator.register_twin(
        user_journey,
        capabilities=["journey_optimization", "persona_analysis", "ux_improvement"],
        domains=["user_experience", "enterprise_software", "administration"]
    )
    
    # Initialize sample process automation twin
    process_steps = [
        ProcessStep(
            step_id="data_collection",
            step_name="Data Collection",
            step_type="manual",
            description="Collect and validate input data",
            automation_potential=0.8,
            average_duration_minutes=30,
            error_rate=0.05,
            cost_per_execution=25.0
        ),
        ProcessStep(
            step_id="analysis",
            step_name="Data Analysis",
            step_type="automated",
            description="Automated data analysis and reporting",
            is_automated=True,
            automation_potential=0.95,
            average_duration_minutes=5,
            error_rate=0.01,
            cost_per_execution=5.0
        )
    ]
    
    process_metrics = ProcessMetrics(
        cycle_time_minutes=45,
        processing_time_minutes=35,
        wait_time_minutes=10,
        first_time_right_rate=0.9,
        daily_volume=100,
        total_cost_per_transaction=30.0
    )
    
    process_automation = ProcessAutomationTwin(
        process_id="data_processing_workflow",
        name="Data Processing Workflow",
        process_steps=process_steps,
        current_metrics=process_metrics,
        description="Automated data processing and analysis workflow"
    )
    
    await orchestrator.register_twin(
        process_automation,
        capabilities=["process_optimization", "automation_planning", "roi_analysis"],
        domains=["business_process", "data_processing", "automation"]
    )
    
    return orchestrator