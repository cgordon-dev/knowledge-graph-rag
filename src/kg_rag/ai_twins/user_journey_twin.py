"""
User Journey Digital Twin for simulating user behavior patterns.

Models user interactions, preferences, pain points, and journey optimization
with adaptive learning and persona-driven query handling.
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from kg_rag.ai_twins.base_twin import BaseTwin, TwinCharacteristics
from kg_rag.core.exceptions import PersonaTwinError, PersonaValidationError
from kg_rag.mcp_servers.orchestrator import get_orchestrator


class UserJourneyStep(BaseModel):
    """Represents a step in a user journey."""
    
    step_id: str = Field(..., description="Unique step identifier")
    step_name: str = Field(..., description="Human-readable step name")
    step_type: str = Field(..., description="Step type (discovery, evaluation, decision, action)")
    description: str = Field(..., description="Step description")
    
    # Journey context
    touchpoints: List[str] = Field(default_factory=list, description="User touchpoints")
    pain_points: List[str] = Field(default_factory=list, description="Pain points in this step")
    success_criteria: List[str] = Field(default_factory=list, description="Success criteria")
    
    # Performance metrics
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Step completion rate")
    average_duration_minutes: float = Field(default=0.0, description="Average time spent")
    abandonment_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Step abandonment rate")
    
    # Optimization potential
    optimization_opportunities: List[str] = Field(default_factory=list, description="Improvement opportunities")
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Optimization priority")


class UserPersona(BaseModel):
    """User persona characteristics."""
    
    persona_id: str = Field(..., description="Persona identifier")
    persona_name: str = Field(..., description="Persona name")
    demographic_profile: Dict[str, Any] = Field(default_factory=dict, description="Demographic information")
    
    # Behavioral patterns
    digital_literacy: float = Field(default=0.7, ge=0.0, le=1.0, description="Digital comfort level")
    patience_level: float = Field(default=0.6, ge=0.0, le=1.0, description="Patience with complex processes")
    detail_preference: float = Field(default=0.5, ge=0.0, le=1.0, description="Preference for detailed information")
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk tolerance level")
    
    # Goals and motivations
    primary_goals: List[str] = Field(default_factory=list, description="Primary goals")
    motivations: List[str] = Field(default_factory=list, description="Key motivations")
    frustrations: List[str] = Field(default_factory=list, description="Common frustrations")
    
    # Journey preferences
    preferred_channels: List[str] = Field(default_factory=list, description="Preferred communication channels")
    decision_factors: List[str] = Field(default_factory=list, description="Key decision factors")


class JourneyOptimization(BaseModel):
    """Journey optimization recommendation."""
    
    optimization_id: str = Field(..., description="Optimization identifier")
    journey_step: str = Field(..., description="Target journey step")
    optimization_type: str = Field(..., description="Type of optimization")
    
    # Analysis
    current_performance: Dict[str, float] = Field(default_factory=dict, description="Current metrics")
    pain_point_analysis: List[str] = Field(default_factory=list, description="Pain points identified")
    opportunity_assessment: str = Field(..., description="Opportunity description")
    
    # Recommendations
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    expected_impact: Dict[str, float] = Field(default_factory=dict, description="Expected impact metrics")
    implementation_complexity: str = Field(default="medium", description="Implementation complexity")
    
    # Prioritization
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Priority score")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")


class UserJourneyCharacteristics(TwinCharacteristics):
    """Extended characteristics for user journey twins."""
    
    # Journey-specific traits
    empathy_level: float = Field(default=0.8, ge=0.0, le=1.0, description="User empathy level")
    optimization_focus: float = Field(default=0.7, ge=0.0, le=1.0, description="Focus on optimization")
    data_driven_approach: float = Field(default=0.8, ge=0.0, le=1.0, description="Data-driven decision making")
    user_advocacy: float = Field(default=0.9, ge=0.0, le=1.0, description="User advocacy strength")
    
    # Analysis patterns
    pain_point_sensitivity: float = Field(default=0.8, ge=0.0, le=1.0, description="Sensitivity to user pain points")
    journey_completeness: float = Field(default=0.7, ge=0.0, le=1.0, description="Focus on complete journey view")
    
    # Communication style
    storytelling_preference: float = Field(default=0.8, ge=0.0, le=1.0, description="Use of storytelling")
    metric_orientation: float = Field(default=0.7, ge=0.0, le=1.0, description="Focus on metrics and data")


class UserJourneyTwin(BaseTwin):
    """
    User Journey Digital Twin for modeling user behavior and optimization.
    
    Simulates user journeys, identifies pain points, recommends optimizations,
    and adapts based on user feedback and behavior patterns.
    """
    
    def __init__(
        self,
        journey_id: str,
        name: str,
        persona: UserPersona,
        journey_steps: List[UserJourneyStep],
        characteristics: Optional[UserJourneyCharacteristics] = None,
        description: Optional[str] = None
    ):
        """
        Initialize User Journey Digital Twin.
        
        Args:
            journey_id: Unique journey identifier
            name: Journey name
            persona: User persona model
            journey_steps: Journey steps definition
            characteristics: Journey twin characteristics
            description: Journey description
        """
        self.persona = persona
        self.journey_steps = {step.step_id: step for step in journey_steps}
        
        # Generate description if not provided
        if not description:
            description = f"User journey for {persona.persona_name} with {len(journey_steps)} steps"
        
        # Initialize with journey characteristics
        journey_chars = characteristics or UserJourneyCharacteristics()
        
        super().__init__(
            twin_id=journey_id,
            twin_type="user_journey",
            name=name,
            description=description,
            characteristics=journey_chars
        )
        
        # Journey-specific state
        self.optimization_history: List[JourneyOptimization] = []
        self.journey_analytics: Dict[str, Any] = {}
        self.persona_insights: Dict[str, Any] = {}
        
        # Performance tracking
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.average_optimization_impact = 0.0
        
        self.logger.info(
            "User journey twin initialized",
            journey_id=journey_id,
            persona=persona.persona_name,
            steps_count=len(journey_steps)
        )
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process query from user journey perspective.
        
        Args:
            query: User journey query
            context: Additional context
            
        Returns:
            Journey-focused response with optimization recommendations
        """
        try:
            # Classify query type
            query_type = self._classify_journey_query(query)
            
            # Generate journey-specific response
            if query_type == "pain_point_analysis":
                response_data = await self._analyze_pain_points(query, context)
            elif query_type == "optimization_recommendation":
                response_data = await self._generate_optimization_recommendations(query, context)
            elif query_type == "journey_mapping":
                response_data = await self._provide_journey_mapping(query, context)
            elif query_type == "persona_analysis":
                response_data = await self._analyze_persona_behavior(query, context)
            else:
                response_data = await self._generate_general_journey_response(query, context)
            
            # Add journey context and persona insights
            enhanced_response = await self._enhance_with_journey_context(response_data, query)
            
            # Calculate confidence based on journey knowledge and persona fit
            confidence = self._calculate_journey_confidence(query, response_data)
            
            return {
                "response": enhanced_response,
                "confidence": confidence,
                "metadata": {
                    "query_type": query_type,
                    "persona_name": self.persona.persona_name,
                    "journey_steps_analyzed": len(self.journey_steps),
                    "optimization_count": len(self.optimization_history),
                    "journey_perspective": True
                }
            }
            
        except Exception as e:
            self.logger.error(
                "Journey query processing failed",
                journey_id=self.twin_id,
                persona=self.persona.persona_name,
                error=str(e)
            )
            raise PersonaTwinError(f"Journey processing failed: {e}", self.persona.persona_name)
    
    async def analyze_journey_step(
        self,
        step_id: str,
        performance_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a specific journey step for optimization opportunities.
        
        Args:
            step_id: Journey step identifier
            performance_data: Step performance metrics
            
        Returns:
            Step analysis with optimization recommendations
        """
        try:
            if step_id not in self.journey_steps:
                raise PersonaValidationError(
                    f"Journey step '{step_id}' not found",
                    persona_id=self.twin_id,
                    validation_errors={"step_id": step_id}
                )
            
            step = self.journey_steps[step_id]
            
            # Analyze step performance
            step_analysis = {
                "step_info": {
                    "step_id": step_id,
                    "step_name": step.step_name,
                    "step_type": step.step_type,
                    "description": step.description
                },
                "current_performance": {
                    "completion_rate": step.completion_rate,
                    "average_duration": step.average_duration_minutes,
                    "abandonment_rate": step.abandonment_rate
                },
                "pain_points": step.pain_points.copy(),
                "touchpoints": step.touchpoints.copy()
            }
            
            # Update with new performance data if provided
            if performance_data:
                step_analysis["current_performance"].update(performance_data)
                # Update the step object
                for key, value in performance_data.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
            
            # Generate optimization recommendations
            optimization_opportunities = await self._identify_step_optimizations(step, step_analysis)
            step_analysis["optimization_opportunities"] = optimization_opportunities
            
            # Calculate step health score
            health_score = self._calculate_step_health_score(step)
            step_analysis["health_score"] = health_score
            
            # Persona-specific insights
            persona_insights = self._generate_persona_insights_for_step(step)
            step_analysis["persona_insights"] = persona_insights
            
            return step_analysis
            
        except Exception as e:
            self.logger.error(
                "Journey step analysis failed",
                step_id=step_id,
                error=str(e)
            )
            raise PersonaTwinError(f"Step analysis failed: {e}", step_id)
    
    async def recommend_journey_optimizations(
        self,
        priority_threshold: float = 0.6,
        max_recommendations: int = 10
    ) -> List[JourneyOptimization]:
        """
        Generate comprehensive journey optimization recommendations.
        
        Args:
            priority_threshold: Minimum priority score for recommendations
            max_recommendations: Maximum number of recommendations
            
        Returns:
            List of prioritized optimization recommendations
        """
        try:
            recommendations = []
            
            # Analyze each journey step
            for step_id, step in self.journey_steps.items():
                step_health = self._calculate_step_health_score(step)
                
                if step_health < 0.8:  # Step needs optimization
                    # Generate optimization for this step
                    optimization = await self._create_step_optimization(step, step_health)
                    
                    if optimization.priority_score >= priority_threshold:
                        recommendations.append(optimization)
            
            # Add journey-wide optimizations
            journey_wide_optimizations = await self._identify_journey_wide_optimizations()
            for opt in journey_wide_optimizations:
                if opt.priority_score >= priority_threshold:
                    recommendations.append(opt)
            
            # Sort by priority and limit results
            recommendations.sort(key=lambda x: x.priority_score, reverse=True)
            recommendations = recommendations[:max_recommendations]
            
            # Store recommendations in history
            self.optimization_history.extend(recommendations)
            self.total_optimizations += len(recommendations)
            
            self.logger.info(
                "Journey optimizations generated",
                journey_id=self.twin_id,
                recommendations_count=len(recommendations),
                avg_priority=np.mean([r.priority_score for r in recommendations]) if recommendations else 0
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(
                "Journey optimization generation failed",
                error=str(e)
            )
            raise PersonaTwinError(f"Optimization generation failed: {e}", self.twin_id)
    
    def get_persona_prompt(self) -> str:
        """Generate persona prompt for user journey twin."""
        goals_text = ", ".join(self.persona.primary_goals) if self.persona.primary_goals else "general user satisfaction"
        channels_text = ", ".join(self.persona.preferred_channels) if self.persona.preferred_channels else "various touchpoints"
        
        prompt = f"""You are {self.name}, a user journey specialist focused on optimizing the experience for {self.persona.persona_name}.

## User Persona Profile
- **Persona**: {self.persona.persona_name}
- **Digital Literacy**: {self.persona.digital_literacy:.1%}
- **Patience Level**: {self.persona.patience_level:.1%}
- **Primary Goals**: {goals_text}
- **Preferred Channels**: {channels_text}

## Journey Characteristics
- **Empathy Level**: {self.characteristics.empathy_level:.1%} - You deeply understand user needs and frustrations
- **Optimization Focus**: {self.characteristics.optimization_focus:.1%} - You actively seek improvement opportunities
- **User Advocacy**: {self.characteristics.user_advocacy:.1%} - You champion user needs in all recommendations
- **Pain Point Sensitivity**: {self.characteristics.pain_point_sensitivity:.1%} - You identify and address user frustrations

## Journey Steps Managed
- **Total Steps**: {len(self.journey_steps)}
- **Optimization History**: {len(self.optimization_history)} previous optimizations
- **Success Rate**: {(self.successful_optimizations / max(self.total_optimizations, 1)) * 100:.1f}%

## Response Guidelines
1. **User-Centered Perspective**: Always view challenges through the user's eyes
2. **Journey Completeness**: Consider the entire user journey, not just individual steps
3. **Data-Driven Insights**: Support recommendations with performance metrics
4. **Empathetic Communication**: Acknowledge user frustrations and pain points
5. **Optimization Focus**: Provide actionable recommendations for improvement

## Communication Style
- **Storytelling**: {self.characteristics.storytelling_preference:.1%} - Use narrative to explain user experiences
- **Metric Orientation**: {self.characteristics.metric_orientation:.1%} - Support insights with data
- **Formality**: {self.characteristics.formality_level:.1%} professional but empathetic
- **Detail Level**: {self.characteristics.detail_preference:.1%} - Provide comprehensive journey analysis

When responding, maintain your user journey perspective while providing actionable insights for optimization."""
        
        return prompt
    
    def _classify_journey_query(self, query: str) -> str:
        """Classify query type for journey analysis."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["pain point", "frustration", "problem", "issue"]):
            return "pain_point_analysis"
        elif any(term in query_lower for term in ["optimize", "improve", "enhance", "better"]):
            return "optimization_recommendation"
        elif any(term in query_lower for term in ["journey", "map", "flow", "path"]):
            return "journey_mapping"
        elif any(term in query_lower for term in ["persona", "user", "behavior", "preference"]):
            return "persona_analysis"
        else:
            return "general_journey"
    
    async def _analyze_pain_points(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Analyze pain points in the user journey."""
        all_pain_points = []
        step_pain_points = {}
        
        for step_id, step in self.journey_steps.items():
            if step.pain_points:
                all_pain_points.extend(step.pain_points)
                step_pain_points[step.step_name] = step.pain_points
        
        response = f"**Pain Point Analysis for {self.persona.persona_name}:**\n\n"
        
        if step_pain_points:
            response += "**Journey Step Pain Points:**\n"
            for step_name, pain_points in step_pain_points.items():
                response += f"- **{step_name}**: {', '.join(pain_points)}\n"
        
        response += f"\n**Persona-Specific Considerations:**\n"
        response += f"- Digital Literacy Level: {self.persona.digital_literacy:.1%} - "
        if self.persona.digital_literacy < 0.5:
            response += "May struggle with complex interfaces\n"
        else:
            response += "Comfortable with digital interactions\n"
        
        response += f"- Patience Level: {self.persona.patience_level:.1%} - "
        if self.persona.patience_level < 0.5:
            response += "Requires streamlined, efficient processes\n"
        else:
            response += "Can tolerate more complex workflows\n"
        
        if self.persona.frustrations:
            response += f"- Known Frustrations: {', '.join(self.persona.frustrations)}\n"
        
        return response
    
    async def _generate_optimization_recommendations(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate optimization recommendations."""
        recommendations = await self.recommend_journey_optimizations(priority_threshold=0.5, max_recommendations=5)
        
        response = f"**Journey Optimization Recommendations for {self.persona.persona_name}:**\n\n"
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                response += f"**{i}. {rec.optimization_type.title()} - {rec.journey_step}**\n"
                response += f"   Priority: {rec.priority_score:.1%} | Confidence: {rec.confidence_level:.1%}\n"
                response += f"   Opportunity: {rec.opportunity_assessment}\n"
                
                if rec.recommended_actions:
                    response += f"   Actions: {', '.join(rec.recommended_actions[:3])}\n"
                
                if rec.expected_impact:
                    impact_items = [f"{k}: +{v:.1%}" for k, v in rec.expected_impact.items()]
                    response += f"   Expected Impact: {', '.join(impact_items)}\n"
                
                response += "\n"
        else:
            response += "No high-priority optimization opportunities identified at this time.\n"
            response += "Current journey performance appears to be meeting user needs effectively.\n"
        
        return response
    
    async def _provide_journey_mapping(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Provide journey mapping insights."""
        response = f"**User Journey Map for {self.persona.persona_name}:**\n\n"
        
        # Journey overview
        response += f"**Journey Overview:**\n"
        response += f"- Total Steps: {len(self.journey_steps)}\n"
        response += f"- Primary Goals: {', '.join(self.persona.primary_goals) if self.persona.primary_goals else 'General satisfaction'}\n"
        response += f"- Preferred Channels: {', '.join(self.persona.preferred_channels) if self.persona.preferred_channels else 'Various touchpoints'}\n\n"
        
        # Step-by-step breakdown
        response += "**Journey Steps:**\n"
        sorted_steps = sorted(self.journey_steps.values(), key=lambda x: x.step_name)
        
        for step in sorted_steps:
            health_score = self._calculate_step_health_score(step)
            health_emoji = "🟢" if health_score > 0.8 else "🟡" if health_score > 0.6 else "🔴"
            
            response += f"{health_emoji} **{step.step_name}** ({step.step_type})\n"
            response += f"   Completion Rate: {step.completion_rate:.1%} | "
            response += f"Avg Duration: {step.average_duration_minutes:.1f} min\n"
            
            if step.touchpoints:
                response += f"   Touchpoints: {', '.join(step.touchpoints)}\n"
            
            if step.pain_points:
                response += f"   Pain Points: {', '.join(step.pain_points)}\n"
            
            response += "\n"
        
        return response
    
    async def _analyze_persona_behavior(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Analyze persona behavior patterns."""
        response = f"**Persona Behavior Analysis: {self.persona.persona_name}**\n\n"
        
        response += "**Behavioral Profile:**\n"
        response += f"- Digital Literacy: {self.persona.digital_literacy:.1%}\n"
        response += f"- Patience Level: {self.persona.patience_level:.1%}\n"
        response += f"- Detail Preference: {self.persona.detail_preference:.1%}\n"
        response += f"- Risk Tolerance: {self.persona.risk_tolerance:.1%}\n\n"
        
        if self.persona.motivations:
            response += f"**Key Motivations:**\n"
            for motivation in self.persona.motivations:
                response += f"- {motivation}\n"
            response += "\n"
        
        if self.persona.decision_factors:
            response += f"**Decision Factors:**\n"
            for factor in self.persona.decision_factors:
                response += f"- {factor}\n"
            response += "\n"
        
        # Journey-specific insights
        avg_completion = np.mean([step.completion_rate for step in self.journey_steps.values()])
        avg_duration = np.mean([step.average_duration_minutes for step in self.journey_steps.values()])
        
        response += f"**Journey Performance:**\n"
        response += f"- Average Step Completion: {avg_completion:.1%}\n"
        response += f"- Average Step Duration: {avg_duration:.1f} minutes\n"
        response += f"- Total Optimizations Applied: {len(self.optimization_history)}\n"
        
        return response
    
    async def _generate_general_journey_response(self, query: str, context: Optional[Dict[str, Any]]) -> str:
        """Generate general journey-focused response."""
        response = f"**User Journey Insights for {self.persona.persona_name}:**\n\n"
        
        # Journey health summary
        step_health_scores = [self._calculate_step_health_score(step) for step in self.journey_steps.values()]
        avg_health = np.mean(step_health_scores) if step_health_scores else 0.0
        
        if avg_health > 0.8:
            response += "✅ **Journey Health: Excellent** - The user journey is performing well across all steps.\n\n"
        elif avg_health > 0.6:
            response += "⚠️ **Journey Health: Good** - Some optimization opportunities exist.\n\n"
        else:
            response += "🔴 **Journey Health: Needs Attention** - Multiple steps require optimization.\n\n"
        
        # Key insights
        response += "**Key Journey Insights:**\n"
        response += f"- Persona Alignment: Journey designed for {self.persona.digital_literacy:.1%} digital literacy\n"
        response += f"- Optimization Potential: {len([s for s in step_health_scores if s < 0.8])} steps need improvement\n"
        response += f"- User Advocacy Focus: Recommendations prioritize user needs and experience\n"
        
        return response
    
    async def _enhance_with_journey_context(self, response: str, query: str) -> str:
        """Enhance response with journey context and persona insights."""
        if self.characteristics.storytelling_preference > 0.7:
            response += f"\n\n*Journey Perspective: This analysis reflects the experience of {self.persona.persona_name}, "
            response += f"considering their {self.persona.digital_literacy:.1%} digital comfort level and journey preferences.*"
        
        if self.characteristics.metric_orientation > 0.7:
            avg_health = np.mean([self._calculate_step_health_score(step) for step in self.journey_steps.values()])
            response += f"\n\n*Journey Health Score: {avg_health:.1%} | Optimization Count: {len(self.optimization_history)}*"
        
        return response
    
    def _calculate_journey_confidence(self, query: str, response_data: str) -> float:
        """Calculate confidence in journey response."""
        # Base confidence on journey completeness and persona alignment
        base_confidence = 0.7
        
        # Boost for journey-specific queries
        if any(term in query.lower() for term in ["journey", "user", "experience", "optimize"]):
            base_confidence += 0.1
        
        # Boost for detailed response
        if len(response_data) > 300:
            base_confidence += 0.1
        
        # Boost for persona alignment
        if self.persona.persona_name.lower() in response_data.lower():
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    async def _identify_step_optimizations(self, step: UserJourneyStep, analysis: Dict[str, Any]) -> List[str]:
        """Identify optimization opportunities for a specific step."""
        optimizations = []
        
        # Check completion rate
        if step.completion_rate < 0.8:
            optimizations.append(f"Improve step clarity and user guidance (completion: {step.completion_rate:.1%})")
        
        # Check abandonment rate
        if step.abandonment_rate > 0.2:
            optimizations.append(f"Reduce friction points (abandonment: {step.abandonment_rate:.1%})")
        
        # Check duration vs persona patience
        if step.average_duration_minutes > 10 and self.persona.patience_level < 0.5:
            optimizations.append("Streamline process for low-patience users")
        
        # Check pain points
        if step.pain_points:
            optimizations.append(f"Address {len(step.pain_points)} identified pain points")
        
        # Persona-specific optimizations
        if self.persona.digital_literacy < 0.5 and step.step_type == "discovery":
            optimizations.append("Simplify interface for users with lower digital literacy")
        
        return optimizations
    
    def _calculate_step_health_score(self, step: UserJourneyStep) -> float:
        """Calculate health score for a journey step."""
        # Combine multiple factors
        completion_factor = step.completion_rate
        duration_factor = max(0, 1 - (step.average_duration_minutes / 30))  # Penalize long durations
        abandonment_factor = 1 - step.abandonment_rate
        pain_point_factor = max(0, 1 - (len(step.pain_points) / 5))  # Penalize many pain points
        
        health_score = (completion_factor * 0.4 + duration_factor * 0.2 + 
                       abandonment_factor * 0.3 + pain_point_factor * 0.1)
        
        return min(health_score, 1.0)
    
    def _generate_persona_insights_for_step(self, step: UserJourneyStep) -> Dict[str, str]:
        """Generate persona-specific insights for a step."""
        insights = {}
        
        # Digital literacy considerations
        if self.persona.digital_literacy < 0.5:
            insights["digital_literacy"] = "Consider simplified interface and additional guidance"
        elif self.persona.digital_literacy > 0.8:
            insights["digital_literacy"] = "Can handle advanced features and self-service options"
        
        # Patience level considerations
        if self.persona.patience_level < 0.5:
            insights["patience"] = "Prioritize speed and efficiency over comprehensive options"
        elif self.persona.patience_level > 0.8:
            insights["patience"] = "Can tolerate detailed processes if they add value"
        
        # Risk tolerance considerations
        if self.persona.risk_tolerance < 0.3:
            insights["risk"] = "Requires additional reassurance and safety measures"
        elif self.persona.risk_tolerance > 0.7:
            insights["risk"] = "Open to innovative solutions and new approaches"
        
        return insights
    
    async def _create_step_optimization(self, step: UserJourneyStep, health_score: float) -> JourneyOptimization:
        """Create optimization recommendation for a specific step."""
        optimization_id = f"opt_{self.twin_id}_{step.step_id}_{int(datetime.utcnow().timestamp())}"
        
        # Determine optimization type based on issues
        if step.abandonment_rate > 0.3:
            opt_type = "friction_reduction"
        elif step.completion_rate < 0.7:
            opt_type = "clarity_improvement"
        elif step.average_duration_minutes > 15:
            opt_type = "efficiency_enhancement"
        else:
            opt_type = "experience_optimization"
        
        # Generate recommendations based on persona and step issues
        recommendations = []
        if step.completion_rate < 0.8:
            recommendations.append("Improve step instructions and user guidance")
        if step.pain_points:
            recommendations.append(f"Address pain points: {', '.join(step.pain_points[:2])}")
        if self.persona.digital_literacy < 0.5:
            recommendations.append("Simplify interface for users with lower digital comfort")
        
        # Calculate expected impact
        expected_impact = {
            "completion_rate": min(0.2, 0.9 - step.completion_rate),
            "abandonment_rate": min(0.15, step.abandonment_rate),
            "user_satisfaction": 0.1
        }
        
        return JourneyOptimization(
            optimization_id=optimization_id,
            journey_step=step.step_name,
            optimization_type=opt_type,
            current_performance={
                "completion_rate": step.completion_rate,
                "abandonment_rate": step.abandonment_rate,
                "average_duration": step.average_duration_minutes
            },
            pain_point_analysis=step.pain_points.copy(),
            opportunity_assessment=f"Improve {step.step_name} performance from {health_score:.1%} health score",
            recommended_actions=recommendations,
            expected_impact=expected_impact,
            implementation_complexity="medium",
            priority_score=1.0 - health_score,  # Lower health = higher priority
            confidence_level=0.8
        )
    
    async def _identify_journey_wide_optimizations(self) -> List[JourneyOptimization]:
        """Identify optimizations that span multiple journey steps."""
        optimizations = []
        
        # Check for overall journey issues
        total_steps = len(self.journey_steps)
        avg_completion = np.mean([step.completion_rate for step in self.journey_steps.values()])
        avg_duration = np.mean([step.average_duration_minutes for step in self.journey_steps.values()])
        
        if avg_completion < 0.7:
            opt_id = f"journey_opt_{self.twin_id}_{int(datetime.utcnow().timestamp())}"
            optimizations.append(JourneyOptimization(
                optimization_id=opt_id,
                journey_step="entire_journey",
                optimization_type="journey_completion",
                current_performance={"average_completion": avg_completion},
                pain_point_analysis=["Low overall completion rates"],
                opportunity_assessment="Improve overall journey completion rates",
                recommended_actions=[
                    "Review and simplify journey flow",
                    "Add progress indicators",
                    "Implement save-and-resume functionality"
                ],
                expected_impact={"completion_rate": 0.15, "user_satisfaction": 0.2},
                implementation_complexity="high",
                priority_score=0.8,
                confidence_level=0.7
            ))
        
        if avg_duration > 20 and self.persona.patience_level < 0.5:
            opt_id = f"journey_opt_{self.twin_id}_{int(datetime.utcnow().timestamp())}_duration"
            optimizations.append(JourneyOptimization(
                optimization_id=opt_id,
                journey_step="entire_journey",
                optimization_type="journey_efficiency",
                current_performance={"average_duration": avg_duration},
                pain_point_analysis=["Journey too long for user patience level"],
                opportunity_assessment="Reduce overall journey time for impatient users",
                recommended_actions=[
                    "Eliminate non-essential steps",
                    "Parallelize where possible",
                    "Implement smart defaults"
                ],
                expected_impact={"duration_reduction": 0.3, "abandonment_rate": -0.1},
                implementation_complexity="medium",
                priority_score=0.7,
                confidence_level=0.8
            ))
        
        return optimizations
    
    def get_journey_statistics(self) -> Dict[str, Any]:
        """Get comprehensive journey statistics."""
        step_stats = []
        for step in self.journey_steps.values():
            step_stats.append({
                "step_name": step.step_name,
                "step_type": step.step_type,
                "completion_rate": step.completion_rate,
                "abandonment_rate": step.abandonment_rate,
                "health_score": self._calculate_step_health_score(step)
            })
        
        return {
            "journey_id": self.twin_id,
            "persona_name": self.persona.persona_name,
            "total_steps": len(self.journey_steps),
            "total_interactions": self.interaction_count,
            "optimization_count": len(self.optimization_history),
            "successful_optimizations": self.successful_optimizations,
            "optimization_success_rate": (
                self.successful_optimizations / max(self.total_optimizations, 1) * 100
            ),
            "average_step_health": np.mean([
                self._calculate_step_health_score(step) for step in self.journey_steps.values()
            ]),
            "step_statistics": step_stats,
            "persona_profile": {
                "digital_literacy": self.persona.digital_literacy,
                "patience_level": self.persona.patience_level,
                "primary_goals": self.persona.primary_goals
            },
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None
        }