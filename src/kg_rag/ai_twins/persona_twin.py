"""
Persona Digital Twin for general persona-based interactions.

Provides a flexible persona system for modeling different user types,
communication styles, and interaction patterns within the knowledge graph.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from kg_rag.ai_twins.base_twin import BaseTwin, TwinCharacteristics
from kg_rag.core.exceptions import PersonaTwinError, PersonaValidationError
from kg_rag.mcp_servers.orchestrator import get_orchestrator


class PersonaProfile(BaseModel):
    """Detailed persona profile configuration."""
    
    persona_name: str = Field(..., description="Persona name")
    role: str = Field(..., description="Professional role or context")
    background: str = Field(..., description="Background and experience")
    
    # Communication preferences
    communication_style: str = Field(default="professional", description="Communication style")
    preferred_detail_level: str = Field(default="moderate", description="Detail preference")
    technical_level: str = Field(default="intermediate", description="Technical expertise level")
    
    # Goals and motivations
    primary_objectives: List[str] = Field(default_factory=list, description="Primary objectives")
    key_concerns: List[str] = Field(default_factory=list, description="Key concerns and priorities")
    success_metrics: List[str] = Field(default_factory=list, description="Success criteria")
    
    # Domain expertise
    areas_of_expertise: List[str] = Field(default_factory=list, description="Areas of expertise")
    learning_interests: List[str] = Field(default_factory=list, description="Learning interests")
    
    # Behavioral patterns
    decision_making_style: str = Field(default="analytical", description="Decision-making style")
    collaboration_preference: str = Field(default="collaborative", description="Collaboration style")
    information_processing: str = Field(default="systematic", description="Information processing style")


class PersonaCharacteristics(TwinCharacteristics):
    """Extended characteristics for persona twins."""
    
    # Persona-specific traits
    authenticity_level: float = Field(default=0.8, ge=0.0, le=1.0, description="Persona authenticity")
    adaptability: float = Field(default=0.7, ge=0.0, le=1.0, description="Adaptability to context")
    consistency_maintenance: float = Field(default=0.9, ge=0.0, le=1.0, description="Persona consistency")
    context_sensitivity: float = Field(default=0.8, ge=0.0, le=1.0, description="Context sensitivity")
    
    # Interaction patterns
    proactiveness: float = Field(default=0.6, ge=0.0, le=1.0, description="Proactive communication")
    question_asking: float = Field(default=0.7, ge=0.0, le=1.0, description="Tendency to ask questions")
    example_usage: float = Field(default=0.8, ge=0.0, le=1.0, description="Use of examples")
    
    # Professional traits
    professional_tone: float = Field(default=0.8, ge=0.0, le=1.0, description="Professional tone level")
    supportiveness: float = Field(default=0.9, ge=0.0, le=1.0, description="Supportive communication")


class PersonaTwin(BaseTwin):
    """
    Persona Digital Twin for flexible persona-based interactions.
    
    Provides adaptable persona modeling for different user types, roles,
    and interaction contexts within the knowledge graph system.
    """
    
    def __init__(
        self,
        persona_id: str,
        name: str,
        profile: PersonaProfile,
        characteristics: Optional[PersonaCharacteristics] = None,
        description: Optional[str] = None
    ):
        """
        Initialize Persona Digital Twin.
        
        Args:
            persona_id: Unique persona identifier
            name: Persona name
            profile: Detailed persona profile
            characteristics: Persona behavioral characteristics
            description: Persona description
        """
        self.profile = profile
        
        # Generate description if not provided
        if not description:
            description = f"{profile.persona_name} - {profile.role} with {profile.technical_level} technical expertise"
        
        # Initialize with persona characteristics
        persona_chars = characteristics or PersonaCharacteristics()
        
        super().__init__(
            twin_id=persona_id,
            twin_type="persona",
            name=name,
            description=description,
            characteristics=persona_chars
        )
        
        # Persona-specific state
        self.context_history: List[Dict[str, Any]] = []
        self.persona_adaptations: List[Dict[str, Any]] = []
        self.interaction_patterns: Dict[str, Any] = {}
        
        # Performance tracking
        self.consistency_score = 1.0
        self.context_adaptation_count = 0
        self.successful_adaptations = 0
        
        self.logger.info(
            "Persona twin initialized",
            persona_id=persona_id,
            role=profile.role,
            communication_style=profile.communication_style
        )
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process query through persona lens.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Persona-specific response with role-based perspective
        """
        try:
            # Analyze query for persona relevance
            query_analysis = self._analyze_query_context(query, context)
            
            # Adapt persona characteristics if needed
            await self._adapt_to_context(query_analysis, context)
            
            # Generate persona-specific response
            response_content = await self._generate_persona_response(query, query_analysis, context)
            
            # Apply persona communication style
            styled_response = self._apply_communication_style(response_content, query_analysis)
            
            # Calculate confidence based on persona alignment
            confidence = self._calculate_persona_confidence(query, query_analysis)
            
            # Store interaction pattern
            self._update_interaction_patterns(query, query_analysis, styled_response)
            
            return {
                "response": styled_response,
                "confidence": confidence,
                "metadata": {
                    "persona_name": self.profile.persona_name,
                    "role": self.profile.role,
                    "communication_style": self.profile.communication_style,
                    "technical_level": self.profile.technical_level,
                    "context_adaptation": query_analysis.get("adaptation_applied", False),
                    "persona_perspective": True
                }
            }
            
        except Exception as e:
            self.logger.error(
                "Persona query processing failed",
                persona_id=self.twin_id,
                role=self.profile.role,
                error=str(e)
            )
            raise PersonaTwinError(f"Persona processing failed: {e}", self.profile.persona_name)
    
    async def adapt_to_role_context(
        self,
        role_context: Dict[str, Any],
        temporary: bool = False
    ) -> Dict[str, Any]:
        """
        Adapt persona to specific role context.
        
        Args:
            role_context: Context requiring role adaptation
            temporary: Whether adaptation is temporary
            
        Returns:
            Adaptation result and configuration
        """
        try:
            adaptation_id = f"adapt_{self.twin_id}_{int(datetime.utcnow().timestamp())}"
            
            # Analyze required adaptations
            required_adaptations = self._analyze_role_requirements(role_context)
            
            # Apply characteristic adjustments
            original_characteristics = self.characteristics.dict().copy()
            adaptations_applied = {}
            
            # Adjust communication style
            if "formal_communication" in role_context:
                if role_context["formal_communication"]:
                    self.characteristics.formality_level = min(1.0, self.characteristics.formality_level + 0.2)
                    adaptations_applied["formality_increased"] = True
            
            # Adjust technical depth
            if "technical_complexity" in role_context:
                complexity = role_context["technical_complexity"]
                if complexity == "high":
                    self.characteristics.technical_depth = min(1.0, self.characteristics.technical_depth + 0.3)
                elif complexity == "low":
                    self.characteristics.technical_depth = max(0.3, self.characteristics.technical_depth - 0.2)
                adaptations_applied["technical_depth_adjusted"] = complexity
            
            # Adjust detail preference
            if "detail_requirement" in role_context:
                detail_req = role_context["detail_requirement"]
                if detail_req == "high":
                    self.characteristics.detail_preference = min(1.0, self.characteristics.detail_preference + 0.2)
                elif detail_req == "low":
                    self.characteristics.detail_preference = max(0.3, self.characteristics.detail_preference - 0.2)
                adaptations_applied["detail_preference_adjusted"] = detail_req
            
            # Create adaptation record
            adaptation_record = {
                "adaptation_id": adaptation_id,
                "timestamp": datetime.utcnow().isoformat(),
                "role_context": role_context,
                "original_characteristics": original_characteristics,
                "adaptations_applied": adaptations_applied,
                "temporary": temporary,
                "success": True
            }
            
            # Store adaptation
            self.persona_adaptations.append(adaptation_record)
            self.context_adaptation_count += 1
            self.successful_adaptations += 1
            
            # Update consistency score
            self._update_consistency_score(adaptations_applied)
            
            self.logger.info(
                "Persona adaptation completed",
                persona_id=self.twin_id,
                adaptation_id=adaptation_id,
                adaptations_count=len(adaptations_applied)
            )
            
            return {
                "adaptation_id": adaptation_id,
                "adaptations_applied": adaptations_applied,
                "consistency_score": self.consistency_score,
                "temporary": temporary,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(
                "Persona adaptation failed",
                persona_id=self.twin_id,
                error=str(e)
            )
            raise PersonaTwinError(f"Persona adaptation failed: {e}", self.twin_id)
    
    async def revert_adaptations(
        self,
        adaptation_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Revert persona adaptations.
        
        Args:
            adaptation_ids: Specific adaptations to revert, or None for all temporary
            
        Returns:
            Reversion result
        """
        try:
            reverted_adaptations = []
            
            # Determine which adaptations to revert
            if adaptation_ids:
                adaptations_to_revert = [
                    adapt for adapt in self.persona_adaptations
                    if adapt["adaptation_id"] in adaptation_ids
                ]
            else:
                # Revert all temporary adaptations
                adaptations_to_revert = [
                    adapt for adapt in self.persona_adaptations
                    if adapt.get("temporary", False)
                ]
            
            # Revert adaptations in reverse chronological order
            adaptations_to_revert.sort(key=lambda x: x["timestamp"], reverse=True)
            
            for adaptation in adaptations_to_revert:
                # Restore original characteristics
                original_chars = adaptation["original_characteristics"]
                for key, value in original_chars.items():
                    if hasattr(self.characteristics, key):
                        setattr(self.characteristics, key, value)
                
                reverted_adaptations.append(adaptation["adaptation_id"])
                
                # Remove from adaptations list
                self.persona_adaptations = [
                    adapt for adapt in self.persona_adaptations
                    if adapt["adaptation_id"] != adaptation["adaptation_id"]
                ]
            
            # Recalculate consistency score
            self.consistency_score = self._calculate_consistency_score()
            
            self.logger.info(
                "Persona adaptations reverted",
                persona_id=self.twin_id,
                reverted_count=len(reverted_adaptations)
            )
            
            return {
                "reverted_adaptations": reverted_adaptations,
                "reverted_count": len(reverted_adaptations),
                "consistency_score": self.consistency_score,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(
                "Persona reversion failed",
                persona_id=self.twin_id,
                error=str(e)
            )
            raise PersonaTwinError(f"Persona reversion failed: {e}", self.twin_id)
    
    def get_persona_prompt(self) -> str:
        """Generate comprehensive persona prompt."""
        expertise_text = ", ".join(self.profile.areas_of_expertise) if self.profile.areas_of_expertise else "general knowledge"
        objectives_text = ", ".join(self.profile.primary_objectives) if self.profile.primary_objectives else "user assistance"
        
        prompt = f"""You are {self.profile.persona_name}, a {self.profile.role}.

## Persona Profile
- **Role**: {self.profile.role}
- **Background**: {self.profile.background}
- **Technical Level**: {self.profile.technical_level}
- **Areas of Expertise**: {expertise_text}
- **Primary Objectives**: {objectives_text}

## Communication Style
- **Style**: {self.profile.communication_style}
- **Detail Level**: {self.profile.preferred_detail_level}
- **Decision Making**: {self.profile.decision_making_style}
- **Collaboration**: {self.profile.collaboration_preference}

## Behavioral Characteristics
- **Formality Level**: {self.characteristics.formality_level:.1%} - Professional communication level
- **Technical Depth**: {self.characteristics.technical_depth:.1%} - Technical detail in responses
- **Detail Preference**: {self.characteristics.detail_preference:.1%} - Comprehensive vs. concise responses
- **Empathy Level**: {self.characteristics.empathy_level:.1%} - Understanding and supportive tone
- **Proactiveness**: {self.characteristics.proactiveness:.1%} - Initiative in providing additional value

## Professional Traits
- **Authenticity**: {self.characteristics.authenticity_level:.1%} - Maintain genuine persona consistency
- **Adaptability**: {self.characteristics.adaptability:.1%} - Adjust to different contexts appropriately
- **Supportiveness**: {self.characteristics.supportiveness:.1%} - Helpful and encouraging approach
- **Question Asking**: {self.characteristics.question_asking:.1%} - Clarify requirements when needed

## Key Concerns and Priorities
"""
        
        if self.profile.key_concerns:
            for concern in self.profile.key_concerns:
                prompt += f"- {concern}\n"
        else:
            prompt += "- User satisfaction and goal achievement\n"
        
        prompt += f"""
## Response Guidelines
1. **Role Consistency**: Always respond from the perspective of {self.profile.role}
2. **Communication Style**: Maintain {self.profile.communication_style} communication throughout
3. **Technical Appropriateness**: Provide {self.profile.technical_level}-level technical detail
4. **Value Addition**: Leverage your expertise in {expertise_text}
5. **Goal Alignment**: Support objectives: {objectives_text}

## Success Metrics
"""
        
        if self.profile.success_metrics:
            for metric in self.profile.success_metrics:
                prompt += f"- {metric}\n"
        else:
            prompt += "- Clear, actionable responses that meet user needs\n"
        
        prompt += f"""
When responding, embody {self.profile.persona_name} completely while providing helpful, role-appropriate guidance that reflects your expertise and communication style."""
        
        return prompt
    
    def _analyze_query_context(self, query: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze query for persona relevance and context requirements."""
        query_lower = query.lower()
        
        analysis = {
            "query_type": "general",
            "technical_complexity": "medium",
            "formality_required": False,
            "expertise_area": None,
            "adaptation_needed": False,
            "confidence_factors": []
        }
        
        # Analyze technical complexity
        technical_indicators = ["implement", "configure", "architecture", "design", "algorithm", "optimize"]
        tech_matches = sum(1 for indicator in technical_indicators if indicator in query_lower)
        
        if tech_matches >= 3:
            analysis["technical_complexity"] = "high"
        elif tech_matches >= 1:
            analysis["technical_complexity"] = "medium"
        else:
            analysis["technical_complexity"] = "low"
        
        # Check formality requirements
        formal_indicators = ["request", "recommend", "proposal", "assessment", "evaluation"]
        if any(indicator in query_lower for indicator in formal_indicators):
            analysis["formality_required"] = True
        
        # Match against persona expertise
        for area in self.profile.areas_of_expertise:
            if area.lower() in query_lower:
                analysis["expertise_area"] = area
                analysis["confidence_factors"].append(f"expertise_match_{area}")
        
        # Check if context adaptation is needed
        if context:
            if context.get("audience") != self.profile.technical_level:
                analysis["adaptation_needed"] = True
            if context.get("formality") != self.profile.communication_style:
                analysis["adaptation_needed"] = True
        
        return analysis
    
    async def _adapt_to_context(
        self,
        query_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> None:
        """Adapt persona characteristics based on query analysis."""
        if not query_analysis.get("adaptation_needed", False):
            return
        
        # Create temporary adaptation context
        adaptation_context = {}
        
        # Technical complexity adaptation
        if query_analysis["technical_complexity"] != "medium":
            adaptation_context["technical_complexity"] = query_analysis["technical_complexity"]
        
        # Formality adaptation
        if query_analysis["formality_required"] and self.profile.communication_style != "formal":
            adaptation_context["formal_communication"] = True
        
        # Apply context-specific adaptations
        if context:
            if "detail_requirement" in context:
                adaptation_context["detail_requirement"] = context["detail_requirement"]
        
        # Apply temporary adaptation
        if adaptation_context:
            await self.adapt_to_role_context(adaptation_context, temporary=True)
            query_analysis["adaptation_applied"] = True
    
    async def _generate_persona_response(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate persona-specific response content."""
        # Base response structure
        response_parts = []
        
        # Role-based introduction if appropriate
        if self.characteristics.professional_tone > 0.7:
            if query_analysis["expertise_area"]:
                response_parts.append(f"As a {self.profile.role} with expertise in {query_analysis['expertise_area']},")
            else:
                response_parts.append(f"From my perspective as a {self.profile.role},")
        
        # Main response content based on role and expertise
        main_content = self._generate_role_specific_content(query, query_analysis)
        response_parts.append(main_content)
        
        # Add examples if characteristic indicates
        if self.characteristics.example_usage > 0.7:
            examples = self._generate_relevant_examples(query, query_analysis)
            if examples:
                response_parts.append(f"\n\n**Examples:**\n{examples}")
        
        # Add proactive suggestions if characteristic indicates
        if self.characteristics.proactiveness > 0.7:
            suggestions = self._generate_proactive_suggestions(query, query_analysis)
            if suggestions:
                response_parts.append(f"\n\n**Additional Considerations:**\n{suggestions}")
        
        # Add questions for clarification if appropriate
        if self.characteristics.question_asking > 0.6:
            questions = self._generate_clarifying_questions(query, query_analysis)
            if questions:
                response_parts.append(f"\n\n**Questions for Better Assistance:**\n{questions}")
        
        return "\n".join(filter(None, response_parts))
    
    def _generate_role_specific_content(
        self,
        query: str,
        query_analysis: Dict[str, Any]
    ) -> str:
        """Generate content specific to the persona's role."""
        # Role-specific response patterns
        role_responses = {
            "system administrator": self._generate_admin_response(query, query_analysis),
            "security analyst": self._generate_security_response(query, query_analysis),
            "project manager": self._generate_pm_response(query, query_analysis),
            "developer": self._generate_developer_response(query, query_analysis),
            "architect": self._generate_architect_response(query, query_analysis)
        }
        
        role_lower = self.profile.role.lower()
        
        # Find matching role or use generic response
        for role_key, response_func in role_responses.items():
            if role_key in role_lower:
                return response_func
        
        # Generic professional response
        return self._generate_generic_professional_response(query, query_analysis)
    
    def _generate_admin_response(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate system administrator perspective response."""
        return f"""**System Administration Perspective:**

Based on operational requirements and system management best practices, here's my analysis:

- **Operational Impact**: Consider how this affects system availability and user experience
- **Security Implications**: Evaluate potential security risks and mitigation strategies
- **Resource Requirements**: Assess computational, storage, and network resource needs
- **Maintenance Considerations**: Plan for ongoing maintenance and monitoring requirements
- **Scalability Factors**: Ensure solution can scale with organizational growth

**Recommended Approach:**
1. Conduct thorough testing in development environment
2. Implement monitoring and alerting for key metrics
3. Plan rollback procedures for risk mitigation
4. Document configuration changes for knowledge management"""
    
    def _generate_security_response(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate security analyst perspective response."""
        return f"""**Security Analysis Perspective:**

From a security standpoint, this requires careful evaluation of potential risks and controls:

- **Threat Assessment**: Identify potential attack vectors and threat actors
- **Risk Evaluation**: Analyze impact and likelihood of security incidents
- **Control Requirements**: Implement appropriate security controls and safeguards
- **Compliance Considerations**: Ensure alignment with security policies and regulations
- **Monitoring Strategy**: Establish detection and response capabilities

**Security Recommendations:**
1. Apply principle of least privilege
2. Implement defense-in-depth strategies
3. Conduct security testing and validation
4. Establish incident response procedures"""
    
    def _generate_pm_response(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate project manager perspective response."""
        return f"""**Project Management Perspective:**

From a project delivery standpoint, here's my assessment:

- **Scope Definition**: Clearly define deliverables and success criteria
- **Resource Planning**: Identify required skills, time, and budget
- **Risk Management**: Assess project risks and mitigation strategies
- **Timeline Considerations**: Establish realistic milestones and dependencies
- **Stakeholder Impact**: Consider effects on all project stakeholders

**Project Approach:**
1. Define clear project charter and objectives
2. Establish communication and reporting protocols
3. Implement change management processes
4. Plan for quality assurance and testing phases"""
    
    def _generate_developer_response(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate developer perspective response."""
        return f"""**Development Perspective:**

From a technical implementation standpoint:

- **Technical Architecture**: Consider design patterns and architectural principles
- **Code Quality**: Ensure maintainable, testable, and readable code
- **Performance Optimization**: Address efficiency and scalability requirements
- **Integration Considerations**: Plan for system and API integrations
- **Testing Strategy**: Implement comprehensive testing approach

**Development Recommendations:**
1. Follow established coding standards and best practices
2. Implement proper error handling and logging
3. Design for modularity and reusability
4. Plan for comprehensive testing coverage"""
    
    def _generate_architect_response(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate architect perspective response."""
        return f"""**Architectural Perspective:**

From a system architecture standpoint:

- **System Design**: Evaluate overall system structure and component relationships
- **Technology Selection**: Choose appropriate technologies and frameworks
- **Scalability Architecture**: Design for current and future scale requirements
- **Integration Patterns**: Plan for system connectivity and data flow
- **Quality Attributes**: Address performance, security, and reliability requirements

**Architectural Recommendations:**
1. Apply established architectural patterns and principles
2. Design for loose coupling and high cohesion
3. Plan for horizontal and vertical scaling
4. Implement appropriate abstraction layers"""
    
    def _generate_generic_professional_response(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate generic professional response."""
        return f"""**Professional Analysis:**

Based on my expertise and experience, here's my perspective:

- **Key Considerations**: Identify critical factors that influence success
- **Best Practices**: Apply industry-standard approaches and methodologies
- **Risk Assessment**: Evaluate potential challenges and mitigation strategies
- **Implementation Planning**: Develop structured approach to execution
- **Success Metrics**: Define measurable outcomes and evaluation criteria

**Recommended Next Steps:**
1. Gather additional requirements and context
2. Develop detailed implementation plan
3. Identify required resources and dependencies
4. Establish success criteria and monitoring approach"""
    
    def _generate_relevant_examples(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate relevant examples based on query and persona."""
        examples = []
        
        # Add role-specific examples
        if "security" in self.profile.role.lower():
            examples.append("- Implementation of multi-factor authentication")
            examples.append("- Security incident response procedures")
        elif "admin" in self.profile.role.lower():
            examples.append("- System monitoring dashboard configuration")
            examples.append("- Automated backup and recovery procedures")
        elif "developer" in self.profile.role.lower():
            examples.append("- Code review and testing workflows")
            examples.append("- Continuous integration pipeline setup")
        
        return "\n".join(examples) if examples else ""
    
    def _generate_proactive_suggestions(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate proactive suggestions based on persona characteristics."""
        suggestions = []
        
        # Add suggestions based on role expertise
        if self.profile.areas_of_expertise:
            for area in self.profile.areas_of_expertise[:2]:  # Top 2 areas
                suggestions.append(f"- Consider leveraging {area} expertise for enhanced outcomes")
        
        # Add general professional suggestions
        suggestions.extend([
            "- Document decisions and rationale for future reference",
            "- Plan for regular review and optimization cycles",
            "- Consider stakeholder communication and change management"
        ])
        
        return "\n".join(suggestions)
    
    def _generate_clarifying_questions(self, query: str, query_analysis: Dict[str, Any]) -> str:
        """Generate clarifying questions based on persona and query."""
        questions = []
        
        # Role-specific questions
        if "admin" in self.profile.role.lower():
            questions.append("- What are the current system specifications and constraints?")
            questions.append("- Are there any compliance or regulatory requirements?")
        elif "security" in self.profile.role.lower():
            questions.append("- What is the current security posture and risk tolerance?")
            questions.append("- Are there specific compliance frameworks to consider?")
        elif "project" in self.profile.role.lower():
            questions.append("- What are the timeline and budget constraints?")
            questions.append("- Who are the key stakeholders and decision makers?")
        
        return "\n".join(questions) if questions else ""
    
    def _apply_communication_style(self, content: str, query_analysis: Dict[str, Any]) -> str:
        """Apply persona communication style to response content."""
        # Adjust based on communication style
        if self.profile.communication_style == "formal":
            if not content.startswith("**"):
                content = f"**Professional Response:**\n\n{content}"
        elif self.profile.communication_style == "casual":
            # Make content more conversational
            content = content.replace("**", "").replace(":", " -")
        
        # Add supportive closing if characteristic indicates
        if self.characteristics.supportiveness > 0.8:
            content += f"\n\nI'm here to provide additional assistance as needed. Please let me know if you'd like me to elaborate on any aspect or if you have specific follow-up questions."
        
        return content
    
    def _calculate_persona_confidence(self, query: str, query_analysis: Dict[str, Any]) -> float:
        """Calculate confidence in persona response."""
        base_confidence = 0.7
        
        # Boost for expertise area match
        if query_analysis.get("expertise_area"):
            base_confidence += 0.15
        
        # Boost for role relevance
        role_keywords = self.profile.role.lower().split()
        query_lower = query.lower()
        role_matches = sum(1 for keyword in role_keywords if keyword in query_lower)
        base_confidence += role_matches * 0.05
        
        # Adjust for consistency score
        base_confidence *= self.consistency_score
        
        # Adjust for adaptation success
        if query_analysis.get("adaptation_applied") and self.successful_adaptations > 0:
            adaptation_success_rate = self.successful_adaptations / max(self.context_adaptation_count, 1)
            base_confidence *= adaptation_success_rate
        
        return min(base_confidence, 1.0)
    
    def _update_interaction_patterns(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        response: str
    ) -> None:
        """Update interaction patterns for persona learning."""
        pattern_key = query_analysis.get("query_type", "general")
        
        if pattern_key not in self.interaction_patterns:
            self.interaction_patterns[pattern_key] = {
                "count": 0,
                "average_confidence": 0.0,
                "common_adaptations": [],
                "success_indicators": []
            }
        
        pattern = self.interaction_patterns[pattern_key]
        pattern["count"] += 1
        
        # Track adaptations
        if query_analysis.get("adaptation_applied"):
            adaptation_type = f"{query_analysis.get('technical_complexity', 'medium')}_complexity"
            if adaptation_type not in pattern["common_adaptations"]:
                pattern["common_adaptations"].append(adaptation_type)
    
    def _analyze_role_requirements(self, role_context: Dict[str, Any]) -> List[str]:
        """Analyze what adaptations are required for role context."""
        requirements = []
        
        if role_context.get("formal_communication"):
            requirements.append("increase_formality")
        
        if role_context.get("technical_complexity") == "high":
            requirements.append("increase_technical_depth")
        elif role_context.get("technical_complexity") == "low":
            requirements.append("decrease_technical_depth")
        
        if role_context.get("detail_requirement") == "high":
            requirements.append("increase_detail")
        elif role_context.get("detail_requirement") == "low":
            requirements.append("decrease_detail")
        
        return requirements
    
    def _update_consistency_score(self, adaptations_applied: Dict[str, Any]) -> None:
        """Update consistency score based on adaptations."""
        # Reduce consistency score slightly for each adaptation
        adaptation_impact = len(adaptations_applied) * 0.02
        self.consistency_score = max(0.5, self.consistency_score - adaptation_impact)
        
        # Allow consistency to recover over time
        if self.interaction_count > 0:
            recovery_rate = 0.01 / max(self.interaction_count, 1)
            self.consistency_score = min(1.0, self.consistency_score + recovery_rate)
    
    def _calculate_consistency_score(self) -> float:
        """Calculate overall persona consistency score."""
        if not self.persona_adaptations:
            return 1.0
        
        # Base score starts high
        base_score = 1.0
        
        # Reduce for each adaptation
        adaptation_count = len(self.persona_adaptations)
        adaptation_penalty = adaptation_count * 0.05
        
        # Reduce more for permanent adaptations
        permanent_adaptations = sum(1 for adapt in self.persona_adaptations if not adapt.get("temporary", False))
        permanent_penalty = permanent_adaptations * 0.1
        
        consistency_score = base_score - adaptation_penalty - permanent_penalty
        
        return max(0.3, consistency_score)  # Minimum consistency threshold
    
    def get_persona_statistics(self) -> Dict[str, Any]:
        """Get comprehensive persona statistics."""
        return {
            "persona_id": self.twin_id,
            "persona_name": self.profile.persona_name,
            "role": self.profile.role,
            "communication_style": self.profile.communication_style,
            "technical_level": self.profile.technical_level,
            "total_interactions": self.interaction_count,
            "consistency_score": self.consistency_score,
            "adaptations_applied": len(self.persona_adaptations),
            "temporary_adaptations": sum(1 for adapt in self.persona_adaptations if adapt.get("temporary", False)),
            "adaptation_success_rate": (
                self.successful_adaptations / max(self.context_adaptation_count, 1) * 100
            ) if self.context_adaptation_count > 0 else 100,
            "interaction_patterns": {
                pattern: {
                    "count": data["count"],
                    "common_adaptations": data["common_adaptations"]
                }
                for pattern, data in self.interaction_patterns.items()
            },
            "areas_of_expertise": self.profile.areas_of_expertise,
            "primary_objectives": self.profile.primary_objectives,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None
        }