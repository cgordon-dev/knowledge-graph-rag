"""
Base Digital Twin implementation for Knowledge Graph-RAG system.

Provides common functionality and patterns for all digital twins with
behavioral modeling, learning capabilities, and consistency validation.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field
import structlog

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import PersonaTwinError, PersonaValidationError
from kg_rag.core.logger import get_performance_logger


class TwinInteraction(BaseModel):
    """Represents an interaction with a digital twin."""
    
    interaction_id: str = Field(..., description="Unique interaction identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Interaction timestamp")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Twin response")
    context: Dict[str, Any] = Field(default_factory=dict, description="Interaction context")
    satisfaction_score: Optional[float] = Field(None, description="User satisfaction score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TwinCharacteristics(BaseModel):
    """Behavioral characteristics of a digital twin."""
    
    # Core personality traits
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk tolerance level")
    detail_preference: float = Field(default=0.7, ge=0.0, le=1.0, description="Preference for detailed responses")
    technical_depth: float = Field(default=0.6, ge=0.0, le=1.0, description="Technical communication level")
    response_speed: float = Field(default=0.8, ge=0.0, le=1.0, description="Preference for quick responses")
    
    # Communication style
    formality_level: float = Field(default=0.6, ge=0.0, le=1.0, description="Communication formality")
    empathy_level: float = Field(default=0.7, ge=0.0, le=1.0, description="Empathetic response level")
    
    # Domain-specific traits
    domain_expertise: Dict[str, float] = Field(default_factory=dict, description="Expertise levels by domain")
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Rate of characteristic adaptation")
    
    # Behavioral patterns
    common_phrases: List[str] = Field(default_factory=list, description="Commonly used phrases")
    decision_patterns: Dict[str, Any] = Field(default_factory=dict, description="Decision-making patterns")
    interaction_preferences: Dict[str, Any] = Field(default_factory=dict, description="Interaction preferences")


class TwinMemory(BaseModel):
    """Memory system for digital twins."""
    
    short_term_memory: List[TwinInteraction] = Field(default_factory=list, description="Recent interactions")
    long_term_patterns: Dict[str, Any] = Field(default_factory=dict, description="Learned behavioral patterns")
    adaptation_history: List[Dict[str, Any]] = Field(default_factory=list, description="Characteristic changes over time")
    context_cache: Dict[str, Any] = Field(default_factory=dict, description="Cached context information")
    
    def add_interaction(self, interaction: TwinInteraction) -> None:
        """Add interaction to short-term memory."""
        self.short_term_memory.append(interaction)
        
        # Keep only recent interactions (last 100)
        if len(self.short_term_memory) > 100:
            self.short_term_memory = self.short_term_memory[-100:]
    
    def get_recent_context(self, limit: int = 5) -> List[TwinInteraction]:
        """Get recent interactions for context."""
        return self.short_term_memory[-limit:] if self.short_term_memory else []


class BaseTwin(ABC):
    """Base class for all digital twins."""
    
    def __init__(
        self,
        twin_id: str,
        twin_type: str,
        name: str,
        description: str,
        characteristics: Optional[TwinCharacteristics] = None
    ):
        """
        Initialize base digital twin.
        
        Args:
            twin_id: Unique twin identifier
            twin_type: Type of twin (expert, user_journey, process, etc.)
            name: Human-readable twin name
            description: Twin description
            characteristics: Behavioral characteristics
        """
        self.twin_id = twin_id
        self.twin_type = twin_type
        self.name = name
        self.description = description
        self.characteristics = characteristics or TwinCharacteristics()
        
        # Core components
        self.memory = TwinMemory()
        self.settings = get_settings()
        self.logger = structlog.get_logger(f"twin.{twin_type}.{twin_id}")
        self.performance_logger = get_performance_logger()
        
        # State tracking
        self.created_at = datetime.utcnow()
        self.last_interaction = None
        self.interaction_count = 0
        self.confidence_scores = []
        
        # Vector representations for similarity matching
        self.behavior_vector: Optional[np.ndarray] = None
        self.expertise_vector: Optional[np.ndarray] = None
        
        # Learning and adaptation
        self.adaptation_threshold = self.settings.personas.persona_adaptation_threshold
        self.learning_enabled = True
    
    @abstractmethod
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query and generate response.
        
        Args:
            query: Input query
            context: Additional context
            
        Returns:
            Response dictionary with answer and metadata
        """
        pass
    
    @abstractmethod
    def get_persona_prompt(self) -> str:
        """
        Generate persona prompt for AI models.
        
        Returns:
            Formatted persona prompt
        """
        pass
    
    async def interact(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main interaction method for the twin.
        
        Args:
            query: User query
            context: Interaction context
            user_id: User identifier
            
        Returns:
            Interaction result with response and metadata
        """
        start_time = datetime.utcnow()
        interaction_id = f"{self.twin_id}_{int(start_time.timestamp())}"
        
        try:
            # Log interaction start
            self.logger.info(
                "Twin interaction started",
                twin_id=self.twin_id,
                interaction_id=interaction_id,
                query_length=len(query)
            )
            
            # Process query through twin-specific logic
            response_data = await self.process_query(query, context)
            
            # Create interaction record
            interaction = TwinInteraction(
                interaction_id=interaction_id,
                query=query,
                response=response_data.get("response", ""),
                context=context or {},
                metadata={
                    "twin_id": self.twin_id,
                    "twin_type": self.twin_type,
                    "user_id": user_id,
                    "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                }
            )
            
            # Store interaction in memory
            self.memory.add_interaction(interaction)
            
            # Update state
            self.last_interaction = start_time
            self.interaction_count += 1
            
            # Log performance
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.performance_logger.log_query_performance(
                query_type=f"twin_{self.twin_type}_interaction",
                duration_ms=execution_time,
                result_count=1,
                user_id=user_id,
                persona_id=self.twin_id
            )
            
            # Return complete response
            result = {
                "interaction_id": interaction_id,
                "response": response_data.get("response", ""),
                "confidence": response_data.get("confidence", 0.0),
                "twin_metadata": {
                    "twin_id": self.twin_id,
                    "twin_type": self.twin_type,
                    "twin_name": self.name,
                    "interaction_count": self.interaction_count
                },
                "processing_metadata": response_data.get("metadata", {}),
                "adaptation_summary": self._get_adaptation_summary()
            }
            
            # Trigger learning if enabled
            if self.learning_enabled:
                await self._update_learning_patterns(interaction, response_data)
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Twin interaction failed",
                twin_id=self.twin_id,
                interaction_id=interaction_id,
                error=str(e)
            )
            
            return {
                "interaction_id": interaction_id,
                "error": str(e),
                "twin_metadata": {
                    "twin_id": self.twin_id,
                    "twin_type": self.twin_type,
                    "twin_name": self.name
                }
            }
    
    async def provide_feedback(
        self,
        interaction_id: str,
        satisfaction_score: float,
        feedback_text: Optional[str] = None
    ) -> None:
        """
        Provide feedback on a previous interaction.
        
        Args:
            interaction_id: Interaction to provide feedback on
            satisfaction_score: Satisfaction score (0.0 - 1.0)
            feedback_text: Optional textual feedback
        """
        # Find interaction in memory
        interaction = None
        for stored_interaction in self.memory.short_term_memory:
            if stored_interaction.interaction_id == interaction_id:
                interaction = stored_interaction
                break
        
        if not interaction:
            self.logger.warning(
                "Feedback for unknown interaction",
                twin_id=self.twin_id,
                interaction_id=interaction_id
            )
            return
        
        # Update interaction with feedback
        interaction.satisfaction_score = satisfaction_score
        if feedback_text:
            interaction.metadata["feedback_text"] = feedback_text
        
        # Store satisfaction score for analytics
        self.confidence_scores.append(satisfaction_score)
        
        # Trigger adaptation if needed
        if self.learning_enabled and satisfaction_score < self.adaptation_threshold:
            await self._adapt_from_feedback(interaction, satisfaction_score, feedback_text)
        
        self.logger.info(
            "Feedback processed",
            twin_id=self.twin_id,
            interaction_id=interaction_id,
            satisfaction_score=satisfaction_score
        )
    
    async def _update_learning_patterns(
        self,
        interaction: TwinInteraction,
        response_data: Dict[str, Any]
    ) -> None:
        """Update learning patterns based on interaction."""
        try:
            # Extract patterns from interaction
            query_type = self._classify_query_type(interaction.query)
            response_style = self._analyze_response_style(response_data.get("response", ""))
            
            # Update long-term patterns
            if query_type not in self.memory.long_term_patterns:
                self.memory.long_term_patterns[query_type] = {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "response_styles": []
                }
            
            pattern = self.memory.long_term_patterns[query_type]
            pattern["count"] += 1
            
            confidence = response_data.get("confidence", 0.0)
            pattern["avg_confidence"] = (
                (pattern["avg_confidence"] * (pattern["count"] - 1) + confidence) / pattern["count"]
            )
            pattern["response_styles"].append(response_style)
            
            # Keep only recent styles
            if len(pattern["response_styles"]) > 20:
                pattern["response_styles"] = pattern["response_styles"][-20:]
            
        except Exception as e:
            self.logger.warning("Learning pattern update failed", error=str(e))
    
    async def _adapt_from_feedback(
        self,
        interaction: TwinInteraction,
        satisfaction_score: float,
        feedback_text: Optional[str] = None
    ) -> None:
        """Adapt twin characteristics based on feedback."""
        try:
            adaptation_changes = {}
            
            # Adjust characteristics based on satisfaction score
            if satisfaction_score < 0.3:  # Very unsatisfied
                # Reduce confidence in current approach
                if hasattr(self.characteristics, 'response_speed'):
                    self.characteristics.response_speed *= 0.95
                    adaptation_changes["response_speed"] = "decreased"
                
                if hasattr(self.characteristics, 'technical_depth'):
                    # Adjust technical depth based on query complexity
                    query_complexity = self._assess_query_complexity(interaction.query)
                    if query_complexity > 0.7:
                        self.characteristics.technical_depth *= 1.05
                        adaptation_changes["technical_depth"] = "increased"
                    else:
                        self.characteristics.technical_depth *= 0.95
                        adaptation_changes["technical_depth"] = "decreased"
            
            elif satisfaction_score > 0.8:  # Very satisfied
                # Reinforce current approach
                learning_rate = self.characteristics.learning_rate
                
                # Slightly increase confidence in successful patterns
                if hasattr(self.characteristics, 'detail_preference'):
                    response_length = len(interaction.response)
                    if response_length > 500:  # Detailed response was appreciated
                        self.characteristics.detail_preference += learning_rate * 0.1
                        adaptation_changes["detail_preference"] = "increased"
            
            # Record adaptation
            if adaptation_changes:
                self.memory.adaptation_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "trigger": "feedback",
                    "satisfaction_score": satisfaction_score,
                    "changes": adaptation_changes,
                    "feedback_text": feedback_text
                })
                
                self.logger.info(
                    "Twin characteristics adapted",
                    twin_id=self.twin_id,
                    changes=adaptation_changes,
                    satisfaction_score=satisfaction_score
                )
            
        except Exception as e:
            self.logger.warning("Adaptation from feedback failed", error=str(e))
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for pattern learning."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "define", "explain"]):
            return "explanation"
        elif any(word in query_lower for word in ["how", "guide", "steps"]):
            return "instruction"
        elif any(word in query_lower for word in ["why", "reason", "because"]):
            return "reasoning"
        elif "?" in query:
            return "question"
        else:
            return "statement"
    
    def _analyze_response_style(self, response: str) -> Dict[str, Any]:
        """Analyze response style characteristics."""
        return {
            "length": len(response),
            "sentence_count": response.count('.') + response.count('!') + response.count('?'),
            "formal_words": sum(1 for word in response.split() if len(word) > 6),
            "question_count": response.count('?'),
            "exclamation_count": response.count('!')
        }
    
    def _assess_query_complexity(self, query: str) -> float:
        """Assess query complexity (0.0 - 1.0)."""
        complexity_indicators = [
            len(query.split()) > 20,  # Long query
            any(word in query.lower() for word in ["complex", "detailed", "comprehensive"]),
            query.count('?') > 1,  # Multiple questions
            any(word in query.lower() for word in ["analyze", "compare", "evaluate"]),
        ]
        
        return sum(complexity_indicators) / len(complexity_indicators)
    
    def _get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of twin adaptation."""
        return {
            "interaction_count": self.interaction_count,
            "average_confidence": np.mean(self.confidence_scores) if self.confidence_scores else 0.0,
            "adaptation_count": len(self.memory.adaptation_history),
            "last_adaptation": (
                self.memory.adaptation_history[-1]["timestamp"] 
                if self.memory.adaptation_history else None
            ),
            "learning_enabled": self.learning_enabled
        }
    
    def get_twin_state(self) -> Dict[str, Any]:
        """Get complete twin state for serialization."""
        return {
            "twin_id": self.twin_id,
            "twin_type": self.twin_type,
            "name": self.name,
            "description": self.description,
            "characteristics": self.characteristics.dict(),
            "created_at": self.created_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "interaction_count": self.interaction_count,
            "adaptation_summary": self._get_adaptation_summary(),
            "memory_summary": {
                "short_term_count": len(self.memory.short_term_memory),
                "long_term_patterns": list(self.memory.long_term_patterns.keys()),
                "adaptation_history_count": len(self.memory.adaptation_history)
            }
        }
    
    async def validate_consistency(self) -> Dict[str, Any]:
        """
        Validate twin consistency and behavior.
        
        Returns:
            Validation results with consistency score
        """
        try:
            validation_results = {
                "consistency_score": 0.0,
                "issues": [],
                "recommendations": []
            }
            
            # Check characteristic consistency
            char_consistency = self._validate_characteristics()
            validation_results["characteristics"] = char_consistency
            
            # Check memory integrity
            memory_integrity = self._validate_memory()
            validation_results["memory"] = memory_integrity
            
            # Check behavioral patterns
            behavior_consistency = self._validate_behavior_patterns()
            validation_results["behavior"] = behavior_consistency
            
            # Calculate overall consistency score
            scores = [
                char_consistency.get("score", 0.0),
                memory_integrity.get("score", 0.0),
                behavior_consistency.get("score", 0.0)
            ]
            validation_results["consistency_score"] = np.mean(scores)
            
            return validation_results
            
        except Exception as e:
            raise PersonaValidationError(
                f"Consistency validation failed: {e}",
                persona_id=self.twin_id,
                validation_errors={"exception": str(e)}
            )
    
    def _validate_characteristics(self) -> Dict[str, Any]:
        """Validate twin characteristics."""
        issues = []
        score = 1.0
        
        # Check for valid ranges
        for field, value in self.characteristics.dict().items():
            if isinstance(value, float) and field.endswith(('_level', '_tolerance', '_preference', '_rate')):
                if not 0.0 <= value <= 1.0:
                    issues.append(f"Invalid range for {field}: {value}")
                    score -= 0.1
        
        # Check for characteristic coherence
        if self.characteristics.technical_depth > 0.8 and self.characteristics.detail_preference < 0.3:
            issues.append("High technical depth but low detail preference may be inconsistent")
            score -= 0.1
        
        return {
            "score": max(0.0, score),
            "issues": issues,
            "valid_ranges": True if not issues else False
        }
    
    def _validate_memory(self) -> Dict[str, Any]:
        """Validate twin memory integrity."""
        issues = []
        score = 1.0
        
        # Check memory size limits
        if len(self.memory.short_term_memory) > 150:
            issues.append("Short-term memory exceeds recommended size")
            score -= 0.1
        
        # Check for duplicate interactions
        interaction_ids = [i.interaction_id for i in self.memory.short_term_memory]
        if len(interaction_ids) != len(set(interaction_ids)):
            issues.append("Duplicate interaction IDs found")
            score -= 0.2
        
        return {
            "score": max(0.0, score),
            "issues": issues,
            "memory_size": len(self.memory.short_term_memory)
        }
    
    def _validate_behavior_patterns(self) -> Dict[str, Any]:
        """Validate behavioral pattern consistency."""
        issues = []
        score = 1.0
        
        # Check if adaptation is working
        if self.interaction_count > 10 and not self.memory.adaptation_history:
            issues.append("No adaptations recorded despite multiple interactions")
            score -= 0.1
        
        # Check confidence score trends
        if len(self.confidence_scores) > 5:
            recent_scores = self.confidence_scores[-5:]
            if all(score < 0.5 for score in recent_scores):
                issues.append("Consistently low satisfaction scores")
                score -= 0.2
        
        return {
            "score": max(0.0, score),
            "issues": issues,
            "pattern_count": len(self.memory.long_term_patterns)
        }