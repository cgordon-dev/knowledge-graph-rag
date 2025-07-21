"""
AI Digital Twins Framework for Knowledge Graph-RAG.

Implements behavioral modeling for users, experts, and processes
following the AI Digital Twins best practices framework.
"""

from kg_rag.ai_twins.base_twin import BaseTwin, TwinCharacteristics, TwinInteraction, TwinMemory
from kg_rag.ai_twins.persona_twin import PersonaTwin, PersonaProfile, PersonaCharacteristics
from kg_rag.ai_twins.expert_twin import ExpertTwin, ExpertDomain, ExpertValidation, ExpertCharacteristics
from kg_rag.ai_twins.user_journey_twin import UserJourneyTwin, UserPersona, UserJourneyStep, JourneyOptimization, UserJourneyCharacteristics
from kg_rag.ai_twins.process_automation_twin import ProcessAutomationTwin, ProcessStep, ProcessMetrics, AutomationRecommendation, ProcessAutomationCharacteristics
from kg_rag.ai_twins.twin_orchestrator import TwinOrchestrator, TwinRegistration, TwinInteractionResult, CollaborativeResult

__all__ = [
    # Base classes
    "BaseTwin",
    "TwinCharacteristics", 
    "TwinInteraction",
    "TwinMemory",
    
    # Persona Twin
    "PersonaTwin",
    "PersonaProfile",
    "PersonaCharacteristics",
    
    # Expert Twin
    "ExpertTwin",
    "ExpertDomain",
    "ExpertValidation", 
    "ExpertCharacteristics",
    
    # User Journey Twin
    "UserJourneyTwin",
    "UserPersona",
    "UserJourneyStep",
    "JourneyOptimization",
    "UserJourneyCharacteristics",
    
    # Process Automation Twin
    "ProcessAutomationTwin",
    "ProcessStep",
    "ProcessMetrics",
    "AutomationRecommendation",
    "ProcessAutomationCharacteristics",
    
    # Orchestrator
    "TwinOrchestrator",
    "TwinRegistration", 
    "TwinInteractionResult",
    "CollaborativeResult"
]