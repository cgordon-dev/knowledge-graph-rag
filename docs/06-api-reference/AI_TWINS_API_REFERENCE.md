# AI Digital Twins API Reference

Comprehensive API reference for the Knowledge Graph-RAG AI Digital Twins framework.

## Table of Contents

- [Overview](#overview)
- [Base Twin API](#base-twin-api)
- [Twin Orchestrator API](#twin-orchestrator-api)
- [Expert Twin API](#expert-twin-api)
- [User Journey Twin API](#user-journey-twin-api)
- [Process Automation Twin API](#process-automation-twin-api)
- [Persona Twin API](#persona-twin-api)
- [Data Models](#data-models)
- [Error Handling](#error-handling)
- [Examples](#examples)

## Overview

The AI Digital Twins framework provides behavioral modeling capabilities for different personas, experts, and processes. Each twin type offers specialized functionality while sharing common base capabilities.

### Core Concepts

- **BaseTwin**: Foundation class with memory, learning, and adaptation
- **TwinOrchestrator**: Manages multiple twins and enables collaboration
- **Behavioral Modeling**: Twins adapt their responses based on characteristics
- **Memory Systems**: Short-term interactions and long-term pattern learning
- **Collaborative Intelligence**: Multiple twins working together

### Authentication & Authorization

```python
# Import the framework
from kg_rag.ai_twins import (
    TwinOrchestrator, ExpertTwin, UserJourneyTwin, 
    ProcessAutomationTwin, PersonaTwin
)

# All operations are async
import asyncio
```

---

## Base Twin API

All twins inherit from `BaseTwin` providing core functionality.

### Core Methods

#### `interact(query, context=None, user_id=None)`

Main interaction method for communicating with any twin.

**Parameters:**
- `query` (str): User query or request
- `context` (dict, optional): Additional context information
- `user_id` (str, optional): User identifier for tracking

**Returns:**
- `dict`: Interaction result with response and metadata

**Example:**
```python
result = await twin.interact(
    query="How should I approach this problem?",
    context={"domain": "security", "urgency": "high"},
    user_id="user_123"
)

print(result["response"])  # Twin's response
print(result["confidence"])  # Confidence score (0.0-1.0)
print(result["twin_metadata"])  # Twin information
```

#### `provide_feedback(interaction_id, satisfaction_score, feedback_text=None)`

Provide feedback on a previous interaction for learning.

**Parameters:**
- `interaction_id` (str): ID from previous interaction
- `satisfaction_score` (float): Satisfaction rating (0.0-1.0)
- `feedback_text` (str, optional): Textual feedback

**Example:**
```python
await twin.provide_feedback(
    interaction_id="interaction_123",
    satisfaction_score=0.9,
    feedback_text="Very helpful response"
)
```

#### `validate_consistency()`

Validate twin's behavioral consistency and health.

**Returns:**
- `dict`: Validation results with consistency score and issues

**Example:**
```python
validation = await twin.validate_consistency()
print(f"Consistency score: {validation['consistency_score']}")
if validation['issues']:
    print("Issues found:", validation['issues'])
```

#### `get_twin_state()`

Get complete twin state for serialization or monitoring.

**Returns:**
- `dict`: Complete twin state including characteristics and history

---

## Twin Orchestrator API

Manages multiple twins and enables collaborative processing.

### Initialization

```python
orchestrator = TwinOrchestrator()
```

### Twin Management

#### `register_twin(twin, capabilities=None, domains=None, priority=5)`

Register a twin with the orchestrator.

**Parameters:**
- `twin` (BaseTwin): Twin instance to register
- `capabilities` (list, optional): List of twin capabilities
- `domains` (list, optional): List of domain expertise areas
- `priority` (int, optional): Priority level 1-10

**Returns:**
- `TwinRegistration`: Registration information

**Example:**
```python
registration = await orchestrator.register_twin(
    twin=expert_twin,
    capabilities=["validation", "consultation", "analysis"],
    domains=["compliance", "security"],
    priority=8
)
```

#### `unregister_twin(twin_id)`

Remove a twin from the orchestrator.

**Parameters:**
- `twin_id` (str): Twin identifier

**Returns:**
- `bool`: Success status

### Query Processing

#### `process_query(query, context=None, preferred_twin_type=None, enable_collaboration=True, user_id=None)`

Process a query through appropriate twin(s).

**Parameters:**
- `query` (str): User query
- `context` (dict, optional): Additional context
- `preferred_twin_type` (str, optional): Preferred twin type
- `enable_collaboration` (bool): Enable multi-twin collaboration
- `user_id` (str, optional): User identifier

**Returns:**
- `TwinInteractionResult` or `CollaborativeResult`: Processing result

**Example:**
```python
# Single twin processing
result = await orchestrator.process_query(
    query="How do I implement security controls?",
    preferred_twin_type="expert",
    enable_collaboration=False
)

# Collaborative processing
result = await orchestrator.process_query(
    query="Optimize our user onboarding process",
    enable_collaboration=True
)
```

#### `get_twin_recommendations(query, context=None, max_recommendations=5)`

Get twin recommendations without executing.

**Parameters:**
- `query` (str): Query to analyze
- `context` (dict, optional): Additional context
- `max_recommendations` (int): Maximum recommendations

**Returns:**
- `list`: Twin recommendations with relevance scores

**Example:**
```python
recommendations = await orchestrator.get_twin_recommendations(
    query="I need help with process optimization",
    max_recommendations=3
)

for rec in recommendations:
    print(f"{rec['twin_name']}: {rec['relevance_score']:.2f}")
```

### Collaboration Management

#### `enable_twin_collaboration(primary_twin_id, supporting_twin_ids, collaboration_strategy="consensus")`

Enable collaboration between specific twins.

**Parameters:**
- `primary_twin_id` (str): Primary twin identifier
- `supporting_twin_ids` (list): Supporting twin identifiers
- `collaboration_strategy` (str): Strategy (consensus, expert_review, synthesis)

**Returns:**
- `dict`: Collaboration configuration

### Statistics and Monitoring

#### `get_orchestrator_statistics()`

Get comprehensive orchestrator performance statistics.

**Returns:**
- `dict`: Statistics including performance metrics and twin status

**Example:**
```python
stats = await orchestrator.get_orchestrator_statistics()
print(f"Total twins: {stats['total_twins']}")
print(f"Success rate: {stats['performance_metrics']['success_rate']:.1f}%")
```

---

## Expert Twin API

Domain specialist simulation with validation capabilities.

### Initialization

```python
from kg_rag.ai_twins import ExpertTwin, ExpertDomain

# Define expert domain
domain = ExpertDomain(
    domain_name="compliance",
    expertise_level=0.9,
    specializations=["FedRAMP", "NIST", "SOC2"],
    experience_years=15,
    knowledge_sources=["NIST 800-53", "FedRAMP guidelines"],
    validation_criteria={"min_score": 0.8}
)

# Create expert twin
expert = ExpertTwin(
    expert_id="compliance_expert_001",
    name="Dr. Sarah Mitchell",
    domain=domain,
    description="Senior compliance expert specializing in government frameworks"
)
```

### Expert Consultation

#### `provide_consultation(consultation_request, urgency="normal", required_confidence=0.8)`

Get expert consultation on a specific topic.

**Parameters:**
- `consultation_request` (str): Consultation request
- `urgency` (str): Urgency level (low, normal, high, critical)
- `required_confidence` (float): Minimum confidence threshold

**Returns:**
- `dict`: Consultation response with recommendations

**Example:**
```python
consultation = await expert.provide_consultation(
    consultation_request="How should we implement NIST 800-53 controls for our cloud infrastructure?",
    urgency="high",
    required_confidence=0.85
)

print(consultation["response"])
print(f"Confidence: {consultation['confidence']:.2f}")
```

### Content Validation

#### `validate_content(content, validation_criteria=None, require_consensus=False)`

Validate content against expert knowledge.

**Parameters:**
- `content` (str): Content to validate
- `validation_criteria` (dict, optional): Specific criteria
- `require_consensus` (bool): Require consensus with other experts

**Returns:**
- `ExpertValidation`: Detailed validation result

**Example:**
```python
validation = await expert.validate_content(
    content="Our security implementation includes...",
    require_consensus=True
)

print(f"Valid: {validation.is_valid}")
print(f"Confidence: {validation.confidence_score:.2f}")
print(f"Issues: {validation.issues_identified}")
print(f"Recommendations: {validation.recommendations}")
```

### Expert Statistics

#### `get_expert_statistics()`

Get expert performance and interaction statistics.

**Returns:**
- `dict`: Expert statistics including validation metrics

---

## User Journey Twin API

UX optimization with persona-driven insights.

### Initialization

```python
from kg_rag.ai_twins import UserJourneyTwin, UserPersona, UserJourneyStep

# Define user persona
persona = UserPersona(
    persona_id="enterprise_admin",
    persona_name="Enterprise System Administrator",
    digital_literacy=0.9,
    patience_level=0.6,
    primary_goals=["system_efficiency", "security_compliance"],
    preferred_channels=["web_interface", "documentation"]
)

# Define journey steps
steps = [
    UserJourneyStep(
        step_id="discovery",
        step_name="System Discovery",
        step_type="discovery",
        description="User discovers system capabilities",
        completion_rate=0.85,
        average_duration_minutes=15,
        pain_points=["Complex navigation", "Unclear features"]
    )
]

# Create journey twin
journey = UserJourneyTwin(
    journey_id="admin_journey_001",
    name="Admin User Journey",
    persona=persona,
    journey_steps=steps
)
```

### Journey Analysis

#### `analyze_journey_step(step_id, performance_data=None)`

Analyze a specific journey step for optimization opportunities.

**Parameters:**
- `step_id` (str): Journey step identifier
- `performance_data` (dict, optional): Updated performance metrics

**Returns:**
- `dict`: Step analysis with optimization recommendations

**Example:**
```python
analysis = await journey.analyze_journey_step(
    step_id="discovery",
    performance_data={
        "completion_rate": 0.82,
        "average_duration_minutes": 18
    }
)

print(f"Health score: {analysis['health_score']:.2f}")
print("Optimization opportunities:", analysis['optimization_opportunities'])
```

### Journey Optimization

#### `recommend_journey_optimizations(priority_threshold=0.6, max_recommendations=10)`

Generate comprehensive journey optimization recommendations.

**Parameters:**
- `priority_threshold` (float): Minimum priority score for recommendations
- `max_recommendations` (int): Maximum number of recommendations

**Returns:**
- `list[JourneyOptimization]`: Prioritized optimization recommendations

**Example:**
```python
optimizations = await journey.recommend_journey_optimizations(
    priority_threshold=0.7,
    max_recommendations=5
)

for opt in optimizations:
    print(f"Priority: {opt.priority_score:.2f}")
    print(f"Opportunity: {opt.opportunity_assessment}")
    print(f"Actions: {opt.recommended_actions}")
    print(f"Expected Impact: {opt.expected_impact}")
```

### Journey Statistics

#### `get_journey_statistics()`

Get comprehensive journey performance statistics.

**Returns:**
- `dict`: Journey statistics including step performance and optimization metrics

---

## Process Automation Twin API

Workflow intelligence with ROI-driven automation recommendations.

### Initialization

```python
from kg_rag.ai_twins import ProcessAutomationTwin, ProcessStep, ProcessMetrics

# Define process steps
steps = [
    ProcessStep(
        step_id="data_validation",
        step_name="Data Validation",
        step_type="manual",
        description="Validate incoming data quality",
        automation_potential=0.8,
        average_duration_minutes=30,
        error_rate=0.05,
        cost_per_execution=25.0
    )
]

# Define process metrics
metrics = ProcessMetrics(
    cycle_time_minutes=45,
    processing_time_minutes=35,
    wait_time_minutes=10,
    first_time_right_rate=0.9,
    daily_volume=100,
    total_cost_per_transaction=30.0
)

# Create process twin
process = ProcessAutomationTwin(
    process_id="data_processing_001",
    name="Data Processing Workflow",
    process_steps=steps,
    current_metrics=metrics
)
```

### Automation Analysis

#### `analyze_automation_opportunities(min_potential=0.5, max_recommendations=10)`

Analyze and generate automation recommendations.

**Parameters:**
- `min_potential` (float): Minimum automation potential threshold
- `max_recommendations` (int): Maximum number of recommendations

**Returns:**
- `list[AutomationRecommendation]`: Prioritized automation recommendations

**Example:**
```python
opportunities = await process.analyze_automation_opportunities(
    min_potential=0.6,
    max_recommendations=5
)

for opp in opportunities:
    print(f"Target: {opp.target_step}")
    print(f"Type: {opp.automation_type}")
    print(f"ROI: {opp.roi_projection:.1f}x")
    print(f"Payback: {opp.payback_period_months:.1f} months")
    print(f"Savings: ${opp.estimated_savings.get('annual_savings', 0):,.0f}")
```

### Bottleneck Analysis

#### `perform_bottleneck_analysis()`

Perform comprehensive bottleneck analysis of the process.

**Returns:**
- `dict`: Bottleneck analysis with identification and resolution strategies

**Example:**
```python
analysis = await process.perform_bottleneck_analysis()

print(f"Efficiency score: {analysis['overall_efficiency_score']:.2f}")
print(f"Bottlenecks found: {analysis['bottlenecks_identified']}")

for bottleneck in analysis['bottleneck_details']:
    print(f"Step: {bottleneck['step_name']}")
    print(f"Score: {bottleneck['bottleneck_score']:.2f}")
    print(f"Factors: {bottleneck['bottleneck_factors']}")
```

### ROI Calculation

#### `calculate_automation_roi(recommendation, timeframe_months=12)`

Calculate detailed ROI for an automation recommendation.

**Parameters:**
- `recommendation` (AutomationRecommendation): Recommendation to analyze
- `timeframe_months` (int): Analysis timeframe

**Returns:**
- `dict`: Detailed ROI analysis with costs, savings, and projections

**Example:**
```python
roi_analysis = await process.calculate_automation_roi(
    recommendation=automation_rec,
    timeframe_months=24
)

print(f"Implementation cost: ${roi_analysis['implementation']['implementation_cost']:,.0f}")
print(f"Annual savings: ${roi_analysis['financial_benefits']['monthly_savings'] * 12:,.0f}")
print(f"ROI: {roi_analysis['financial_benefits']['roi_percentage']:.1f}%")
print(f"Payback: {roi_analysis['implementation']['payback_period_months']:.1f} months")
```

### Process Statistics

#### `get_process_statistics()`

Get comprehensive process performance statistics.

**Returns:**
- `dict`: Process statistics including automation metrics and performance data

---

## Persona Twin API

Flexible role-based interaction modeling.

### Initialization

```python
from kg_rag.ai_twins import PersonaTwin, PersonaProfile, PersonaCharacteristics

# Define persona profile
profile = PersonaProfile(
    persona_name="Senior DevOps Engineer",
    role="DevOps Engineer",
    background="10+ years in infrastructure automation and deployment",
    communication_style="technical",
    technical_level="expert",
    primary_objectives=["reliability", "automation", "efficiency"],
    areas_of_expertise=["kubernetes", "ci/cd", "monitoring", "security"]
)

# Create persona twin
persona = PersonaTwin(
    persona_id="devops_expert_001",
    name="Alex Thompson",
    profile=profile
)
```

### Role Adaptation

#### `adapt_to_role_context(role_context, temporary=False)`

Adapt persona to specific role context.

**Parameters:**
- `role_context` (dict): Context requiring role adaptation
- `temporary` (bool): Whether adaptation is temporary

**Returns:**
- `dict`: Adaptation result and configuration

**Example:**
```python
adaptation = await persona.adapt_to_role_context(
    role_context={
        "formal_communication": True,
        "technical_complexity": "high",
        "detail_requirement": "high"
    },
    temporary=True
)

print(f"Adaptations applied: {adaptation['adaptations_applied']}")
print(f"Consistency score: {adaptation['consistency_score']:.2f}")
```

#### `revert_adaptations(adaptation_ids=None)`

Revert persona adaptations.

**Parameters:**
- `adaptation_ids` (list, optional): Specific adaptations to revert

**Returns:**
- `dict`: Reversion result

**Example:**
```python
reversion = await persona.revert_adaptations()
print(f"Reverted {reversion['reverted_count']} adaptations")
```

### Persona Statistics

#### `get_persona_statistics()`

Get comprehensive persona statistics and adaptation history.

**Returns:**
- `dict`: Persona statistics including consistency and adaptation metrics

---

## Data Models

### Core Data Types

#### `TwinCharacteristics`

Base characteristics for all twins.

```python
class TwinCharacteristics(BaseModel):
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0)
    detail_preference: float = Field(default=0.7, ge=0.0, le=1.0)
    technical_depth: float = Field(default=0.6, ge=0.0, le=1.0)
    response_speed: float = Field(default=0.8, ge=0.0, le=1.0)
    formality_level: float = Field(default=0.6, ge=0.0, le=1.0)
    empathy_level: float = Field(default=0.7, ge=0.0, le=1.0)
```

#### `TwinInteraction`

Represents an interaction with a digital twin.

```python
class TwinInteraction(BaseModel):
    interaction_id: str
    timestamp: datetime
    query: str
    response: str
    context: Dict[str, Any] = Field(default_factory=dict)
    satisfaction_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

#### `TwinMemory`

Memory system for digital twins.

```python
class TwinMemory(BaseModel):
    short_term_memory: List[TwinInteraction] = Field(default_factory=list)
    long_term_patterns: Dict[str, Any] = Field(default_factory=dict)
    adaptation_history: List[Dict[str, Any]] = Field(default_factory=list)
    context_cache: Dict[str, Any] = Field(default_factory=dict)
```

### Expert Twin Data Types

#### `ExpertDomain`

Expert domain specification.

```python
class ExpertDomain(BaseModel):
    domain_name: str
    expertise_level: float = Field(ge=0.0, le=1.0)
    specializations: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    experience_years: int = Field(default=5)
    knowledge_sources: List[str] = Field(default_factory=list)
    validation_criteria: Dict[str, Any] = Field(default_factory=dict)
```

#### `ExpertValidation`

Expert validation result.

```python
class ExpertValidation(BaseModel):
    validation_id: str
    expert_id: str
    content: str
    is_valid: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    accuracy_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    feedback: str = Field(default="")
    issues_identified: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
```

### User Journey Data Types

#### `UserPersona`

User persona characteristics.

```python
class UserPersona(BaseModel):
    persona_id: str
    persona_name: str
    digital_literacy: float = Field(default=0.7, ge=0.0, le=1.0)
    patience_level: float = Field(default=0.6, ge=0.0, le=1.0)
    primary_goals: List[str] = Field(default_factory=list)
    preferred_channels: List[str] = Field(default_factory=list)
    decision_factors: List[str] = Field(default_factory=list)
```

#### `UserJourneyStep`

Represents a step in a user journey.

```python
class UserJourneyStep(BaseModel):
    step_id: str
    step_name: str
    step_type: str  # discovery, evaluation, decision, action
    description: str
    completion_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    average_duration_minutes: float = Field(default=0.0)
    abandonment_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    pain_points: List[str] = Field(default_factory=list)
```

### Process Automation Data Types

#### `ProcessStep`

Represents a step in a business process.

```python
class ProcessStep(BaseModel):
    step_id: str
    step_name: str
    step_type: str  # manual, automated, decision, approval
    description: str
    is_automated: bool = Field(default=False)
    automation_potential: float = Field(default=0.0, ge=0.0, le=1.0)
    average_duration_minutes: float = Field(default=0.0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    cost_per_execution: float = Field(default=0.0)
```

#### `AutomationRecommendation`

Automation recommendation for a process step.

```python
class AutomationRecommendation(BaseModel):
    recommendation_id: str
    target_step: str
    automation_type: str
    automation_feasibility: float = Field(ge=0.0, le=1.0)
    estimated_savings: Dict[str, float] = Field(default_factory=dict)
    roi_projection: float = Field(default=0.0)
    payback_period_months: float = Field(default=0.0)
    priority_score: float = Field(ge=0.0, le=1.0)
    confidence_level: float = Field(ge=0.0, le=1.0)
```

---

## Error Handling

### Exception Hierarchy

```python
from kg_rag.core.exceptions import (
    PersonaTwinError,           # Base twin error
    PersonaValidationError,     # Validation failures
    ExpertTwinError,           # Expert twin specific errors
    ExpertValidationError,     # Expert validation errors
    MCPServerError,            # MCP server errors
    ConfigurationError         # Configuration errors
)
```

### Error Response Format

All API methods return structured error information:

```python
try:
    result = await twin.interact("query")
except PersonaTwinError as e:
    print(f"Twin error: {e.message}")
    print(f"Twin ID: {e.persona_id}")
    print(f"Error details: {e.details}")
```

### Common Error Scenarios

1. **Twin Not Found**: Twin ID doesn't exist
2. **Validation Failure**: Input validation errors
3. **Confidence Too Low**: Response confidence below threshold
4. **Resource Exhaustion**: Memory or processing limits exceeded
5. **Configuration Error**: Invalid configuration parameters

---

## Examples

### Complete Example: Multi-Twin Collaboration

```python
import asyncio
from kg_rag.ai_twins import *

async def example_collaboration():
    # Initialize orchestrator
    orchestrator = TwinOrchestrator()
    
    # Create expert twin
    compliance_domain = ExpertDomain(
        domain_name="compliance",
        expertise_level=0.95,
        specializations=["FedRAMP", "NIST", "SOC2"]
    )
    expert = ExpertTwin("expert_001", "Dr. Smith", compliance_domain)
    
    # Create process twin
    process_steps = [ProcessStep(
        step_id="audit_prep",
        step_name="Audit Preparation",
        step_type="manual",
        automation_potential=0.7
    )]
    process_metrics = ProcessMetrics(cycle_time_minutes=120)
    process = ProcessAutomationTwin("process_001", "Audit Process", 
                                   process_steps, process_metrics)
    
    # Register twins
    await orchestrator.register_twin(expert, domains=["compliance"])
    await orchestrator.register_twin(process, domains=["business_process"])
    
    # Collaborative query
    result = await orchestrator.process_query(
        query="How can we optimize our compliance audit process?",
        enable_collaboration=True
    )
    
    print("Collaborative Response:")
    print(result.synthesized_response)
    print(f"\nTwins involved: {len(result.contributing_twins) + 1}")
    print(f"Confidence: {result.confidence_score:.2f}")

# Run example
asyncio.run(example_collaboration())
```

### Example: Expert Validation Workflow

```python
async def expert_validation_workflow():
    # Create expert
    security_domain = ExpertDomain(
        domain_name="security",
        expertise_level=0.9,
        specializations=["penetration_testing", "vulnerability_assessment"]
    )
    expert = ExpertTwin("security_expert", "Alice Johnson", security_domain)
    
    # Content to validate
    content = """
    Our security implementation includes:
    - Multi-factor authentication for all users
    - Regular vulnerability scans
    - Encrypted data transmission
    - Access logging and monitoring
    """
    
    # Validate content
    validation = await expert.validate_content(
        content=content,
        require_consensus=False
    )
    
    print(f"Content valid: {validation.is_valid}")
    print(f"Confidence: {validation.confidence_score:.2f}")
    print(f"Accuracy: {validation.accuracy_score:.2f}")
    
    if validation.issues_identified:
        print("\nIssues found:")
        for issue in validation.issues_identified:
            print(f"- {issue}")
    
    if validation.recommendations:
        print("\nRecommendations:")
        for rec in validation.recommendations:
            print(f"- {rec}")

asyncio.run(expert_validation_workflow())
```

### Example: Journey Optimization

```python
async def journey_optimization_example():
    # Create user persona
    persona = UserPersona(
        persona_id="power_user",
        persona_name="Power User",
        digital_literacy=0.9,
        patience_level=0.4,
        primary_goals=["efficiency", "control"]
    )
    
    # Create journey steps
    steps = [
        UserJourneyStep(
            step_id="login",
            step_name="User Login",
            step_type="action",
            completion_rate=0.95,
            average_duration_minutes=2
        ),
        UserJourneyStep(
            step_id="navigation",
            step_name="Feature Navigation", 
            step_type="discovery",
            completion_rate=0.75,
            average_duration_minutes=8,
            pain_points=["Complex menus", "Hidden features"]
        )
    ]
    
    # Create journey twin
    journey = UserJourneyTwin("user_journey", "Power User Journey", 
                             persona, steps)
    
    # Get optimization recommendations
    optimizations = await journey.recommend_journey_optimizations(
        priority_threshold=0.6
    )
    
    print(f"Found {len(optimizations)} optimization opportunities:")
    for opt in optimizations:
        print(f"\nStep: {opt.journey_step}")
        print(f"Type: {opt.optimization_type}")
        print(f"Priority: {opt.priority_score:.2f}")
        print(f"Expected impact: {opt.expected_impact}")

asyncio.run(journey_optimization_example())
```

---

This API reference provides comprehensive documentation for the AI Digital Twins framework. Each twin type offers specialized capabilities while maintaining consistent interfaces for integration and collaboration.