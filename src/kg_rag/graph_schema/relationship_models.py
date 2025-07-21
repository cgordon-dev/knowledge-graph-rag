"""
Relationship models for Neo4j Vector Graph Schema.

Defines all relationship types with properties and constraints
for connecting nodes in the knowledge graph.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator
import structlog


class RelationshipType(str, Enum):
    """Enumeration of supported relationship types."""
    
    # Document relationships
    CONTAINS = "CONTAINS"
    PART_OF = "PART_OF"
    REFERENCES = "REFERENCES"
    MENTIONS = "MENTIONS"
    DERIVES_FROM = "DERIVES_FROM"
    
    # Entity relationships
    RELATED_TO = "RELATED_TO"
    SIMILAR_TO = "SIMILAR_TO"
    INSTANCE_OF = "INSTANCE_OF"
    SUBCLASS_OF = "SUBCLASS_OF"
    
    # Concept relationships
    IS_A = "IS_A"
    HAS_PART = "HAS_PART"
    BELONGS_TO = "BELONGS_TO"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    
    # Control and compliance relationships
    IMPLEMENTS = "IMPLEMENTS"
    COMPLIES_TO = "COMPLIES_TO"
    DEPENDS_ON = "DEPENDS_ON"
    SUPPORTS = "SUPPORTS"
    VALIDATES = "VALIDATES"
    
    # Process relationships
    TRIGGERS = "TRIGGERS"
    FOLLOWS = "FOLLOWS"
    INFLUENCES = "INFLUENCES"
    REQUIRES = "REQUIRES"
    PRODUCES = "PRODUCES"
    
    # Persona relationships
    PREFERS = "PREFERS"
    USES = "USES"
    INTERACTS_WITH = "INTERACTS_WITH"
    ASSIGNED_TO = "ASSIGNED_TO"


class BaseRelationship(BaseModel):
    """Base relationship model with common properties."""
    
    # Core properties
    relationship_id: str = Field(..., description="Unique relationship identifier")
    relationship_type: RelationshipType = Field(..., description="Relationship type")
    source_node_id: str = Field(..., description="Source node ID")
    target_node_id: str = Field(..., description="Target node ID")
    
    # Relationship properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship weight/strength")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Relationship confidence")
    
    # Context and evidence
    context: Optional[str] = Field(None, description="Relationship context")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    source: Optional[str] = Field(None, description="Data source")
    version: int = Field(default=1, description="Relationship version")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties."""
        props = {
            "relationship_id": self.relationship_id,
            "relationship_type": self.relationship_type,
            "weight": self.weight,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version
        }
        
        # Add optional fields
        if self.context:
            props["context"] = self.context
        if self.evidence:
            props["evidence"] = self.evidence
        if self.source:
            props["source"] = self.source
        
        # Add custom properties
        props.update(self.properties)
        
        return props
    
    def get_cypher_create_query(self) -> str:
        """Generate Cypher CREATE query for this relationship."""
        props = self.to_neo4j_properties()
        props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
        
        return f"""
        MATCH (source {{node_id: $source_node_id}})
        MATCH (target {{node_id: $target_node_id}})
        CREATE (source)-[r:{self.relationship_type} {{{props_str}}}]->(target)
        """
    
    def get_reverse_relationship_type(self) -> Optional[RelationshipType]:
        """Get the reverse relationship type if applicable."""
        reverse_mapping = {
            RelationshipType.CONTAINS: RelationshipType.PART_OF,
            RelationshipType.PART_OF: RelationshipType.CONTAINS,
            RelationshipType.IMPLEMENTS: RelationshipType.VALIDATES,
            RelationshipType.VALIDATES: RelationshipType.IMPLEMENTS,
            RelationshipType.TRIGGERS: RelationshipType.FOLLOWS,
            RelationshipType.FOLLOWS: RelationshipType.TRIGGERS,
        }
        return reverse_mapping.get(self.relationship_type)


class ContainsRelationship(BaseRelationship):
    """Relationship indicating containment (document contains chunks, etc.)."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.CONTAINS, const=True)
    
    # Containment-specific properties
    order: Optional[int] = Field(None, description="Order within container")
    position: Optional[Dict[str, Any]] = Field(None, description="Position information")
    percentage: Optional[float] = Field(None, ge=0.0, le=1.0, description="Percentage of container")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with containment-specific fields."""
        props = super().to_neo4j_properties()
        
        if self.order is not None:
            props["order"] = self.order
        if self.position:
            props["position"] = self.position
        if self.percentage is not None:
            props["percentage"] = self.percentage
        
        return props


class ReferencesRelationship(BaseRelationship):
    """Relationship indicating reference or citation."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.REFERENCES, const=True)
    
    # Reference-specific properties
    reference_type: str = Field(..., description="Type of reference (citation, mention, link)")
    location: Optional[str] = Field(None, description="Location within source")
    frequency: int = Field(default=1, description="Number of references")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with reference-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "reference_type": self.reference_type,
            "frequency": self.frequency
        })
        
        if self.location:
            props["location"] = self.location
        
        return props


class RelatedToRelationship(BaseRelationship):
    """Generic relationship indicating relatedness."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.RELATED_TO, const=True)
    
    # Relatedness-specific properties
    relation_nature: str = Field(..., description="Nature of relationship")
    similarity_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Similarity score")
    semantic_distance: Optional[float] = Field(None, ge=0.0, description="Semantic distance")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with relatedness-specific fields."""
        props = super().to_neo4j_properties()
        props["relation_nature"] = self.relation_nature
        
        if self.similarity_score is not None:
            props["similarity_score"] = self.similarity_score
        if self.semantic_distance is not None:
            props["semantic_distance"] = self.semantic_distance
        
        return props


class ImplementsRelationship(BaseRelationship):
    """Relationship indicating implementation of controls or requirements."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.IMPLEMENTS, const=True)
    
    # Implementation-specific properties
    implementation_status: str = Field(..., description="Implementation status")
    implementation_level: str = Field(default="partial", description="Implementation level")
    effectiveness: Optional[float] = Field(None, ge=0.0, le=1.0, description="Implementation effectiveness")
    
    # Assessment information
    last_assessed: Optional[datetime] = Field(None, description="Last assessment date")
    assessment_result: Optional[str] = Field(None, description="Assessment result")
    gaps: List[str] = Field(default_factory=list, description="Implementation gaps")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with implementation-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "implementation_status": self.implementation_status,
            "implementation_level": self.implementation_level
        })
        
        if self.effectiveness is not None:
            props["effectiveness"] = self.effectiveness
        if self.last_assessed:
            props["last_assessed"] = self.last_assessed.isoformat()
        if self.assessment_result:
            props["assessment_result"] = self.assessment_result
        if self.gaps:
            props["gaps"] = self.gaps
        
        return props


class CompliesToRelationship(BaseRelationship):
    """Relationship indicating compliance to standards or frameworks."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.COMPLIES_TO, const=True)
    
    # Compliance-specific properties
    compliance_status: str = Field(..., description="Compliance status")
    compliance_level: str = Field(..., description="Compliance level (Low, Moderate, High)")
    framework_version: Optional[str] = Field(None, description="Framework version")
    
    # Compliance assessment
    compliance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Compliance score")
    last_audit: Optional[datetime] = Field(None, description="Last audit date")
    next_audit: Optional[datetime] = Field(None, description="Next audit date")
    audit_findings: List[str] = Field(default_factory=list, description="Audit findings")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with compliance-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "compliance_status": self.compliance_status,
            "compliance_level": self.compliance_level
        })
        
        if self.framework_version:
            props["framework_version"] = self.framework_version
        if self.compliance_score is not None:
            props["compliance_score"] = self.compliance_score
        if self.last_audit:
            props["last_audit"] = self.last_audit.isoformat()
        if self.next_audit:
            props["next_audit"] = self.next_audit.isoformat()
        if self.audit_findings:
            props["audit_findings"] = self.audit_findings
        
        return props


class DependsOnRelationship(BaseRelationship):
    """Relationship indicating dependency between nodes."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.DEPENDS_ON, const=True)
    
    # Dependency-specific properties
    dependency_type: str = Field(..., description="Type of dependency")
    criticality: str = Field(default="medium", description="Dependency criticality")
    
    # Dependency constraints
    is_mandatory: bool = Field(default=True, description="Whether dependency is mandatory")
    can_be_substituted: bool = Field(default=False, description="Whether dependency can be substituted")
    substitutes: List[str] = Field(default_factory=list, description="Possible substitutes")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with dependency-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "dependency_type": self.dependency_type,
            "criticality": self.criticality,
            "is_mandatory": self.is_mandatory,
            "can_be_substituted": self.can_be_substituted
        })
        
        if self.substitutes:
            props["substitutes"] = self.substitutes
        
        return props


class InfluencesRelationship(BaseRelationship):
    """Relationship indicating influence between nodes."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.INFLUENCES, const=True)
    
    # Influence-specific properties
    influence_type: str = Field(..., description="Type of influence")
    influence_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Strength of influence")
    influence_direction: str = Field(default="positive", description="Direction of influence")
    
    # Temporal aspects
    influence_delay: Optional[float] = Field(None, description="Influence delay in time units")
    influence_duration: Optional[float] = Field(None, description="Duration of influence")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with influence-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "influence_type": self.influence_type,
            "influence_strength": self.influence_strength,
            "influence_direction": self.influence_direction
        })
        
        if self.influence_delay is not None:
            props["influence_delay"] = self.influence_delay
        if self.influence_duration is not None:
            props["influence_duration"] = self.influence_duration
        
        return props


class MentionsRelationship(BaseRelationship):
    """Relationship indicating mention of entities in documents."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.MENTIONS, const=True)
    
    # Mention-specific properties
    mention_context: str = Field(..., description="Context of the mention")
    mention_frequency: int = Field(default=1, description="Number of mentions")
    sentiment: Optional[str] = Field(None, description="Sentiment of mention")
    
    # Position information
    positions: List[Dict[str, Any]] = Field(default_factory=list, description="Mention positions")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with mention-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "mention_context": self.mention_context,
            "mention_frequency": self.mention_frequency
        })
        
        if self.sentiment:
            props["sentiment"] = self.sentiment
        if self.positions:
            props["positions"] = self.positions
        
        return props


class SimilarToRelationship(BaseRelationship):
    """Relationship indicating similarity between nodes."""
    
    relationship_type: RelationshipType = Field(default=RelationshipType.SIMILAR_TO, const=True)
    
    # Similarity-specific properties
    similarity_metric: str = Field(..., description="Similarity metric used")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    similarity_aspects: List[str] = Field(default_factory=list, description="Aspects of similarity")
    
    # Comparison metadata
    comparison_method: str = Field(..., description="Method used for comparison")
    feature_vector: Optional[List[float]] = Field(None, description="Feature vector for comparison")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with similarity-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "similarity_metric": self.similarity_metric,
            "similarity_score": self.similarity_score,
            "comparison_method": self.comparison_method
        })
        
        if self.similarity_aspects:
            props["similarity_aspects"] = self.similarity_aspects
        if self.feature_vector:
            props["feature_vector"] = self.feature_vector
        
        return props


# Relationship factory for creating relationships from data
class RelationshipFactory:
    """Factory for creating relationship instances from data."""
    
    RELATIONSHIP_TYPES = {
        RelationshipType.CONTAINS: ContainsRelationship,
        RelationshipType.REFERENCES: ReferencesRelationship,
        RelationshipType.RELATED_TO: RelatedToRelationship,
        RelationshipType.IMPLEMENTS: ImplementsRelationship,
        RelationshipType.COMPLIES_TO: CompliesToRelationship,
        RelationshipType.DEPENDS_ON: DependsOnRelationship,
        RelationshipType.INFLUENCES: InfluencesRelationship,
        RelationshipType.MENTIONS: MentionsRelationship,
        RelationshipType.SIMILAR_TO: SimilarToRelationship,
    }
    
    @classmethod
    def create_relationship(cls, relationship_type: RelationshipType, data: Dict[str, Any]) -> BaseRelationship:
        """Create a relationship instance from data."""
        relationship_class = cls.RELATIONSHIP_TYPES.get(relationship_type, BaseRelationship)
        return relationship_class(**data)
    
    @classmethod
    def from_neo4j_record(cls, record: Dict[str, Any]) -> BaseRelationship:
        """Create relationship from Neo4j record."""
        relationship_type = RelationshipType(record.get("relationship_type"))
        
        # Convert Neo4j record to relationship data
        relationship_data = record.copy()
        
        # Handle timestamps
        if "created_at" in record:
            relationship_data["created_at"] = datetime.fromisoformat(record["created_at"])
        if "updated_at" in record:
            relationship_data["updated_at"] = datetime.fromisoformat(record["updated_at"])
        if "last_assessed" in record:
            relationship_data["last_assessed"] = datetime.fromisoformat(record["last_assessed"])
        if "last_audit" in record:
            relationship_data["last_audit"] = datetime.fromisoformat(record["last_audit"])
        if "next_audit" in record:
            relationship_data["next_audit"] = datetime.fromisoformat(record["next_audit"])
        
        return cls.create_relationship(relationship_type, relationship_data)
    
    @classmethod
    def create_bidirectional_relationships(
        cls, 
        relationship_type: RelationshipType, 
        source_id: str, 
        target_id: str, 
        data: Dict[str, Any]
    ) -> List[BaseRelationship]:
        """Create bidirectional relationships."""
        relationships = []
        
        # Create forward relationship
        forward_data = data.copy()
        forward_data.update({
            "source_node_id": source_id,
            "target_node_id": target_id,
            "relationship_type": relationship_type
        })
        relationships.append(cls.create_relationship(relationship_type, forward_data))
        
        # Create reverse relationship if applicable
        reverse_type = BaseRelationship(
            relationship_id="temp", 
            relationship_type=relationship_type,
            source_node_id=source_id,
            target_node_id=target_id
        ).get_reverse_relationship_type()
        
        if reverse_type:
            reverse_data = data.copy()
            reverse_data.update({
                "source_node_id": target_id,
                "target_node_id": source_id,
                "relationship_type": reverse_type,
                "relationship_id": f"{data.get('relationship_id', 'rel')}_reverse"
            })
            relationships.append(cls.create_relationship(reverse_type, reverse_data))
        
        return relationships


# Utility functions for relationship management
def calculate_relationship_strength(
    properties: Dict[str, Any], 
    relationship_type: RelationshipType
) -> float:
    """Calculate relationship strength based on properties and type."""
    
    base_strength = properties.get("weight", 0.5)
    confidence = properties.get("confidence", 1.0)
    
    # Type-specific adjustments
    type_multipliers = {
        RelationshipType.CONTAINS: 1.0,
        RelationshipType.REFERENCES: 0.8,
        RelationshipType.SIMILAR_TO: properties.get("similarity_score", 0.5),
        RelationshipType.IMPLEMENTS: properties.get("effectiveness", 0.7),
        RelationshipType.COMPLIES_TO: properties.get("compliance_score", 0.6),
        RelationshipType.DEPENDS_ON: 0.9 if properties.get("is_mandatory", True) else 0.6,
        RelationshipType.INFLUENCES: properties.get("influence_strength", 0.5),
    }
    
    multiplier = type_multipliers.get(relationship_type, 1.0)
    
    return min(base_strength * confidence * multiplier, 1.0)


def validate_relationship_consistency(relationship: BaseRelationship) -> List[str]:
    """Validate relationship consistency and return issues."""
    issues = []
    
    # Check basic consistency
    if relationship.weight < 0 or relationship.weight > 1:
        issues.append("Weight must be between 0 and 1")
    
    if relationship.confidence < 0 or relationship.confidence > 1:
        issues.append("Confidence must be between 0 and 1")
    
    if relationship.source_node_id == relationship.target_node_id:
        issues.append("Self-referential relationships may indicate data issues")
    
    # Type-specific validation
    if isinstance(relationship, SimilarToRelationship):
        if relationship.similarity_score < 0 or relationship.similarity_score > 1:
            issues.append("Similarity score must be between 0 and 1")
    
    if isinstance(relationship, ImplementsRelationship):
        if relationship.effectiveness is not None:
            if relationship.effectiveness < 0 or relationship.effectiveness > 1:
                issues.append("Implementation effectiveness must be between 0 and 1")
    
    return issues