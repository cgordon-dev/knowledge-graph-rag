"""
Node models for Neo4j Vector Graph Schema.

Defines all node types with vector embedding support, properties,
and constraints for the knowledge graph structure.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import numpy as np
from pydantic import BaseModel, Field, validator
import structlog

from kg_rag.core.config import get_settings


class NodeType(str, Enum):
    """Enumeration of supported node types."""
    
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    ENTITY = "Entity"
    CONCEPT = "Concept"
    CONTROL = "Control"
    PERSONA = "Persona"
    PROCESS = "Process"
    TOPIC = "Topic"
    KEYWORD = "Keyword"
    METADATA = "Metadata"


class VectorEmbedding(BaseModel):
    """Vector embedding with metadata."""
    
    vector: List[float] = Field(..., description="Embedding vector")
    model: str = Field(..., description="Embedding model used")
    dimension: int = Field(..., description="Vector dimension")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    
    @validator('vector')
    def validate_vector_length(cls, v, values):
        """Validate vector dimension matches declared dimension."""
        if 'dimension' in values and len(v) != values['dimension']:
            raise ValueError(f"Vector length {len(v)} doesn't match dimension {values['dimension']}")
        return v
    
    def to_neo4j_format(self) -> List[float]:
        """Convert to Neo4j vector format."""
        return self.vector
    
    def similarity(self, other: 'VectorEmbedding') -> float:
        """Calculate cosine similarity with another embedding."""
        if self.dimension != other.dimension:
            raise ValueError("Cannot compare embeddings of different dimensions")
        
        # Convert to numpy arrays
        a = np.array(self.vector)
        b = np.array(other.vector)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))


class BaseNode(BaseModel):
    """Base node model with common properties and vector support."""
    
    # Core properties
    node_id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Node type")
    title: str = Field(..., description="Node title")
    description: Optional[str] = Field(None, description="Node description")
    
    # Vector embeddings
    content_embedding: Optional[VectorEmbedding] = Field(None, description="Content embedding")
    title_embedding: Optional[VectorEmbedding] = Field(None, description="Title embedding")
    
    # Metadata
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional properties")
    tags: List[str] = Field(default_factory=list, description="Node tags")
    categories: List[str] = Field(default_factory=list, description="Node categories")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    # System metadata
    source: Optional[str] = Field(None, description="Data source")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Data confidence score")
    version: int = Field(default=1, description="Node version")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties."""
        props = {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence": self.confidence,
            "version": self.version
        }
        
        # Add optional fields
        if self.description:
            props["description"] = self.description
        if self.source:
            props["source"] = self.source
        if self.tags:
            props["tags"] = self.tags
        if self.categories:
            props["categories"] = self.categories
        
        # Add vector embeddings
        if self.content_embedding:
            props["content_embedding"] = self.content_embedding.to_neo4j_format()
            props["content_embedding_model"] = self.content_embedding.model
        if self.title_embedding:
            props["title_embedding"] = self.title_embedding.to_neo4j_format()
            props["title_embedding_model"] = self.title_embedding.model
        
        # Add custom properties
        props.update(self.properties)
        
        return props
    
    def get_cypher_create_query(self) -> str:
        """Generate Cypher CREATE query for this node."""
        props = self.to_neo4j_properties()
        props_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
        return f"CREATE (n:{self.node_type} {{{props_str}}})"
    
    def get_embedding_vector(self, embedding_type: str = "content") -> Optional[List[float]]:
        """Get embedding vector by type."""
        if embedding_type == "content" and self.content_embedding:
            return self.content_embedding.vector
        elif embedding_type == "title" and self.title_embedding:
            return self.title_embedding.vector
        return None


class DocumentNode(BaseNode):
    """Document node representing a complete document."""
    
    node_type: NodeType = Field(default=NodeType.DOCUMENT, const=True)
    
    # Document-specific properties
    content: str = Field(..., description="Document content")
    document_type: str = Field(..., description="Document type (pdf, txt, md, etc.)")
    file_path: Optional[str] = Field(None, description="Original file path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages")
    
    # Content analysis
    word_count: int = Field(default=0, description="Word count")
    language: str = Field(default="en", description="Document language")
    
    # Processing metadata
    processing_status: str = Field(default="pending", description="Processing status")
    chunk_count: int = Field(default=0, description="Number of chunks")
    entity_count: int = Field(default=0, description="Number of extracted entities")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with document-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "content": self.content,
            "document_type": self.document_type,
            "word_count": self.word_count,
            "language": self.language,
            "processing_status": self.processing_status,
            "chunk_count": self.chunk_count,
            "entity_count": self.entity_count
        })
        
        if self.file_path:
            props["file_path"] = self.file_path
        if self.file_size:
            props["file_size"] = self.file_size
        if self.page_count:
            props["page_count"] = self.page_count
        
        return props


class ChunkNode(BaseNode):
    """Chunk node representing a portion of a document."""
    
    node_type: NodeType = Field(default=NodeType.CHUNK, const=True)
    
    # Chunk-specific properties
    content: str = Field(..., description="Chunk content")
    chunk_index: int = Field(..., description="Chunk index within document")
    start_position: int = Field(..., description="Start position in document")
    end_position: int = Field(..., description="End position in document")
    
    # Content analysis
    word_count: int = Field(default=0, description="Word count")
    sentence_count: int = Field(default=0, description="Sentence count")
    
    # Quality metrics
    completeness_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Chunk completeness")
    coherence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Content coherence")
    
    # References
    document_id: str = Field(..., description="Parent document ID")
    section_title: Optional[str] = Field(None, description="Section title")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with chunk-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "completeness_score": self.completeness_score,
            "coherence_score": self.coherence_score,
            "document_id": self.document_id
        })
        
        if self.section_title:
            props["section_title"] = self.section_title
        
        return props


class EntityNode(BaseNode):
    """Entity node representing a named entity."""
    
    node_type: NodeType = Field(default=NodeType.ENTITY, const=True)
    
    # Entity-specific properties
    entity_type: str = Field(..., description="Entity type (PERSON, ORG, TECH, etc.)")
    aliases: List[str] = Field(default_factory=list, description="Entity aliases")
    canonical_name: str = Field(..., description="Canonical entity name")
    
    # Entity attributes
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Entity attributes")
    context: str = Field(default="", description="Entity context")
    
    # Frequency and importance
    mention_count: int = Field(default=1, description="Number of mentions")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Entity importance")
    
    # Relationships
    related_entities: List[str] = Field(default_factory=list, description="Related entity IDs")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with entity-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "entity_type": self.entity_type,
            "canonical_name": self.canonical_name,
            "context": self.context,
            "mention_count": self.mention_count,
            "importance_score": self.importance_score
        })
        
        if self.aliases:
            props["aliases"] = self.aliases
        if self.attributes:
            props["attributes"] = self.attributes
        if self.related_entities:
            props["related_entities"] = self.related_entities
        
        return props


class ConceptNode(BaseNode):
    """Concept node representing abstract concepts and topics."""
    
    node_type: NodeType = Field(default=NodeType.CONCEPT, const=True)
    
    # Concept-specific properties
    concept_type: str = Field(..., description="Concept type (topic, theme, principle)")
    definition: str = Field(..., description="Concept definition")
    domain: str = Field(..., description="Domain or field")
    
    # Concept relationships
    parent_concepts: List[str] = Field(default_factory=list, description="Parent concept IDs")
    child_concepts: List[str] = Field(default_factory=list, description="Child concept IDs")
    related_concepts: List[str] = Field(default_factory=list, description="Related concept IDs")
    
    # Metrics
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Concept coverage in corpus")
    clarity_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Concept clarity")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with concept-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "concept_type": self.concept_type,
            "definition": self.definition,
            "domain": self.domain,
            "coverage_score": self.coverage_score,
            "clarity_score": self.clarity_score
        })
        
        if self.parent_concepts:
            props["parent_concepts"] = self.parent_concepts
        if self.child_concepts:
            props["child_concepts"] = self.child_concepts
        if self.related_concepts:
            props["related_concepts"] = self.related_concepts
        
        return props


class ControlNode(BaseNode):
    """Control node representing security controls and compliance requirements."""
    
    node_type: NodeType = Field(default=NodeType.CONTROL, const=True)
    
    # Control-specific properties
    control_id: str = Field(..., description="Official control identifier")
    control_family: str = Field(..., description="Control family")
    framework: str = Field(..., description="Framework (NIST, FedRAMP, etc.)")
    
    # Implementation details
    implementation_guidance: str = Field(..., description="Implementation guidance")
    assessment_procedures: List[str] = Field(default_factory=list, description="Assessment procedures")
    
    # Compliance metadata
    compliance_level: str = Field(..., description="Compliance level (Low, Moderate, High)")
    priority: str = Field(default="Medium", description="Implementation priority")
    status: str = Field(default="Not Implemented", description="Implementation status")
    
    # Relationships
    related_controls: List[str] = Field(default_factory=list, description="Related control IDs")
    dependent_controls: List[str] = Field(default_factory=list, description="Dependent control IDs")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with control-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "control_id": self.control_id,
            "control_family": self.control_family,
            "framework": self.framework,
            "implementation_guidance": self.implementation_guidance,
            "compliance_level": self.compliance_level,
            "priority": self.priority,
            "status": self.status
        })
        
        if self.assessment_procedures:
            props["assessment_procedures"] = self.assessment_procedures
        if self.related_controls:
            props["related_controls"] = self.related_controls
        if self.dependent_controls:
            props["dependent_controls"] = self.dependent_controls
        
        return props


class PersonaNode(BaseNode):
    """Persona node representing user personas and roles."""
    
    node_type: NodeType = Field(default=NodeType.PERSONA, const=True)
    
    # Persona-specific properties
    persona_type: str = Field(..., description="Persona type (user, expert, role)")
    role: str = Field(..., description="Professional role")
    expertise_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Expertise level")
    
    # Behavioral characteristics
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk tolerance")
    detail_preference: float = Field(default=0.5, ge=0.0, le=1.0, description="Detail preference")
    technical_depth: float = Field(default=0.5, ge=0.0, le=1.0, description="Technical depth")
    
    # Goals and preferences
    primary_goals: List[str] = Field(default_factory=list, description="Primary goals")
    communication_style: str = Field(default="balanced", description="Communication style")
    decision_factors: List[str] = Field(default_factory=list, description="Decision factors")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with persona-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "persona_type": self.persona_type,
            "role": self.role,
            "expertise_level": self.expertise_level,
            "risk_tolerance": self.risk_tolerance,
            "detail_preference": self.detail_preference,
            "technical_depth": self.technical_depth,
            "communication_style": self.communication_style
        })
        
        if self.primary_goals:
            props["primary_goals"] = self.primary_goals
        if self.decision_factors:
            props["decision_factors"] = self.decision_factors
        
        return props


class ProcessNode(BaseNode):
    """Process node representing business processes and workflows."""
    
    node_type: NodeType = Field(default=NodeType.PROCESS, const=True)
    
    # Process-specific properties
    process_type: str = Field(..., description="Process type (business, technical, compliance)")
    status: str = Field(default="active", description="Process status")
    owner: str = Field(..., description="Process owner")
    
    # Process metrics
    cycle_time: float = Field(default=0.0, description="Average cycle time in minutes")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Process success rate")
    automation_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Automation level")
    
    # Process steps
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Process steps")
    inputs: List[str] = Field(default_factory=list, description="Process inputs")
    outputs: List[str] = Field(default_factory=list, description="Process outputs")
    
    # Compliance and controls
    required_controls: List[str] = Field(default_factory=list, description="Required control IDs")
    compliance_frameworks: List[str] = Field(default_factory=list, description="Applicable frameworks")
    
    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j properties with process-specific fields."""
        props = super().to_neo4j_properties()
        props.update({
            "process_type": self.process_type,
            "status": self.status,
            "owner": self.owner,
            "cycle_time": self.cycle_time,
            "success_rate": self.success_rate,
            "automation_level": self.automation_level
        })
        
        if self.steps:
            props["steps"] = self.steps
        if self.inputs:
            props["inputs"] = self.inputs
        if self.outputs:
            props["outputs"] = self.outputs
        if self.required_controls:
            props["required_controls"] = self.required_controls
        if self.compliance_frameworks:
            props["compliance_frameworks"] = self.compliance_frameworks
        
        return props


# Node factory for creating nodes from data
class NodeFactory:
    """Factory for creating node instances from data."""
    
    NODE_TYPES = {
        NodeType.DOCUMENT: DocumentNode,
        NodeType.CHUNK: ChunkNode,
        NodeType.ENTITY: EntityNode,
        NodeType.CONCEPT: ConceptNode,
        NodeType.CONTROL: ControlNode,
        NodeType.PERSONA: PersonaNode,
        NodeType.PROCESS: ProcessNode
    }
    
    @classmethod
    def create_node(cls, node_type: NodeType, data: Dict[str, Any]) -> BaseNode:
        """Create a node instance from data."""
        node_class = cls.NODE_TYPES.get(node_type)
        if not node_class:
            raise ValueError(f"Unknown node type: {node_type}")
        
        return node_class(**data)
    
    @classmethod
    def from_neo4j_record(cls, record: Dict[str, Any]) -> BaseNode:
        """Create node from Neo4j record."""
        node_type = NodeType(record.get("node_type"))
        
        # Convert Neo4j record to node data
        node_data = record.copy()
        
        # Handle vector embeddings
        if "content_embedding" in record and "content_embedding_model" in record:
            node_data["content_embedding"] = VectorEmbedding(
                vector=record["content_embedding"],
                model=record["content_embedding_model"],
                dimension=len(record["content_embedding"])
            )
        
        if "title_embedding" in record and "title_embedding_model" in record:
            node_data["title_embedding"] = VectorEmbedding(
                vector=record["title_embedding"],
                model=record["title_embedding_model"],
                dimension=len(record["title_embedding"])
            )
        
        # Handle timestamps
        if "created_at" in record:
            node_data["created_at"] = datetime.fromisoformat(record["created_at"])
        if "updated_at" in record:
            node_data["updated_at"] = datetime.fromisoformat(record["updated_at"])
        
        return cls.create_node(node_type, node_data)