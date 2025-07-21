"""
Neo4j Vector Graph Schema for Knowledge Graph-RAG.

Provides comprehensive graph schema with vector embeddings support,
hybrid search capabilities, and knowledge graph operations.
"""

from kg_rag.graph_schema.schema_manager import GraphSchemaManager
from kg_rag.graph_schema.node_models import (
    BaseNode, DocumentNode, ChunkNode, EntityNode, ConceptNode,
    ControlNode, PersonaNode, ProcessNode
)
from kg_rag.graph_schema.relationship_models import (
    BaseRelationship, ContainsRelationship, ReferencesRelationship,
    RelatedToRelationship, ImplementsRelationship, CompliesToRelationship,
    DependsOnRelationship, InfluencesRelationship
)
from kg_rag.graph_schema.vector_operations import VectorGraphOperations
from kg_rag.graph_schema.query_builder import GraphQueryBuilder
from kg_rag.graph_schema.schema_validator import SchemaValidator

__all__ = [
    # Schema Management
    "GraphSchemaManager",
    "SchemaValidator",
    
    # Node Models
    "BaseNode",
    "DocumentNode", 
    "ChunkNode",
    "EntityNode",
    "ConceptNode",
    "ControlNode",
    "PersonaNode",
    "ProcessNode",
    
    # Relationship Models
    "BaseRelationship",
    "ContainsRelationship",
    "ReferencesRelationship", 
    "RelatedToRelationship",
    "ImplementsRelationship",
    "CompliesToRelationship",
    "DependsOnRelationship",
    "InfluencesRelationship",
    
    # Operations
    "VectorGraphOperations",
    "GraphQueryBuilder"
]