"""
Schema validator for Neo4j Vector Graph Schema.

Provides comprehensive validation for graph schema integrity,
data quality, and performance optimization recommendations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import re

import structlog
from neo4j import AsyncDriver
from neo4j.exceptions import Neo4jError

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import SchemaValidationError, DatabaseError
from kg_rag.graph_schema.node_models import NodeType, VectorEmbedding
from kg_rag.graph_schema.relationship_models import RelationshipType

logger = structlog.get_logger(__name__)


class ValidationSeverity(str):
    """Validation issue severity levels."""
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"


class ValidationIssue:
    """Represents a schema validation issue."""
    
    def __init__(
        self,
        severity: ValidationSeverity,
        category: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        recommendations: Optional[List[str]] = None
    ):
        self.severity = severity
        self.category = category
        self.message = message
        self.details = details or {}
        self.recommendations = recommendations or []
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat()
        }


class SchemaValidator:
    """Comprehensive schema validator for Neo4j vector graph."""
    
    def __init__(self, driver: AsyncDriver):
        """Initialize schema validator.
        
        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
        self.settings = get_settings()
        self.issues: List[ValidationIssue] = []
        
    async def validate_complete_schema(self) -> Dict[str, Any]:
        """Perform comprehensive schema validation.
        
        Returns:
            Dict with complete validation results
        """
        logger.info("Starting comprehensive schema validation")
        
        self.issues = []
        
        validation_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "validation_id": f"validation_{int(datetime.utcnow().timestamp())}",
            "overall_health": "unknown",
            "categories": {
                "schema_structure": {"status": "unknown", "issues": []},
                "data_integrity": {"status": "unknown", "issues": []},
                "performance": {"status": "unknown", "issues": []},
                "vector_embeddings": {"status": "unknown", "issues": []},
                "relationships": {"status": "unknown", "issues": []},
                "compliance": {"status": "unknown", "issues": []}
            },
            "statistics": {},
            "recommendations": []
        }
        
        try:
            # Run all validation checks
            await self._validate_schema_structure()
            await self._validate_data_integrity()
            await self._validate_performance()
            await self._validate_vector_embeddings()
            await self._validate_relationships()
            await self._validate_compliance()
            
            # Categorize issues
            for issue in self.issues:
                if issue.category in validation_result["categories"]:
                    validation_result["categories"][issue.category]["issues"].append(issue.to_dict())
            
            # Determine category statuses
            for category_name, category_data in validation_result["categories"].items():
                category_issues = category_data["issues"]
                
                if any(issue["severity"] == ValidationSeverity.CRITICAL for issue in category_issues):
                    category_data["status"] = "critical"
                elif any(issue["severity"] == ValidationSeverity.WARNING for issue in category_issues):
                    category_data["status"] = "warning"
                else:
                    category_data["status"] = "healthy"
            
            # Determine overall health
            critical_categories = [
                cat for cat, data in validation_result["categories"].items()
                if data["status"] == "critical"
            ]
            warning_categories = [
                cat for cat, data in validation_result["categories"].items()
                if data["status"] == "warning"
            ]
            
            if critical_categories:
                validation_result["overall_health"] = "critical"
            elif warning_categories:
                validation_result["overall_health"] = "warning"
            else:
                validation_result["overall_health"] = "healthy"
            
            # Collect statistics
            validation_result["statistics"] = await self._collect_validation_statistics()
            
            # Generate recommendations
            validation_result["recommendations"] = self._generate_recommendations()
            
            logger.info(
                "Schema validation completed",
                overall_health=validation_result["overall_health"],
                total_issues=len(self.issues),
                critical_issues=len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]),
                warning_issues=len([i for i in self.issues if i.severity == ValidationSeverity.WARNING])
            )
            
        except Exception as e:
            error_msg = f"Schema validation failed: {str(e)}"
            logger.error(error_msg, error=str(e))
            validation_result["overall_health"] = "error"
            validation_result["error"] = error_msg
            
        return validation_result
    
    async def _validate_schema_structure(self):
        """Validate basic schema structure."""
        logger.debug("Validating schema structure")
        
        try:
            async with self.driver.session() as session:
                # Check for required constraints
                result = await session.run("SHOW CONSTRAINTS")
                constraints = [record["name"] async for record in result]
                
                required_constraints = [
                    "unique_node_id",
                    "unique_document_id", 
                    "unique_entity_id",
                    "unique_relationship_id"
                ]
                
                missing_constraints = [
                    constraint for constraint in required_constraints
                    if constraint not in constraints
                ]
                
                if missing_constraints:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="schema_structure",
                        message="Missing required constraints",
                        details={"missing_constraints": missing_constraints},
                        recommendations=[
                            "Run schema initialization to create missing constraints",
                            "Ensure proper unique identifiers for all nodes and relationships"
                        ]
                    ))
                
                # Check for required indexes
                result = await session.run("SHOW INDEXES")
                indexes = [record["name"] async for record in result]
                
                required_indexes = [
                    "node_type_index",
                    "entity_type_index",
                    "relationship_type_index"
                ]
                
                missing_indexes = [
                    index for index in required_indexes
                    if index not in indexes
                ]
                
                if missing_indexes:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="schema_structure",
                        message="Missing recommended indexes",
                        details={"missing_indexes": missing_indexes},
                        recommendations=[
                            "Create missing indexes to improve query performance",
                            "Monitor query performance for slow operations"
                        ]
                    ))
                
                # Check vector indexes
                vector_indexes = [idx for idx in indexes if "embedding" in idx.lower()]
                expected_vector_indexes = [
                    "document_content_embeddings",
                    "chunk_content_embeddings",
                    "entity_content_embeddings"
                ]
                
                missing_vector_indexes = [
                    idx for idx in expected_vector_indexes
                    if idx not in indexes
                ]
                
                if missing_vector_indexes:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="schema_structure", 
                        message="Missing vector indexes",
                        details={"missing_vector_indexes": missing_vector_indexes},
                        recommendations=[
                            "Create vector indexes for embedding fields",
                            "Enable vector similarity search capabilities"
                        ]
                    ))
                    
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="schema_structure",
                message=f"Failed to validate schema structure: {str(e)}",
                recommendations=["Check database connectivity and permissions"]
            ))
    
    async def _validate_data_integrity(self):
        """Validate data integrity."""
        logger.debug("Validating data integrity")
        
        try:
            async with self.driver.session() as session:
                # Check for orphaned nodes
                result = await session.run("""
                    MATCH (n)
                    WHERE NOT (n)--()
                    RETURN count(n) as orphaned_count, labels(n) as node_types
                """)
                
                async for record in result:
                    orphaned_count = record["orphaned_count"]
                    if orphaned_count > 0:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category="data_integrity",
                            message=f"Found {orphaned_count} orphaned nodes",
                            details={"orphaned_count": orphaned_count},
                            recommendations=[
                                "Review orphaned nodes for cleanup",
                                "Ensure proper relationship creation"
                            ]
                        ))
                
                # Check for missing required fields
                for node_type in NodeType:
                    result = await session.run(f"""
                        MATCH (n:{node_type.value})
                        WHERE n.node_id IS NULL OR n.title IS NULL OR n.node_type IS NULL
                        RETURN count(n) as invalid_count
                    """)
                    
                    record = await result.single()
                    if record and record["invalid_count"] > 0:
                        self.issues.append(ValidationIssue(
                            severity=ValidationSeverity.CRITICAL,
                            category="data_integrity",
                            message=f"Found {node_type.value} nodes with missing required fields",
                            details={
                                "node_type": node_type.value,
                                "invalid_count": record["invalid_count"]
                            },
                            recommendations=[
                                "Update nodes to include all required fields",
                                "Implement data validation before node creation"
                            ]
                        ))
                
                # Check for duplicate node IDs
                result = await session.run("""
                    MATCH (n)
                    WITH n.node_id as node_id, count(n) as count
                    WHERE count > 1
                    RETURN node_id, count
                    LIMIT 10
                """)
                
                duplicates = [(record["node_id"], record["count"]) async for record in result]
                if duplicates:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="data_integrity",
                        message="Found duplicate node IDs",
                        details={"duplicates": duplicates},
                        recommendations=[
                            "Remove or merge duplicate nodes",
                            "Ensure unique ID generation process"
                        ]
                    ))
                
                # Check relationship consistency
                result = await session.run("""
                    MATCH ()-[r]-()
                    WHERE r.relationship_id IS NULL OR r.relationship_type IS NULL
                    RETURN count(r) as invalid_rel_count
                """)
                
                record = await result.single()
                if record and record["invalid_rel_count"] > 0:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="data_integrity",
                        message=f"Found {record['invalid_rel_count']} relationships with missing required fields",
                        recommendations=[
                            "Update relationships to include required fields",
                            "Implement relationship validation"
                        ]
                    ))
                    
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="data_integrity",
                message=f"Failed to validate data integrity: {str(e)}",
                recommendations=["Check database connectivity and permissions"]
            ))
    
    async def _validate_performance(self):
        """Validate performance characteristics."""
        logger.debug("Validating performance")
        
        try:
            async with self.driver.session() as session:
                # Check node counts for performance planning
                result = await session.run("MATCH (n) RETURN count(n) as total_nodes")
                record = await result.single()
                total_nodes = record["total_nodes"] if record else 0
                
                if total_nodes > 1000000:  # 1M nodes
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        message=f"Large number of nodes ({total_nodes:,}) may impact performance",
                        details={"total_nodes": total_nodes},
                        recommendations=[
                            "Consider graph partitioning strategies",
                            "Monitor query performance",
                            "Implement query optimization"
                        ]
                    ))
                
                # Check relationship counts
                result = await session.run("MATCH ()-[r]-() RETURN count(r) as total_relationships")
                record = await result.single()
                total_relationships = record["total_relationships"] if record else 0
                
                if total_relationships > 5000000:  # 5M relationships
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        message=f"Large number of relationships ({total_relationships:,}) may impact performance",
                        details={"total_relationships": total_relationships},
                        recommendations=[
                            "Consider relationship type optimization",
                            "Implement query result limiting",
                            "Monitor memory usage"
                        ]
                    ))
                
                # Check for nodes with excessive relationships
                result = await session.run("""
                    MATCH (n)
                    WITH n, count{(n)--()--()} as degree
                    WHERE degree > 1000
                    RETURN n.node_id as node_id, n.node_type as node_type, degree
                    ORDER BY degree DESC
                    LIMIT 10
                """)
                
                high_degree_nodes = [
                    {
                        "node_id": record["node_id"],
                        "node_type": record["node_type"],
                        "degree": record["degree"]
                    }
                    async for record in result
                ]
                
                if high_degree_nodes:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="performance",
                        message="Found nodes with very high connectivity",
                        details={"high_degree_nodes": high_degree_nodes},
                        recommendations=[
                            "Consider breaking down highly connected nodes",
                            "Implement relationship filtering",
                            "Monitor query performance for these nodes"
                        ]
                    ))
                    
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="performance",
                message=f"Failed to validate performance: {str(e)}",
                recommendations=["Check database connectivity"]
            ))
    
    async def _validate_vector_embeddings(self):
        """Validate vector embeddings."""
        logger.debug("Validating vector embeddings")
        
        expected_dimension = self.settings.ai_models.embedding_dimension
        
        try:
            async with self.driver.session() as session:
                # Check embedding dimensions
                for node_type in [NodeType.DOCUMENT, NodeType.CHUNK, NodeType.ENTITY, NodeType.CONCEPT]:
                    for embedding_field in ["content_embedding", "title_embedding"]:
                        result = await session.run(f"""
                            MATCH (n:{node_type.value})
                            WHERE n.{embedding_field} IS NOT NULL
                            WITH n.{embedding_field} as embedding
                            WHERE size(embedding) <> $expected_dimension
                            RETURN count(*) as invalid_count, 
                                   collect(size(embedding))[0..5] as sample_dimensions
                        """, expected_dimension=expected_dimension)
                        
                        record = await result.single()
                        if record and record["invalid_count"] > 0:
                            self.issues.append(ValidationIssue(
                                severity=ValidationSeverity.CRITICAL,
                                category="vector_embeddings",
                                message=f"Found {node_type.value} nodes with incorrect {embedding_field} dimensions",
                                details={
                                    "node_type": node_type.value,
                                    "embedding_field": embedding_field,
                                    "invalid_count": record["invalid_count"],
                                    "expected_dimension": expected_dimension,
                                    "sample_dimensions": record["sample_dimensions"]
                                },
                                recommendations=[
                                    "Re-generate embeddings with correct dimensions",
                                    "Verify embedding model configuration"
                                ]
                            ))
                
                # Check for missing embeddings
                for node_type in [NodeType.DOCUMENT, NodeType.CHUNK, NodeType.ENTITY]:
                    result = await session.run(f"""
                        MATCH (n:{node_type.value})
                        WHERE n.content_embedding IS NULL
                        RETURN count(n) as missing_count
                    """)
                    
                    record = await result.single()
                    if record and record["missing_count"] > 0:
                        # Determine severity based on node type
                        severity = ValidationSeverity.WARNING
                        if node_type in [NodeType.DOCUMENT, NodeType.CHUNK]:
                            severity = ValidationSeverity.CRITICAL
                        
                        self.issues.append(ValidationIssue(
                            severity=severity,
                            category="vector_embeddings",
                            message=f"Found {node_type.value} nodes without content embeddings",
                            details={
                                "node_type": node_type.value,
                                "missing_count": record["missing_count"]
                            },
                            recommendations=[
                                "Generate embeddings for all content nodes",
                                "Implement embedding pipeline for new content"
                            ]
                        ))
                
                # Check embedding quality (zero vectors)
                result = await session.run("""
                    MATCH (n)
                    WHERE n.content_embedding IS NOT NULL
                    WITH n, reduce(sum = 0, x IN n.content_embedding | sum + x*x) as magnitude
                    WHERE magnitude = 0
                    RETURN count(n) as zero_vector_count
                """)
                
                record = await result.single()
                if record and record["zero_vector_count"] > 0:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="vector_embeddings",
                        message=f"Found {record['zero_vector_count']} nodes with zero embeddings",
                        details={"zero_vector_count": record["zero_vector_count"]},
                        recommendations=[
                            "Investigate zero embedding generation",
                            "Re-process content with zero embeddings"
                        ]
                    ))
                    
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                category="vector_embeddings",
                message=f"Failed to validate vector embeddings: {str(e)}",
                recommendations=["Check database connectivity and embedding field names"]
            ))
    
    async def _validate_relationships(self):
        """Validate relationship integrity."""
        logger.debug("Validating relationships")
        
        try:
            async with self.driver.session() as session:
                # Check for invalid relationship weights
                result = await session.run("""
                    MATCH ()-[r]-()
                    WHERE r.weight < 0 OR r.weight > 1 OR r.confidence < 0 OR r.confidence > 1
                    RETURN count(r) as invalid_score_count
                """)
                
                record = await result.single()
                if record and record["invalid_score_count"] > 0:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="relationships",
                        message=f"Found {record['invalid_score_count']} relationships with invalid scores",
                        details={"invalid_score_count": record["invalid_score_count"]},
                        recommendations=[
                            "Update relationship scores to be between 0 and 1",
                            "Implement score validation in relationship creation"
                        ]
                    ))
                
                # Check for unidirectional relationships that should be bidirectional
                result = await session.run("""
                    MATCH (a)-[r:SIMILAR_TO]->(b)
                    WHERE NOT (b)-[:SIMILAR_TO]->(a)
                    RETURN count(r) as unidirectional_similarity_count
                """)
                
                record = await result.single()
                if record and record["unidirectional_similarity_count"] > 0:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="relationships",
                        message=f"Found {record['unidirectional_similarity_count']} unidirectional similarity relationships",
                        details={"unidirectional_count": record["unidirectional_similarity_count"]},
                        recommendations=[
                            "Create bidirectional similarity relationships",
                            "Review similarity relationship creation logic"
                        ]
                    ))
                    
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="relationships",
                message=f"Failed to validate relationships: {str(e)}",
                recommendations=["Check database connectivity"]
            ))
    
    async def _validate_compliance(self):
        """Validate compliance-specific requirements."""
        logger.debug("Validating compliance requirements")
        
        try:
            async with self.driver.session() as session:
                # Check for controls without implementation status
                result = await session.run("""
                    MATCH (c:Control)
                    WHERE c.status IS NULL OR c.status = ""
                    RETURN count(c) as missing_status_count
                """)
                
                record = await result.single()
                if record and record["missing_status_count"] > 0:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category="compliance",
                        message=f"Found {record['missing_status_count']} controls without implementation status",
                        details={"missing_status_count": record["missing_status_count"]},
                        recommendations=[
                            "Update all controls with implementation status",
                            "Implement status tracking for compliance controls"
                        ]
                    ))
                
                # Check for high-priority controls that are not implemented
                result = await session.run("""
                    MATCH (c:Control)
                    WHERE c.priority IN ["High", "Critical"] AND c.status = "Not Implemented"
                    RETURN count(c) as high_priority_unimplemented
                """)
                
                record = await result.single()
                if record and record["high_priority_unimplemented"] > 0:
                    self.issues.append(ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        category="compliance",
                        message=f"Found {record['high_priority_unimplemented']} high-priority unimplemented controls",
                        details={"high_priority_unimplemented": record["high_priority_unimplemented"]},
                        recommendations=[
                            "Prioritize implementation of high-priority controls",
                            "Create implementation roadmap",
                            "Assign ownership for control implementation"
                        ]
                    ))
                    
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="compliance",
                message=f"Failed to validate compliance requirements: {str(e)}",
                recommendations=["Check database connectivity and control node structure"]
            ))
    
    async def _collect_validation_statistics(self) -> Dict[str, Any]:
        """Collect validation statistics."""
        statistics = {
            "total_issues": len(self.issues),
            "issues_by_severity": {
                "critical": len([i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]),
                "warning": len([i for i in self.issues if i.severity == ValidationSeverity.WARNING]),
                "info": len([i for i in self.issues if i.severity == ValidationSeverity.INFO])
            },
            "issues_by_category": {}
        }
        
        # Count issues by category
        for issue in self.issues:
            if issue.category not in statistics["issues_by_category"]:
                statistics["issues_by_category"][issue.category] = 0
            statistics["issues_by_category"][issue.category] += 1
        
        return statistics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate high-level recommendations based on validation results."""
        recommendations = []
        
        # Critical issues recommendations
        critical_issues = [i for i in self.issues if i.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            recommendations.append("Address critical issues immediately to ensure system stability")
            
            # Schema structure issues
            if any(i.category == "schema_structure" for i in critical_issues):
                recommendations.append("Run schema initialization to create missing constraints and indexes")
            
            # Data integrity issues
            if any(i.category == "data_integrity" for i in critical_issues):
                recommendations.append("Implement data validation and cleanup procedures")
            
            # Vector embedding issues
            if any(i.category == "vector_embeddings" for i in critical_issues):
                recommendations.append("Re-generate embeddings with correct dimensions and validate embedding pipeline")
        
        # Performance recommendations
        performance_issues = [i for i in self.issues if i.category == "performance"]
        if performance_issues:
            recommendations.append("Monitor and optimize query performance for large datasets")
        
        # General recommendations
        if len(self.issues) > 10:
            recommendations.append("Consider implementing automated validation checks in CI/CD pipeline")
        
        return recommendations