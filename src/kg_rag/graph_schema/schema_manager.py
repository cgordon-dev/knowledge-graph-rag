"""
Schema manager for Neo4j Vector Graph Schema.

Handles schema creation, validation, management, and migration operations
for the knowledge graph with vector embeddings support.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import uuid

import structlog
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncResult, AsyncTransaction
from neo4j.exceptions import Neo4jError

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import GraphSchemaError, DatabaseError
from kg_rag.graph_schema.node_models import NodeType, BaseNode, NodeFactory
from kg_rag.graph_schema.relationship_models import RelationshipType, BaseRelationship, RelationshipFactory

logger = structlog.get_logger(__name__)


class GraphSchemaManager:
    """Manages Neo4j graph schema with vector embeddings support."""
    
    def __init__(self, driver: Optional[AsyncDriver] = None):
        """Initialize schema manager.
        
        Args:
            driver: Optional Neo4j driver instance
        """
        self.driver = driver
        self.settings = get_settings()
        self._constraints_created = False
        self._indexes_created = False
        self._vector_indexes_created = False
        
    async def _get_driver(self) -> AsyncDriver:
        """Get or create Neo4j driver."""
        if self.driver is None:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.database.neo4j_uri,
                auth=(self.settings.database.neo4j_user, self.settings.database.neo4j_password)
            )
        return self.driver
    
    async def close(self):
        """Close the database connection."""
        if self.driver:
            await self.driver.close()
    
    async def initialize_schema(self, drop_existing: bool = False) -> Dict[str, Any]:
        """Initialize complete graph schema.
        
        Args:
            drop_existing: Whether to drop existing schema first
            
        Returns:
            Dict with initialization results
        """
        logger.info("Initializing Neo4j vector graph schema", drop_existing=drop_existing)
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "drop_existing": drop_existing,
            "constraints_created": 0,
            "indexes_created": 0,
            "vector_indexes_created": 0,
            "errors": []
        }
        
        try:
            driver = await self._get_driver()
            
            async with driver.session() as session:
                # Drop existing schema if requested
                if drop_existing:
                    await self._drop_schema(session)
                
                # Create constraints
                constraints_result = await self._create_constraints(session)
                results["constraints_created"] = constraints_result["created"]
                results["errors"].extend(constraints_result["errors"])
                
                # Create standard indexes
                indexes_result = await self._create_indexes(session)
                results["indexes_created"] = indexes_result["created"]
                results["errors"].extend(indexes_result["errors"])
                
                # Create vector indexes
                vector_result = await self._create_vector_indexes(session)
                results["vector_indexes_created"] = vector_result["created"]
                results["errors"].extend(vector_result["errors"])
                
                # Update flags
                self._constraints_created = results["constraints_created"] > 0
                self._indexes_created = results["indexes_created"] > 0
                self._vector_indexes_created = results["vector_indexes_created"] > 0
                
                logger.info(
                    "Schema initialization completed",
                    constraints=results["constraints_created"],
                    indexes=results["indexes_created"],
                    vector_indexes=results["vector_indexes_created"],
                    errors=len(results["errors"])
                )
                
        except Exception as e:
            error_msg = f"Schema initialization failed: {str(e)}"
            logger.error(error_msg, error=str(e))
            results["errors"].append(error_msg)
            raise GraphSchemaError(error_msg) from e
        
        return results
    
    async def _drop_schema(self, session) -> None:
        """Drop existing schema elements."""
        logger.warning("Dropping existing schema")
        
        # Drop constraints
        drop_queries = [
            "DROP CONSTRAINT unique_node_id IF EXISTS",
            "DROP CONSTRAINT unique_document_id IF EXISTS",
            "DROP CONSTRAINT unique_chunk_id IF EXISTS",
            "DROP CONSTRAINT unique_entity_id IF EXISTS",
            "DROP CONSTRAINT unique_concept_id IF EXISTS",
            "DROP CONSTRAINT unique_control_id IF EXISTS",
            "DROP CONSTRAINT unique_persona_id IF EXISTS",
            "DROP CONSTRAINT unique_process_id IF EXISTS",
            "DROP CONSTRAINT unique_relationship_id IF EXISTS"
        ]
        
        # Drop indexes
        drop_queries.extend([
            "DROP INDEX node_type_index IF EXISTS",
            "DROP INDEX node_tags_index IF EXISTS",
            "DROP INDEX node_categories_index IF EXISTS",
            "DROP INDEX document_type_index IF EXISTS",
            "DROP INDEX entity_type_index IF EXISTS",
            "DROP INDEX control_framework_index IF EXISTS",
            "DROP INDEX relationship_type_index IF EXISTS"
        ])
        
        # Drop vector indexes
        drop_queries.extend([
            "DROP INDEX document_content_embeddings IF EXISTS",
            "DROP INDEX document_title_embeddings IF EXISTS",
            "DROP INDEX chunk_content_embeddings IF EXISTS",
            "DROP INDEX chunk_title_embeddings IF EXISTS",
            "DROP INDEX entity_content_embeddings IF EXISTS",
            "DROP INDEX entity_title_embeddings IF EXISTS",
            "DROP INDEX concept_content_embeddings IF EXISTS",
            "DROP INDEX concept_title_embeddings IF EXISTS"
        ])
        
        for query in drop_queries:
            try:
                await session.run(query)
            except Neo4jError as e:
                # Some drops might fail if objects don't exist
                logger.debug(f"Drop query failed (expected): {query}", error=str(e))
    
    async def _create_constraints(self, session) -> Dict[str, Any]:
        """Create schema constraints."""
        logger.info("Creating schema constraints")
        
        constraints = [
            # Node ID uniqueness constraints
            ("unique_node_id", "CREATE CONSTRAINT unique_node_id FOR (n:BaseNode) REQUIRE n.node_id IS UNIQUE"),
            ("unique_document_id", "CREATE CONSTRAINT unique_document_id FOR (n:Document) REQUIRE n.node_id IS UNIQUE"),
            ("unique_chunk_id", "CREATE CONSTRAINT unique_chunk_id FOR (n:Chunk) REQUIRE n.node_id IS UNIQUE"),
            ("unique_entity_id", "CREATE CONSTRAINT unique_entity_id FOR (n:Entity) REQUIRE n.node_id IS UNIQUE"),
            ("unique_concept_id", "CREATE CONSTRAINT unique_concept_id FOR (n:Concept) REQUIRE n.node_id IS UNIQUE"),
            ("unique_control_id", "CREATE CONSTRAINT unique_control_id FOR (n:Control) REQUIRE n.node_id IS UNIQUE"),
            ("unique_persona_id", "CREATE CONSTRAINT unique_persona_id FOR (n:Persona) REQUIRE n.node_id IS UNIQUE"),
            ("unique_process_id", "CREATE CONSTRAINT unique_process_id FOR (n:Process) REQUIRE n.node_id IS UNIQUE"),
            
            # Relationship ID uniqueness
            ("unique_relationship_id", "CREATE CONSTRAINT unique_relationship_id FOR ()-[r]-() REQUIRE r.relationship_id IS UNIQUE"),
            
            # Required fields constraints
            ("node_title_required", "CREATE CONSTRAINT node_title_required FOR (n:BaseNode) REQUIRE n.title IS NOT NULL"),
            ("node_type_required", "CREATE CONSTRAINT node_type_required FOR (n:BaseNode) REQUIRE n.node_type IS NOT NULL"),
            
            # Control-specific constraints
            ("control_id_required", "CREATE CONSTRAINT control_id_required FOR (n:Control) REQUIRE n.control_id IS NOT NULL"),
            ("control_framework_required", "CREATE CONSTRAINT control_framework_required FOR (n:Control) REQUIRE n.framework IS NOT NULL"),
            
            # Entity-specific constraints
            ("entity_type_required", "CREATE CONSTRAINT entity_type_required FOR (n:Entity) REQUIRE n.entity_type IS NOT NULL"),
            ("canonical_name_required", "CREATE CONSTRAINT canonical_name_required FOR (n:Entity) REQUIRE n.canonical_name IS NOT NULL")
        ]
        
        created = 0
        errors = []
        
        for constraint_name, query in constraints:
            try:
                await session.run(query)
                created += 1
                logger.debug(f"Created constraint: {constraint_name}")
            except Neo4jError as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Constraint already exists: {constraint_name}")
                else:
                    error_msg = f"Failed to create constraint {constraint_name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        return {"created": created, "errors": errors}
    
    async def _create_indexes(self, session) -> Dict[str, Any]:
        """Create standard indexes for performance."""
        logger.info("Creating standard indexes")
        
        indexes = [
            # Node type indexes
            ("node_type_index", "CREATE INDEX node_type_index FOR (n:BaseNode) ON (n.node_type)"),
            ("node_tags_index", "CREATE INDEX node_tags_index FOR (n:BaseNode) ON (n.tags)"),
            ("node_categories_index", "CREATE INDEX node_categories_index FOR (n:BaseNode) ON (n.categories)"),
            ("node_created_at_index", "CREATE INDEX node_created_at_index FOR (n:BaseNode) ON (n.created_at)"),
            ("node_source_index", "CREATE INDEX node_source_index FOR (n:BaseNode) ON (n.source)"),
            
            # Document-specific indexes
            ("document_type_index", "CREATE INDEX document_type_index FOR (n:Document) ON (n.document_type)"),
            ("document_language_index", "CREATE INDEX document_language_index FOR (n:Document) ON (n.language)"),
            ("document_status_index", "CREATE INDEX document_status_index FOR (n:Document) ON (n.processing_status)"),
            
            # Chunk-specific indexes
            ("chunk_document_index", "CREATE INDEX chunk_document_index FOR (n:Chunk) ON (n.document_id)"),
            ("chunk_index_index", "CREATE INDEX chunk_index_index FOR (n:Chunk) ON (n.chunk_index)"),
            
            # Entity-specific indexes
            ("entity_type_index", "CREATE INDEX entity_type_index FOR (n:Entity) ON (n.entity_type)"),
            ("entity_canonical_index", "CREATE INDEX entity_canonical_index FOR (n:Entity) ON (n.canonical_name)"),
            ("entity_importance_index", "CREATE INDEX entity_importance_index FOR (n:Entity) ON (n.importance_score)"),
            
            # Concept-specific indexes
            ("concept_type_index", "CREATE INDEX concept_type_index FOR (n:Concept) ON (n.concept_type)"),
            ("concept_domain_index", "CREATE INDEX concept_domain_index FOR (n:Concept) ON (n.domain)"),
            
            # Control-specific indexes
            ("control_framework_index", "CREATE INDEX control_framework_index FOR (n:Control) ON (n.framework)"),
            ("control_family_index", "CREATE INDEX control_family_index FOR (n:Control) ON (n.control_family)"),
            ("control_status_index", "CREATE INDEX control_status_index FOR (n:Control) ON (n.status)"),
            ("control_priority_index", "CREATE INDEX control_priority_index FOR (n:Control) ON (n.priority)"),
            
            # Persona-specific indexes
            ("persona_type_index", "CREATE INDEX persona_type_index FOR (n:Persona) ON (n.persona_type)"),
            ("persona_role_index", "CREATE INDEX persona_role_index FOR (n:Persona) ON (n.role)"),
            
            # Process-specific indexes
            ("process_type_index", "CREATE INDEX process_type_index FOR (n:Process) ON (n.process_type)"),
            ("process_owner_index", "CREATE INDEX process_owner_index FOR (n:Process) ON (n.owner)"),
            ("process_status_index", "CREATE INDEX process_status_index FOR (n:Process) ON (n.status)"),
            
            # Relationship indexes
            ("relationship_type_index", "CREATE INDEX relationship_type_index FOR ()-[r]-() ON (r.relationship_type)"),
            ("relationship_weight_index", "CREATE INDEX relationship_weight_index FOR ()-[r]-() ON (r.weight)"),
            ("relationship_confidence_index", "CREATE INDEX relationship_confidence_index FOR ()-[r]-() ON (r.confidence)")
        ]
        
        created = 0
        errors = []
        
        for index_name, query in indexes:
            try:
                await session.run(query)
                created += 1
                logger.debug(f"Created index: {index_name}")
            except Neo4jError as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Index already exists: {index_name}")
                else:
                    error_msg = f"Failed to create index {index_name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        return {"created": created, "errors": errors}
    
    async def _create_vector_indexes(self, session) -> Dict[str, Any]:
        """Create vector indexes for embeddings."""
        logger.info("Creating vector indexes")
        
        # Get vector dimensions from settings
        vector_dim = self.settings.ai_models.embedding_dimension
        
        vector_indexes = [
            # Document vector indexes
            ("document_content_embeddings", f"""
                CREATE VECTOR INDEX document_content_embeddings
                FOR (n:Document) ON (n.content_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """),
            ("document_title_embeddings", f"""
                CREATE VECTOR INDEX document_title_embeddings
                FOR (n:Document) ON (n.title_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """),
            
            # Chunk vector indexes
            ("chunk_content_embeddings", f"""
                CREATE VECTOR INDEX chunk_content_embeddings
                FOR (n:Chunk) ON (n.content_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """),
            ("chunk_title_embeddings", f"""
                CREATE VECTOR INDEX chunk_title_embeddings
                FOR (n:Chunk) ON (n.title_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """),
            
            # Entity vector indexes
            ("entity_content_embeddings", f"""
                CREATE VECTOR INDEX entity_content_embeddings
                FOR (n:Entity) ON (n.content_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """),
            ("entity_title_embeddings", f"""
                CREATE VECTOR INDEX entity_title_embeddings
                FOR (n:Entity) ON (n.title_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """),
            
            # Concept vector indexes
            ("concept_content_embeddings", f"""
                CREATE VECTOR INDEX concept_content_embeddings
                FOR (n:Concept) ON (n.content_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """),
            ("concept_title_embeddings", f"""
                CREATE VECTOR INDEX concept_title_embeddings
                FOR (n:Concept) ON (n.title_embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {vector_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)
        ]
        
        created = 0
        errors = []
        
        for index_name, query in vector_indexes:
            try:
                # Clean up the query
                clean_query = " ".join(query.strip().split())
                await session.run(clean_query)
                created += 1
                logger.debug(f"Created vector index: {index_name}")
            except Neo4jError as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Vector index already exists: {index_name}")
                else:
                    error_msg = f"Failed to create vector index {index_name}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        return {"created": created, "errors": errors}
    
    async def validate_schema(self) -> Dict[str, Any]:
        """Validate current schema state."""
        logger.info("Validating schema state")
        
        validation_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "is_valid": True,
            "constraints": {
                "expected": 0,
                "found": 0,
                "missing": []
            },
            "indexes": {
                "expected": 0,
                "found": 0,
                "missing": []
            },
            "vector_indexes": {
                "expected": 0,
                "found": 0,
                "missing": []
            },
            "errors": []
        }
        
        try:
            driver = await self._get_driver()
            
            async with driver.session() as session:
                # Check constraints
                constraints_result = await session.run("SHOW CONSTRAINTS")
                found_constraints = [record["name"] async for record in constraints_result]
                
                expected_constraints = [
                    "unique_node_id", "unique_document_id", "unique_chunk_id",
                    "unique_entity_id", "unique_concept_id", "unique_control_id",
                    "unique_persona_id", "unique_process_id", "unique_relationship_id"
                ]
                
                validation_result["constraints"]["expected"] = len(expected_constraints)
                validation_result["constraints"]["found"] = len(found_constraints)
                validation_result["constraints"]["missing"] = [
                    c for c in expected_constraints if c not in found_constraints
                ]
                
                # Check indexes
                indexes_result = await session.run("SHOW INDEXES")
                found_indexes = [record["name"] async for record in indexes_result]
                
                expected_indexes = [
                    "node_type_index", "document_type_index", "entity_type_index",
                    "control_framework_index", "relationship_type_index"
                ]
                
                validation_result["indexes"]["expected"] = len(expected_indexes)
                validation_result["indexes"]["found"] = len([i for i in found_indexes if i in expected_indexes])
                validation_result["indexes"]["missing"] = [
                    i for i in expected_indexes if i not in found_indexes
                ]
                
                # Check vector indexes
                vector_indexes = [
                    i for i in found_indexes if "embeddings" in i.lower()
                ]
                
                expected_vector_indexes = [
                    "document_content_embeddings", "document_title_embeddings",
                    "chunk_content_embeddings", "chunk_title_embeddings",
                    "entity_content_embeddings", "entity_title_embeddings"
                ]
                
                validation_result["vector_indexes"]["expected"] = len(expected_vector_indexes)
                validation_result["vector_indexes"]["found"] = len(vector_indexes)
                validation_result["vector_indexes"]["missing"] = [
                    i for i in expected_vector_indexes if i not in found_indexes
                ]
                
                # Determine overall validity
                validation_result["is_valid"] = (
                    len(validation_result["constraints"]["missing"]) == 0 and
                    len(validation_result["indexes"]["missing"]) == 0 and
                    len(validation_result["vector_indexes"]["missing"]) == 0
                )
                
                logger.info(
                    "Schema validation completed",
                    is_valid=validation_result["is_valid"],
                    constraints_missing=len(validation_result["constraints"]["missing"]),
                    indexes_missing=len(validation_result["indexes"]["missing"]),
                    vector_indexes_missing=len(validation_result["vector_indexes"]["missing"])
                )
                
        except Exception as e:
            error_msg = f"Schema validation failed: {str(e)}"
            logger.error(error_msg, error=str(e))
            validation_result["errors"].append(error_msg)
            validation_result["is_valid"] = False
        
        return validation_result
    
    async def create_node(self, node: BaseNode) -> Dict[str, Any]:
        """Create a node in the graph.
        
        Args:
            node: Node instance to create
            
        Returns:
            Dict with creation result
        """
        try:
            driver = await self._get_driver()
            
            async with driver.session() as session:
                # Generate Cypher query
                query = node.get_cypher_create_query()
                properties = node.to_neo4j_properties()
                
                # Execute query
                result = await session.run(query, **properties)
                summary = await result.consume()
                
                logger.info(
                    "Node created successfully",
                    node_id=node.node_id,
                    node_type=node.node_type,
                    nodes_created=summary.counters.nodes_created
                )
                
                return {
                    "success": True,
                    "node_id": node.node_id,
                    "nodes_created": summary.counters.nodes_created,
                    "properties_set": summary.counters.properties_set
                }
                
        except Exception as e:
            error_msg = f"Failed to create node {node.node_id}: {str(e)}"
            logger.error(error_msg, node_id=node.node_id, error=str(e))
            raise GraphSchemaError(error_msg) from e
    
    async def create_relationship(self, relationship: BaseRelationship) -> Dict[str, Any]:
        """Create a relationship in the graph.
        
        Args:
            relationship: Relationship instance to create
            
        Returns:
            Dict with creation result
        """
        try:
            driver = await self._get_driver()
            
            async with driver.session() as session:
                # Generate Cypher query
                query = relationship.get_cypher_create_query()
                properties = relationship.to_neo4j_properties()
                
                # Add source and target node IDs
                properties.update({
                    "source_node_id": relationship.source_node_id,
                    "target_node_id": relationship.target_node_id
                })
                
                # Execute query
                result = await session.run(query, **properties)
                summary = await result.consume()
                
                logger.info(
                    "Relationship created successfully",
                    relationship_id=relationship.relationship_id,
                    relationship_type=relationship.relationship_type,
                    relationships_created=summary.counters.relationships_created
                )
                
                return {
                    "success": True,
                    "relationship_id": relationship.relationship_id,
                    "relationships_created": summary.counters.relationships_created,
                    "properties_set": summary.counters.properties_set
                }
                
        except Exception as e:
            error_msg = f"Failed to create relationship {relationship.relationship_id}: {str(e)}"
            logger.error(error_msg, relationship_id=relationship.relationship_id, error=str(e))
            raise GraphSchemaError(error_msg) from e
    
    async def batch_create_nodes(self, nodes: List[BaseNode], batch_size: int = 100) -> Dict[str, Any]:
        """Create multiple nodes in batches.
        
        Args:
            nodes: List of nodes to create
            batch_size: Number of nodes per batch
            
        Returns:
            Dict with batch creation results
        """
        logger.info(f"Creating {len(nodes)} nodes in batches of {batch_size}")
        
        results = {
            "total_nodes": len(nodes),
            "nodes_created": 0,
            "batches_processed": 0,
            "errors": []
        }
        
        try:
            driver = await self._get_driver()
            
            # Process nodes in batches
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                
                async with driver.session() as session:
                    async with session.begin_transaction() as tx:
                        batch_created = 0
                        
                        for node in batch:
                            try:
                                query = node.get_cypher_create_query()
                                properties = node.to_neo4j_properties()
                                
                                result = await tx.run(query, **properties)
                                summary = await result.consume()
                                batch_created += summary.counters.nodes_created
                                
                            except Exception as e:
                                error_msg = f"Failed to create node {node.node_id}: {str(e)}"
                                logger.error(error_msg, node_id=node.node_id)
                                results["errors"].append(error_msg)
                        
                        results["nodes_created"] += batch_created
                        results["batches_processed"] += 1
                        
                        logger.debug(
                            f"Batch {results['batches_processed']} completed",
                            batch_size=len(batch),
                            batch_created=batch_created
                        )
            
            logger.info(
                "Batch node creation completed",
                total_nodes=results["total_nodes"],
                nodes_created=results["nodes_created"],
                errors=len(results["errors"])
            )
            
        except Exception as e:
            error_msg = f"Batch node creation failed: {str(e)}"
            logger.error(error_msg, error=str(e))
            results["errors"].append(error_msg)
            raise GraphSchemaError(error_msg) from e
        
        return results
    
    async def get_schema_statistics(self) -> Dict[str, Any]:
        """Get comprehensive schema statistics.
        
        Returns:
            Dict with schema statistics
        """
        logger.info("Collecting schema statistics")
        
        try:
            driver = await self._get_driver()
            
            async with driver.session() as session:
                # Node counts by type
                node_counts = {}
                for node_type in NodeType:
                    result = await session.run(
                        f"MATCH (n:{node_type.value}) RETURN count(n) as count"
                    )
                    record = await result.single()
                    node_counts[node_type.value] = record["count"] if record else 0
                
                # Relationship counts by type
                relationship_counts = {}
                for rel_type in RelationshipType:
                    result = await session.run(
                        f"MATCH ()-[r:{rel_type.value}]-() RETURN count(r) as count"
                    )
                    record = await result.single()
                    relationship_counts[rel_type.value] = record["count"] if record else 0
                
                # Vector embedding statistics
                vector_stats = {}
                for node_type in ["Document", "Chunk", "Entity", "Concept"]:
                    # Content embeddings
                    result = await session.run(
                        f"MATCH (n:{node_type}) WHERE n.content_embedding IS NOT NULL RETURN count(n) as count"
                    )
                    record = await result.single()
                    vector_stats[f"{node_type.lower()}_content_embeddings"] = record["count"] if record else 0
                    
                    # Title embeddings
                    result = await session.run(
                        f"MATCH (n:{node_type}) WHERE n.title_embedding IS NOT NULL RETURN count(n) as count"
                    )
                    record = await result.single()
                    vector_stats[f"{node_type.lower()}_title_embeddings"] = record["count"] if record else 0
                
                # Overall statistics
                total_nodes_result = await session.run("MATCH (n) RETURN count(n) as count")
                total_nodes_record = await total_nodes_result.single()
                total_nodes = total_nodes_record["count"] if total_nodes_record else 0
                
                total_relationships_result = await session.run("MATCH ()-[r]-() RETURN count(r) as count")
                total_relationships_record = await total_relationships_result.single()
                total_relationships = total_relationships_record["count"] if total_relationships_record else 0
                
                statistics = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "total_nodes": total_nodes,
                    "total_relationships": total_relationships,
                    "node_counts": node_counts,
                    "relationship_counts": relationship_counts,
                    "vector_statistics": vector_stats,
                    "schema_health": {
                        "constraints_active": self._constraints_created,
                        "indexes_active": self._indexes_created,
                        "vector_indexes_active": self._vector_indexes_created
                    }
                }
                
                logger.info(
                    "Schema statistics collected",
                    total_nodes=total_nodes,
                    total_relationships=total_relationships,
                    node_types=len([c for c in node_counts.values() if c > 0]),
                    relationship_types=len([c for c in relationship_counts.values() if c > 0])
                )
                
                return statistics
                
        except Exception as e:
            error_msg = f"Failed to collect schema statistics: {str(e)}"
            logger.error(error_msg, error=str(e))
            raise GraphSchemaError(error_msg) from e
    
    async def migrate_schema(self, migration_version: str) -> Dict[str, Any]:
        """Perform schema migration.
        
        Args:
            migration_version: Target migration version
            
        Returns:
            Dict with migration results
        """
        logger.info(f"Performing schema migration to version {migration_version}")
        
        migration_result = {
            "timestamp": datetime.utcnow().isoformat(),
            "migration_version": migration_version,
            "migration_id": str(uuid.uuid4()),
            "success": False,
            "changes_applied": [],
            "errors": []
        }
        
        try:
            # Schema migrations would be implemented here
            # For now, return a placeholder result
            migration_result["success"] = True
            migration_result["changes_applied"].append("No migrations available yet")
            
            logger.info(f"Schema migration to {migration_version} completed successfully")
            
        except Exception as e:
            error_msg = f"Schema migration failed: {str(e)}"
            logger.error(error_msg, migration_version=migration_version, error=str(e))
            migration_result["errors"].append(error_msg)
            raise GraphSchemaError(error_msg) from e
        
        return migration_result