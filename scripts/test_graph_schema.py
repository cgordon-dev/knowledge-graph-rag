#!/usr/bin/env python3
"""
Test script for Neo4j Vector Graph Schema implementation.

Tests schema creation, node/relationship creation, vector operations,
and query building functionality.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from neo4j import AsyncGraphDatabase
import structlog

from kg_rag.core.config import get_settings
from kg_rag.graph_schema import (
    GraphSchemaManager, VectorGraphOperations, GraphQueryBuilder, SchemaValidator
)
from kg_rag.graph_schema.node_models import (
    DocumentNode, ChunkNode, EntityNode, VectorEmbedding, NodeType
)
from kg_rag.graph_schema.relationship_models import (
    ContainsRelationship, MentionsRelationship, RelationshipType
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def test_schema_initialization():
    """Test schema initialization."""
    logger.info("Testing schema initialization")
    
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.database.neo4j_uri,
        auth=(settings.database.neo4j_user, settings.database.neo4j_password)
    )
    
    try:
        schema_manager = GraphSchemaManager(driver)
        
        # Initialize schema
        result = await schema_manager.initialize_schema(drop_existing=True)
        
        logger.info(
            "Schema initialization completed",
            constraints_created=result["constraints_created"],
            indexes_created=result["indexes_created"],
            vector_indexes_created=result["vector_indexes_created"],
            errors=len(result["errors"])
        )
        
        # Validate schema
        validation_result = await schema_manager.validate_schema()
        logger.info(
            "Schema validation completed",
            is_valid=validation_result["is_valid"],
            constraints_missing=len(validation_result["constraints"]["missing"]),
            indexes_missing=len(validation_result["indexes"]["missing"])
        )
        
        return schema_manager
        
    except Exception as e:
        logger.error(f"Schema initialization failed: {str(e)}")
        raise
    finally:
        if 'schema_manager' in locals():
            await schema_manager.close()


async def test_node_creation():
    """Test node creation."""
    logger.info("Testing node creation")
    
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.database.neo4j_uri,
        auth=(settings.database.neo4j_user, settings.database.neo4j_password)
    )
    
    try:
        schema_manager = GraphSchemaManager(driver)
        
        # Create sample embedding
        sample_embedding = VectorEmbedding(
            vector=[0.1] * settings.ai_models.embedding_dimension,
            model="test-model",
            dimension=settings.ai_models.embedding_dimension
        )
        
        # Create document node
        document = DocumentNode(
            node_id="doc_001",
            title="Test Document",
            description="A test document for schema testing",
            content="This is test content for the document.",
            document_type="text",
            word_count=8,
            content_embedding=sample_embedding,
            tags=["test", "document"],
            categories=["testing"]
        )
        
        result = await schema_manager.create_node(document)
        logger.info("Document node created", result=result)
        
        # Create chunk node
        chunk = ChunkNode(
            node_id="chunk_001",
            title="Test Chunk",
            content="This is test content.",
            chunk_index=0,
            start_position=0,
            end_position=21,
            word_count=4,
            sentence_count=1,
            document_id="doc_001",
            content_embedding=sample_embedding
        )
        
        result = await schema_manager.create_node(chunk)
        logger.info("Chunk node created", result=result)
        
        # Create entity node
        entity = EntityNode(
            node_id="entity_001",
            title="Test Entity",
            entity_type="CONCEPT",
            canonical_name="Test Entity",
            context="A test entity for demonstration",
            mention_count=1,
            importance_score=0.8,
            content_embedding=sample_embedding
        )
        
        result = await schema_manager.create_node(entity)
        logger.info("Entity node created", result=result)
        
        return [document, chunk, entity]
        
    except Exception as e:
        logger.error(f"Node creation failed: {str(e)}")
        raise
    finally:
        await schema_manager.close()


async def test_relationship_creation():
    """Test relationship creation."""
    logger.info("Testing relationship creation")
    
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.database.neo4j_uri,
        auth=(settings.database.neo4j_user, settings.database.neo4j_password)
    )
    
    try:
        schema_manager = GraphSchemaManager(driver)
        
        # Create contains relationship (document contains chunk)
        contains_rel = ContainsRelationship(
            relationship_id="rel_001",
            source_node_id="doc_001",
            target_node_id="chunk_001",
            weight=1.0,
            confidence=1.0,
            order=0
        )
        
        result = await schema_manager.create_relationship(contains_rel)
        logger.info("Contains relationship created", result=result)
        
        # Create mentions relationship (chunk mentions entity)
        mentions_rel = MentionsRelationship(
            relationship_id="rel_002",
            source_node_id="chunk_001",
            target_node_id="entity_001",
            weight=0.8,
            confidence=0.9,
            mention_context="entity mentioned in chunk",
            mention_frequency=1
        )
        
        result = await schema_manager.create_relationship(mentions_rel)
        logger.info("Mentions relationship created", result=result)
        
        return [contains_rel, mentions_rel]
        
    except Exception as e:
        logger.error(f"Relationship creation failed: {str(e)}")
        raise
    finally:
        await schema_manager.close()


async def test_vector_operations():
    """Test vector operations."""
    logger.info("Testing vector operations")
    
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.database.neo4j_uri,
        auth=(settings.database.neo4j_user, settings.database.neo4j_password)
    )
    
    try:
        vector_ops = VectorGraphOperations(driver)
        
        # Test vector similarity search
        query_vector = [0.1] * settings.ai_models.embedding_dimension
        
        results = await vector_ops.vector_similarity_search(
            query_vector=query_vector,
            node_types=[NodeType.DOCUMENT, NodeType.CHUNK, NodeType.ENTITY],
            limit=5,
            similarity_threshold=0.5
        )
        
        logger.info(
            "Vector similarity search completed",
            results_count=len(results),
            max_score=max([r["similarity_score"] for r in results]) if results else 0
        )
        
        # Test hybrid search
        hybrid_results = await vector_ops.hybrid_search(
            query_vector=query_vector,
            node_types=[NodeType.DOCUMENT, NodeType.CHUNK],
            vector_weight=0.7,
            graph_weight=0.3,
            limit=3
        )
        
        logger.info(
            "Hybrid search completed",
            results_count=len(hybrid_results)
        )
        
        # Test find similar nodes
        if results:
            similar_nodes = await vector_ops.find_similar_nodes_by_content(
                node_id=results[0]["node_id"],
                similarity_threshold=0.5,
                limit=3
            )
            
            logger.info(
                "Similar nodes search completed",
                results_count=len(similar_nodes)
            )
        
        # Get vector statistics
        stats = await vector_ops.get_vector_statistics()
        logger.info("Vector statistics collected", stats=stats)
        
    except Exception as e:
        logger.error(f"Vector operations failed: {str(e)}")
        raise
    finally:
        await driver.close()


async def test_query_builder():
    """Test query builder functionality."""
    logger.info("Testing query builder")
    
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.database.neo4j_uri,
        auth=(settings.database.neo4j_user, settings.database.neo4j_password)
    )
    
    try:
        # Test basic node query
        query_builder = GraphQueryBuilder(driver)
        
        query, params = (query_builder
                        .match_node(NodeType.DOCUMENT, variable="doc")
                        .where_property("doc", "document_type", "text")
                        .return_nodes("doc", ["node_id", "title", "word_count"])
                        .order_by("doc.created_at")
                        .limit(10)
                        .build())
        
        logger.info("Built document query", query=query, params=params)
        
        # Execute query
        results = await query_builder.execute()
        logger.info("Document query executed", results_count=len(results))
        
        # Test relationship query
        query_builder = GraphQueryBuilder(driver)
        
        query, params = (query_builder
                        .match_node(NodeType.DOCUMENT, variable="doc")
                        .match_relationship("doc", "chunk", [RelationshipType.CONTAINS], 
                                          relationship_var="contains")
                        .match_node(NodeType.CHUNK, variable="chunk")
                        .return_custom([
                            "doc.node_id as document_id",
                            "doc.title as document_title", 
                            "chunk.node_id as chunk_id",
                            "chunk.word_count as chunk_words"
                        ])
                        .build())
        
        logger.info("Built relationship query", query=query)
        
        # Test convenience functions
        from kg_rag.graph_schema.query_builder import find_documents_by_content
        
        doc_query = find_documents_by_content("test", limit=5, driver=driver)
        query, params = doc_query.build()
        logger.info("Built content search query", query=query)
        
    except Exception as e:
        logger.error(f"Query builder test failed: {str(e)}")
        raise
    finally:
        await driver.close()


async def test_schema_validation():
    """Test comprehensive schema validation."""
    logger.info("Testing schema validation")
    
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.database.neo4j_uri,
        auth=(settings.database.neo4j_user, settings.database.neo4j_password)
    )
    
    try:
        validator = SchemaValidator(driver)
        
        # Run comprehensive validation
        validation_result = await validator.validate_complete_schema()
        
        logger.info(
            "Schema validation completed",
            overall_health=validation_result["overall_health"],
            total_issues=validation_result["statistics"]["total_issues"],
            critical_issues=validation_result["statistics"]["issues_by_severity"]["critical"],
            warning_issues=validation_result["statistics"]["issues_by_severity"]["warning"]
        )
        
        # Log category statuses
        for category, data in validation_result["categories"].items():
            logger.info(
                f"Category validation: {category}",
                status=data["status"],
                issues_count=len(data["issues"])
            )
        
        # Log recommendations
        if validation_result["recommendations"]:
            logger.info("Validation recommendations", recommendations=validation_result["recommendations"])
        
    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise
    finally:
        await driver.close()


async def test_graph_statistics():
    """Test graph statistics collection."""
    logger.info("Testing graph statistics collection")
    
    settings = get_settings()
    driver = AsyncGraphDatabase.driver(
        settings.database.neo4j_uri,
        auth=(settings.database.neo4j_user, settings.database.neo4j_password)
    )
    
    try:
        schema_manager = GraphSchemaManager(driver)
        
        # Get comprehensive statistics
        stats = await schema_manager.get_schema_statistics()
        
        logger.info(
            "Graph statistics collected",
            total_nodes=stats["total_nodes"],
            total_relationships=stats["total_relationships"],
            node_types=len([t for t, c in stats["node_counts"].items() if c > 0]),
            relationship_types=len([t for t, c in stats["relationship_counts"].items() if c > 0])
        )
        
        # Log node counts by type
        for node_type, count in stats["node_counts"].items():
            if count > 0:
                logger.info(f"Node type: {node_type}", count=count)
        
        # Log relationship counts by type
        for rel_type, count in stats["relationship_counts"].items():
            if count > 0:
                logger.info(f"Relationship type: {rel_type}", count=count)
        
        # Log vector statistics
        for vector_type, count in stats["vector_statistics"].items():
            if count > 0:
                logger.info(f"Vector type: {vector_type}", count=count)
        
    except Exception as e:
        logger.error(f"Statistics collection failed: {str(e)}")
        raise
    finally:
        await schema_manager.close()


async def main():
    """Run all tests."""
    logger.info("Starting Neo4j Vector Graph Schema tests")
    
    try:
        # Test 1: Schema initialization
        await test_schema_initialization()
        
        # Test 2: Node creation
        await test_node_creation()
        
        # Test 3: Relationship creation
        await test_relationship_creation()
        
        # Test 4: Vector operations
        await test_vector_operations()
        
        # Test 5: Query builder
        await test_query_builder()
        
        # Test 6: Schema validation
        await test_schema_validation()
        
        # Test 7: Graph statistics
        await test_graph_statistics()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())