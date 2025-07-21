"""
Query builder for Neo4j Vector Graph Schema.

Provides fluent API for constructing complex graph queries with vector
similarity, relationship traversal, and filtering capabilities.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum
import re

import structlog
from neo4j import AsyncDriver

from kg_rag.core.exceptions import QueryBuilderError
from kg_rag.graph_schema.node_models import NodeType
from kg_rag.graph_schema.relationship_models import RelationshipType

logger = structlog.get_logger(__name__)


class SortOrder(str, Enum):
    """Sort order enumeration."""
    ASC = "ASC"
    DESC = "DESC"


class MatchType(str, Enum):
    """Match type enumeration."""
    EXACT = "exact"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"


class GraphQueryBuilder:
    """Fluent query builder for Neo4j graph operations."""
    
    def __init__(self, driver: Optional[AsyncDriver] = None):
        """Initialize query builder.
        
        Args:
            driver: Optional Neo4j driver instance
        """
        self.driver = driver
        self._reset()
    
    def _reset(self):
        """Reset query builder state."""
        self._match_clauses = []
        self._where_clauses = []
        self._with_clauses = []
        self._return_clauses = []
        self._order_by_clauses = []
        self._limit_value = None
        self._skip_value = None
        self._parameters = {}
        self._variable_counter = 0
    
    def _get_next_variable(self, prefix: str = "var") -> str:
        """Get next available variable name."""
        self._variable_counter += 1
        return f"{prefix}{self._variable_counter}"
    
    def _add_parameter(self, value: Any, prefix: str = "param") -> str:
        """Add parameter and return parameter name."""
        param_name = f"{prefix}_{len(self._parameters)}"
        self._parameters[param_name] = value
        return param_name
    
    def match_node(
        self,
        node_types: Optional[Union[NodeType, List[NodeType]]] = None,
        properties: Optional[Dict[str, Any]] = None,
        variable: Optional[str] = None
    ) -> 'GraphQueryBuilder':
        """Add node match clause.
        
        Args:
            node_types: Node type(s) to match
            properties: Node properties to match
            variable: Variable name for the node
            
        Returns:
            Self for method chaining
        """
        if variable is None:
            variable = self._get_next_variable("n")
        
        # Build node labels
        labels = ""
        if node_types:
            if isinstance(node_types, NodeType):
                node_types = [node_types]
            labels = ":" + ":".join([f"`{nt.value}`" for nt in node_types])
        
        # Build properties clause
        props_clause = ""
        if properties:
            prop_conditions = []
            for key, value in properties.items():
                param_name = self._add_parameter(value)
                prop_conditions.append(f"{key}: ${param_name}")
            props_clause = " {" + ", ".join(prop_conditions) + "}"
        
        match_clause = f"MATCH ({variable}{labels}{props_clause})"
        self._match_clauses.append(match_clause)
        
        return self
    
    def match_relationship(
        self,
        source_var: str,
        target_var: str,
        relationship_types: Optional[Union[RelationshipType, List[RelationshipType]]] = None,
        direction: str = "->",
        relationship_var: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        min_hops: Optional[int] = None,
        max_hops: Optional[int] = None
    ) -> 'GraphQueryBuilder':
        """Add relationship match clause.
        
        Args:
            source_var: Source node variable
            target_var: Target node variable
            relationship_types: Relationship type(s) to match
            direction: Relationship direction (-> or <-)
            relationship_var: Variable name for the relationship
            properties: Relationship properties to match
            min_hops: Minimum number of hops
            max_hops: Maximum number of hops
            
        Returns:
            Self for method chaining
        """
        if relationship_var is None:
            relationship_var = self._get_next_variable("r")
        
        # Build relationship types
        rel_types = ""
        if relationship_types:
            if isinstance(relationship_types, RelationshipType):
                relationship_types = [relationship_types]
            rel_types = ":" + "|".join([rt.value for rt in relationship_types])
        
        # Build properties clause
        props_clause = ""
        if properties:
            prop_conditions = []
            for key, value in properties.items():
                param_name = self._add_parameter(value)
                prop_conditions.append(f"{key}: ${param_name}")
            props_clause = " {" + ", ".join(prop_conditions) + "}"
        
        # Build hops clause
        hops_clause = ""
        if min_hops is not None or max_hops is not None:
            if min_hops is not None and max_hops is not None:
                hops_clause = f"*{min_hops}..{max_hops}"
            elif min_hops is not None:
                hops_clause = f"*{min_hops}.."
            elif max_hops is not None:
                hops_clause = f"*..{max_hops}"
        
        # Build relationship clause
        if direction == "->":
            rel_clause = f"-[{relationship_var}{rel_types}{hops_clause}{props_clause}]->"
        elif direction == "<-":
            rel_clause = f"<-[{relationship_var}{rel_types}{hops_clause}{props_clause}]-"
        else:
            rel_clause = f"-[{relationship_var}{rel_types}{hops_clause}{props_clause}]-"
        
        match_clause = f"MATCH ({source_var}){rel_clause}({target_var})"
        self._match_clauses.append(match_clause)
        
        return self
    
    def match_path(
        self,
        path_pattern: str,
        path_var: Optional[str] = None
    ) -> 'GraphQueryBuilder':
        """Add path match clause.
        
        Args:
            path_pattern: Cypher path pattern
            path_var: Variable name for the path
            
        Returns:
            Self for method chaining
        """
        if path_var:
            match_clause = f"MATCH {path_var} = {path_pattern}"
        else:
            match_clause = f"MATCH {path_pattern}"
        
        self._match_clauses.append(match_clause)
        return self
    
    def where(self, condition: str, **parameters) -> 'GraphQueryBuilder':
        """Add WHERE clause condition.
        
        Args:
            condition: WHERE condition
            **parameters: Named parameters for the condition
            
        Returns:
            Self for method chaining
        """
        # Add parameters
        for param_name, param_value in parameters.items():
            self._parameters[param_name] = param_value
        
        self._where_clauses.append(condition)
        return self
    
    def where_property(
        self,
        variable: str,
        property_name: str,
        value: Any,
        match_type: MatchType = MatchType.EXACT
    ) -> 'GraphQueryBuilder':
        """Add property-based WHERE condition.
        
        Args:
            variable: Node/relationship variable
            property_name: Property name
            value: Property value
            match_type: Type of matching to perform
            
        Returns:
            Self for method chaining
        """
        param_name = self._add_parameter(value)
        
        if match_type == MatchType.EXACT:
            condition = f"{variable}.{property_name} = ${param_name}"
        elif match_type == MatchType.CONTAINS:
            condition = f"{variable}.{property_name} CONTAINS ${param_name}"
        elif match_type == MatchType.STARTS_WITH:
            condition = f"{variable}.{property_name} STARTS WITH ${param_name}"
        elif match_type == MatchType.ENDS_WITH:
            condition = f"{variable}.{property_name} ENDS WITH ${param_name}"
        elif match_type == MatchType.REGEX:
            condition = f"{variable}.{property_name} =~ ${param_name}"
        else:
            raise QueryBuilderError(f"Unsupported match type: {match_type}")
        
        self._where_clauses.append(condition)
        return self
    
    def where_in(
        self,
        variable: str,
        property_name: str,
        values: List[Any]
    ) -> 'GraphQueryBuilder':
        """Add IN clause condition.
        
        Args:
            variable: Node/relationship variable
            property_name: Property name
            values: List of values to match
            
        Returns:
            Self for method chaining
        """
        param_name = self._add_parameter(values)
        condition = f"{variable}.{property_name} IN ${param_name}"
        self._where_clauses.append(condition)
        return self
    
    def where_range(
        self,
        variable: str,
        property_name: str,
        min_value: Optional[Any] = None,
        max_value: Optional[Any] = None,
        inclusive: bool = True
    ) -> 'GraphQueryBuilder':
        """Add range condition.
        
        Args:
            variable: Node/relationship variable
            property_name: Property name
            min_value: Minimum value (optional)
            max_value: Maximum value (optional)
            inclusive: Whether range is inclusive
            
        Returns:
            Self for method chaining
        """
        conditions = []
        
        if min_value is not None:
            param_name = self._add_parameter(min_value)
            operator = ">=" if inclusive else ">"
            conditions.append(f"{variable}.{property_name} {operator} ${param_name}")
        
        if max_value is not None:
            param_name = self._add_parameter(max_value)
            operator = "<=" if inclusive else "<"
            conditions.append(f"{variable}.{property_name} {operator} ${param_name}")
        
        if conditions:
            self._where_clauses.extend(conditions)
        
        return self
    
    def vector_similarity(
        self,
        variable: str,
        embedding_field: str,
        query_vector: List[float],
        similarity_threshold: float = 0.7,
        index_name: Optional[str] = None
    ) -> 'GraphQueryBuilder':
        """Add vector similarity condition.
        
        Args:
            variable: Node variable
            embedding_field: Embedding field name
            query_vector: Query vector
            similarity_threshold: Minimum similarity threshold
            index_name: Optional vector index name
            
        Returns:
            Self for method chaining
        """
        vector_param = self._add_parameter(query_vector)
        threshold_param = self._add_parameter(similarity_threshold)
        
        if index_name:
            # Use specific vector index
            match_clause = f"""
            CALL db.index.vector.queryNodes('{index_name}', 100, ${vector_param})
            YIELD node, score
            WHERE node = {variable} AND score >= ${threshold_param}
            """
        else:
            # Use generic vector similarity
            match_clause = f"""
            WHERE {variable}.{embedding_field} IS NOT NULL 
            AND gds.similarity.cosine({variable}.{embedding_field}, ${vector_param}) >= ${threshold_param}
            """
        
        self._where_clauses.append(match_clause.strip())
        return self
    
    def with_clause(self, expressions: Union[str, List[str]]) -> 'GraphQueryBuilder':
        """Add WITH clause.
        
        Args:
            expressions: WITH expressions
            
        Returns:
            Self for method chaining
        """
        if isinstance(expressions, str):
            expressions = [expressions]
        
        with_clause = "WITH " + ", ".join(expressions)
        self._with_clauses.append(with_clause)
        return self
    
    def return_nodes(
        self,
        variables: Union[str, List[str]],
        properties: Optional[List[str]] = None
    ) -> 'GraphQueryBuilder':
        """Add RETURN clause for nodes.
        
        Args:
            variables: Node variables to return
            properties: Specific properties to return
            
        Returns:
            Self for method chaining
        """
        if isinstance(variables, str):
            variables = [variables]
        
        if properties:
            return_expressions = []
            for var in variables:
                for prop in properties:
                    return_expressions.append(f"{var}.{prop}")
            self._return_clauses.extend(return_expressions)
        else:
            self._return_clauses.extend(variables)
        
        return self
    
    def return_relationships(
        self,
        variables: Union[str, List[str]],
        properties: Optional[List[str]] = None
    ) -> 'GraphQueryBuilder':
        """Add RETURN clause for relationships.
        
        Args:
            variables: Relationship variables to return
            properties: Specific properties to return
            
        Returns:
            Self for method chaining
        """
        if isinstance(variables, str):
            variables = [variables]
        
        if properties:
            return_expressions = []
            for var in variables:
                for prop in properties:
                    return_expressions.append(f"{var}.{prop}")
            self._return_clauses.extend(return_expressions)
        else:
            self._return_clauses.extend(variables)
        
        return self
    
    def return_custom(self, expressions: Union[str, List[str]]) -> 'GraphQueryBuilder':
        """Add custom RETURN expressions.
        
        Args:
            expressions: Custom return expressions
            
        Returns:
            Self for method chaining
        """
        if isinstance(expressions, str):
            expressions = [expressions]
        
        self._return_clauses.extend(expressions)
        return self
    
    def order_by(
        self,
        expressions: Union[str, List[str]],
        order: SortOrder = SortOrder.ASC
    ) -> 'GraphQueryBuilder':
        """Add ORDER BY clause.
        
        Args:
            expressions: Sort expressions
            order: Sort order
            
        Returns:
            Self for method chaining
        """
        if isinstance(expressions, str):
            expressions = [expressions]
        
        order_expressions = [f"{expr} {order.value}" for expr in expressions]
        self._order_by_clauses.extend(order_expressions)
        return self
    
    def limit(self, count: int) -> 'GraphQueryBuilder':
        """Add LIMIT clause.
        
        Args:
            count: Maximum number of results
            
        Returns:
            Self for method chaining
        """
        self._limit_value = count
        return self
    
    def skip(self, count: int) -> 'GraphQueryBuilder':
        """Add SKIP clause.
        
        Args:
            count: Number of results to skip
            
        Returns:
            Self for method chaining
        """
        self._skip_value = count
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the Cypher query.
        
        Returns:
            Tuple of (query_string, parameters)
        """
        if not self._match_clauses and not self._with_clauses:
            raise QueryBuilderError("Query must have at least one MATCH or WITH clause")
        
        if not self._return_clauses:
            raise QueryBuilderError("Query must have a RETURN clause")
        
        query_parts = []
        
        # Add MATCH clauses
        query_parts.extend(self._match_clauses)
        
        # Add WHERE clauses
        if self._where_clauses:
            where_conditions = " AND ".join(self._where_clauses)
            query_parts.append(f"WHERE {where_conditions}")
        
        # Add WITH clauses
        query_parts.extend(self._with_clauses)
        
        # Add RETURN clause
        return_clause = "RETURN " + ", ".join(self._return_clauses)
        query_parts.append(return_clause)
        
        # Add ORDER BY clause
        if self._order_by_clauses:
            order_clause = "ORDER BY " + ", ".join(self._order_by_clauses)
            query_parts.append(order_clause)
        
        # Add SKIP clause
        if self._skip_value is not None:
            query_parts.append(f"SKIP {self._skip_value}")
        
        # Add LIMIT clause
        if self._limit_value is not None:
            query_parts.append(f"LIMIT {self._limit_value}")
        
        query = "\n".join(query_parts)
        
        logger.debug("Built Cypher query", query=query, parameters=self._parameters)
        
        return query, self._parameters.copy()
    
    async def execute(self, driver: Optional[AsyncDriver] = None) -> List[Dict[str, Any]]:
        """Execute the built query.
        
        Args:
            driver: Optional Neo4j driver (uses instance driver if not provided)
            
        Returns:
            List of query results
        """
        if driver is None:
            driver = self.driver
        
        if driver is None:
            raise QueryBuilderError("No Neo4j driver available for query execution")
        
        query, parameters = self.build()
        
        try:
            async with driver.session() as session:
                result = await session.run(query, parameters)
                records = [dict(record) async for record in result]
                
                logger.info(
                    "Query executed successfully",
                    results_count=len(records),
                    query_length=len(query)
                )
                
                return records
                
        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg, query=query, parameters=parameters, error=str(e))
            raise QueryBuilderError(error_msg) from e
        finally:
            # Reset builder state for reuse
            self._reset()
    
    def clone(self) -> 'GraphQueryBuilder':
        """Create a copy of the query builder.
        
        Returns:
            New GraphQueryBuilder instance with same state
        """
        new_builder = GraphQueryBuilder(self.driver)
        new_builder._match_clauses = self._match_clauses.copy()
        new_builder._where_clauses = self._where_clauses.copy()
        new_builder._with_clauses = self._with_clauses.copy()
        new_builder._return_clauses = self._return_clauses.copy()
        new_builder._order_by_clauses = self._order_by_clauses.copy()
        new_builder._limit_value = self._limit_value
        new_builder._skip_value = self._skip_value
        new_builder._parameters = self._parameters.copy()
        new_builder._variable_counter = self._variable_counter
        
        return new_builder


# Convenience functions for common query patterns
def find_documents_by_content(
    content_query: str,
    limit: int = 10,
    driver: Optional[AsyncDriver] = None
) -> GraphQueryBuilder:
    """Find documents by content search.
    
    Args:
        content_query: Content to search for
        limit: Maximum results
        driver: Optional Neo4j driver
        
    Returns:
        Configured GraphQueryBuilder
    """
    return (GraphQueryBuilder(driver)
            .match_node(NodeType.DOCUMENT, variable="doc")
            .where_property("doc", "content", f".*{re.escape(content_query)}.*", MatchType.REGEX)
            .return_nodes("doc", ["node_id", "title", "content", "document_type"])
            .order_by("doc.created_at", SortOrder.DESC)
            .limit(limit))


def find_related_entities(
    entity_id: str,
    max_hops: int = 2,
    relationship_types: Optional[List[RelationshipType]] = None,
    driver: Optional[AsyncDriver] = None
) -> GraphQueryBuilder:
    """Find entities related to a given entity.
    
    Args:
        entity_id: Source entity ID
        max_hops: Maximum relationship hops
        relationship_types: Relationship types to traverse
        driver: Optional Neo4j driver
        
    Returns:
        Configured GraphQueryBuilder
    """
    builder = (GraphQueryBuilder(driver)
               .match_node(NodeType.ENTITY, {"node_id": entity_id}, "source")
               .match_relationship("source", "target", relationship_types, 
                                 max_hops=max_hops, relationship_var="rel")
               .match_node(NodeType.ENTITY, variable="target")
               .return_custom([
                   "target.node_id as entity_id",
                   "target.canonical_name as name",
                   "target.entity_type as type",
                   "target.importance_score as importance",
                   "collect(rel.relationship_type) as relationships"
               ])
               .order_by("target.importance_score", SortOrder.DESC))
    
    return builder


def find_compliance_gaps(
    framework: str,
    implementation_status: str = "Not Implemented",
    driver: Optional[AsyncDriver] = None
) -> GraphQueryBuilder:
    """Find compliance gaps for a framework.
    
    Args:
        framework: Compliance framework name
        implementation_status: Status to filter by
        driver: Optional Neo4j driver
        
    Returns:
        Configured GraphQueryBuilder
    """
    return (GraphQueryBuilder(driver)
            .match_node(NodeType.CONTROL, variable="control")
            .where_property("control", "framework", framework)
            .where_property("control", "status", implementation_status)
            .return_nodes("control", [
                "node_id", "control_id", "title", "control_family",
                "compliance_level", "priority", "implementation_guidance"
            ])
            .order_by("control.priority", SortOrder.DESC))


def find_process_automation_candidates(
    min_automation_potential: float = 0.7,
    process_type: Optional[str] = None,
    driver: Optional[AsyncDriver] = None
) -> GraphQueryBuilder:
    """Find processes with high automation potential.
    
    Args:
        min_automation_potential: Minimum automation potential score
        process_type: Optional process type filter
        driver: Optional Neo4j driver
        
    Returns:
        Configured GraphQueryBuilder
    """
    builder = (GraphQueryBuilder(driver)
               .match_node(NodeType.PROCESS, variable="process")
               .where_range("process", "automation_level", max_value=min_automation_potential))
    
    if process_type:
        builder.where_property("process", "process_type", process_type)
    
    return (builder
            .return_nodes("process", [
                "node_id", "title", "process_type", "automation_level",
                "cycle_time", "success_rate", "owner"
            ])
            .order_by("process.automation_level", SortOrder.DESC))