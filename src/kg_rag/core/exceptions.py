"""
Custom exceptions for Knowledge Graph-RAG system.

Provides comprehensive error handling with security and compliance considerations.
"""

from typing import Any, Dict, Optional


class KGRAGException(Exception):
    """Base exception for Knowledge Graph-RAG system."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize KGRAGException.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
            cause: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "KGRAG_UNKNOWN_ERROR"
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/API responses."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "type": self.__class__.__name__
        }


# =============================================================================
# Configuration and Setup Exceptions
# =============================================================================

class ConfigurationError(KGRAGException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, setting_name: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_CONFIG_ERROR",
            details={"setting_name": setting_name} if setting_name else None
        )


class DatabaseConnectionError(KGRAGException):
    """Raised when database connection fails."""
    
    def __init__(self, message: str, database_type: str, connection_uri: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_DB_CONNECTION_ERROR",
            details={
                "database_type": database_type,
                "connection_uri": connection_uri  # Sanitized in logger
            }
        )


class ModelLoadError(KGRAGException):
    """Raised when AI model loading fails."""
    
    def __init__(self, message: str, model_name: str, model_path: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_MODEL_LOAD_ERROR",
            details={
                "model_name": model_name,
                "model_path": model_path
            }
        )


# =============================================================================
# AI Digital Twins Exceptions
# =============================================================================

class PersonaTwinError(KGRAGException):
    """Base exception for persona twin operations."""
    
    def __init__(self, message: str, persona_id: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_PERSONA_ERROR",
            details={"persona_id": persona_id} if persona_id else None
        )


class PersonaNotFoundError(PersonaTwinError):
    """Raised when requested persona is not found."""
    
    def __init__(self, persona_id: str):
        super().__init__(
            message=f"Persona not found: {persona_id}",
            persona_id=persona_id
        )
        self.error_code = "KGRAG_PERSONA_NOT_FOUND"


class PersonaValidationError(PersonaTwinError):
    """Raised when persona validation fails."""
    
    def __init__(self, message: str, persona_id: str, validation_errors: Dict[str, Any]):
        super().__init__(
            message=message,
            persona_id=persona_id
        )
        self.error_code = "KGRAG_PERSONA_VALIDATION_ERROR"
        self.details.update({"validation_errors": validation_errors})


class ExpertTwinError(KGRAGException):
    """Raised when expert twin operations fail."""
    
    def __init__(self, message: str, expert_domain: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_EXPERT_TWIN_ERROR",
            details={"expert_domain": expert_domain} if expert_domain else None
        )


class ExpertValidationError(ExpertTwinError):
    """Raised when expert validation fails."""
    
    def __init__(self, message: str, expert_domain: str, confidence_score: float):
        super().__init__(
            message=message,
            expert_domain=expert_domain
        )
        self.error_code = "KGRAG_EXPERT_VALIDATION_ERROR"
        self.details.update({"confidence_score": confidence_score})


# =============================================================================
# MCP Server Exceptions
# =============================================================================

class MCPServerError(KGRAGException):
    """Base exception for MCP server operations."""
    
    def __init__(self, message: str, server_name: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_MCP_SERVER_ERROR",
            details={"server_name": server_name} if server_name else None
        )


class MCPServerNotAvailableError(MCPServerError):
    """Raised when MCP server is not available."""
    
    def __init__(self, server_name: str):
        super().__init__(
            message=f"MCP server not available: {server_name}",
            server_name=server_name
        )
        self.error_code = "KGRAG_MCP_SERVER_UNAVAILABLE"


class MCPToolError(MCPServerError):
    """Raised when MCP tool execution fails."""
    
    def __init__(self, message: str, tool_name: str, server_name: str):
        super().__init__(
            message=message,
            server_name=server_name
        )
        self.error_code = "KGRAG_MCP_TOOL_ERROR"
        self.details.update({"tool_name": tool_name})


# =============================================================================
# Graph Database Exceptions
# =============================================================================

class GraphDatabaseError(KGRAGException):
    """Base exception for graph database operations."""
    
    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_GRAPH_DB_ERROR",
            details={"query": query} if query else None
        )


class GraphQueryError(GraphDatabaseError):
    """Raised when graph query execution fails."""
    
    def __init__(self, message: str, cypher_query: str, parameters: Optional[Dict] = None):
        super().__init__(
            message=message,
            query=cypher_query
        )
        self.error_code = "KGRAG_GRAPH_QUERY_ERROR"
        self.details.update({"parameters": parameters} if parameters else {})


class VectorIndexError(GraphDatabaseError):
    """Raised when vector index operations fail."""
    
    def __init__(self, message: str, index_name: str, operation: str):
        super().__init__(
            message=message
        )
        self.error_code = "KGRAG_VECTOR_INDEX_ERROR"
        self.details.update({
            "index_name": index_name,
            "operation": operation
        })


# =============================================================================
# Search and Retrieval Exceptions
# =============================================================================

class SearchError(KGRAGException):
    """Base exception for search operations."""
    
    def __init__(self, message: str, search_type: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_SEARCH_ERROR",
            details={"search_type": search_type} if search_type else None
        )


class VectorSearchError(SearchError):
    """Raised when vector search fails."""
    
    def __init__(self, message: str, vector_dimension: Optional[int] = None):
        super().__init__(
            message=message,
            search_type="vector"
        )
        self.error_code = "KGRAG_VECTOR_SEARCH_ERROR"
        self.details.update({"vector_dimension": vector_dimension} if vector_dimension else {})


class HybridSearchError(SearchError):
    """Raised when hybrid search fails."""
    
    def __init__(self, message: str, graph_error: Optional[str] = None, vector_error: Optional[str] = None):
        super().__init__(
            message=message,
            search_type="hybrid"
        )
        self.error_code = "KGRAG_HYBRID_SEARCH_ERROR"
        self.details.update({
            "graph_error": graph_error,
            "vector_error": vector_error
        })


# =============================================================================
# Document Processing Exceptions
# =============================================================================

class DocumentProcessingError(KGRAGException):
    """Base exception for document processing operations."""
    
    def __init__(self, message: str, document_path: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_DOCUMENT_PROCESSING_ERROR",
            details={"document_path": document_path} if document_path else None
        )


class DocumentParsingError(DocumentProcessingError):
    """Raised when document parsing fails."""
    
    def __init__(self, message: str, document_path: str, document_type: str):
        super().__init__(
            message=message,
            document_path=document_path
        )
        self.error_code = "KGRAG_DOCUMENT_PARSING_ERROR"
        self.details.update({"document_type": document_type})


class ChunkingError(DocumentProcessingError):
    """Raised when document chunking fails."""
    
    def __init__(self, message: str, document_path: str, chunk_index: Optional[int] = None):
        super().__init__(
            message=message,
            document_path=document_path
        )
        self.error_code = "KGRAG_CHUNKING_ERROR"
        self.details.update({"chunk_index": chunk_index} if chunk_index is not None else {})


class EmbeddingGenerationError(DocumentProcessingError):
    """Raised when embedding generation fails."""
    
    def __init__(self, message: str, text_length: Optional[int] = None, model_name: Optional[str] = None):
        super().__init__(
            message=message
        )
        self.error_code = "KGRAG_EMBEDDING_GENERATION_ERROR"
        self.details.update({
            "text_length": text_length,
            "model_name": model_name
        })


# =============================================================================
# Security and Compliance Exceptions
# =============================================================================

class SecurityError(KGRAGException):
    """Base exception for security-related errors."""
    
    def __init__(self, message: str, security_domain: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_SECURITY_ERROR",
            details={"security_domain": security_domain} if security_domain else None
        )


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, user_id: Optional[str] = None):
        super().__init__(
            message=message,
            security_domain="authentication"
        )
        self.error_code = "KGRAG_AUTHENTICATION_ERROR"
        self.details.update({"user_id": user_id} if user_id else {})


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str, user_id: str, resource: str, action: str):
        super().__init__(
            message=message,
            security_domain="authorization"
        )
        self.error_code = "KGRAG_AUTHORIZATION_ERROR"
        self.details.update({
            "user_id": user_id,
            "resource": resource,
            "action": action
        })


class ComplianceError(SecurityError):
    """Raised when compliance requirements are violated."""
    
    def __init__(self, message: str, compliance_framework: str, control_id: Optional[str] = None):
        super().__init__(
            message=message,
            security_domain="compliance"
        )
        self.error_code = "KGRAG_COMPLIANCE_ERROR"
        self.details.update({
            "compliance_framework": compliance_framework,
            "control_id": control_id
        })


class DataPrivacyError(SecurityError):
    """Raised when data privacy requirements are violated."""
    
    def __init__(self, message: str, data_type: str, privacy_regulation: Optional[str] = None):
        super().__init__(
            message=message,
            security_domain="data_privacy"
        )
        self.error_code = "KGRAG_DATA_PRIVACY_ERROR"
        self.details.update({
            "data_type": data_type,
            "privacy_regulation": privacy_regulation
        })


# =============================================================================
# API and Communication Exceptions
# =============================================================================

class APIError(KGRAGException):
    """Base exception for API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_API_ERROR",
            details={"status_code": status_code} if status_code else None
        )


class ValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str, field_errors: Dict[str, str]):
        super().__init__(
            message=message,
            status_code=400
        )
        self.error_code = "KGRAG_VALIDATION_ERROR"
        self.details.update({"field_errors": field_errors})


class RateLimitError(APIError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, limit: int, window: int):
        super().__init__(
            message=message,
            status_code=429
        )
        self.error_code = "KGRAG_RATE_LIMIT_ERROR"
        self.details.update({
            "limit": limit,
            "window": window
        })


# =============================================================================
# Performance and Resource Exceptions
# =============================================================================

class PerformanceError(KGRAGException):
    """Base exception for performance-related errors."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(
            message=message,
            error_code="KGRAG_PERFORMANCE_ERROR",
            details={"operation": operation} if operation else None
        )


class TimeoutError(PerformanceError):
    """Raised when operations exceed timeout limits."""
    
    def __init__(self, message: str, operation: str, timeout_seconds: float):
        super().__init__(
            message=message,
            operation=operation
        )
        self.error_code = "KGRAG_TIMEOUT_ERROR"
        self.details.update({"timeout_seconds": timeout_seconds})


class ResourceExhaustionError(PerformanceError):
    """Raised when system resources are exhausted."""
    
    def __init__(self, message: str, resource_type: str, current_usage: float, limit: float):
        super().__init__(
            message=message
        )
        self.error_code = "KGRAG_RESOURCE_EXHAUSTION_ERROR"
        self.details.update({
            "resource_type": resource_type,
            "current_usage": current_usage,
            "limit": limit
        })


class MemoryError(ResourceExhaustionError):
    """Raised when memory limits are exceeded."""
    
    def __init__(self, message: str, current_usage_mb: float, limit_mb: float):
        super().__init__(
            message=message,
            resource_type="memory",
            current_usage=current_usage_mb,
            limit=limit_mb
        )
        self.error_code = "KGRAG_MEMORY_ERROR"


# =============================================================================
# Utility Functions
# =============================================================================

def handle_exception(exception: Exception, context: Optional[Dict[str, Any]] = None) -> KGRAGException:
    """
    Convert generic exceptions to KGRAGException with context.
    
    Args:
        exception: Original exception
        context: Additional context information
        
    Returns:
        KGRAGException with proper error handling
    """
    if isinstance(exception, KGRAGException):
        return exception
    
    # Map common exception types
    if isinstance(exception, ConnectionError):
        return DatabaseConnectionError(
            message=str(exception),
            database_type="unknown",
            connection_uri=context.get("connection_uri") if context else None
        )
    elif isinstance(exception, TimeoutError):
        return TimeoutError(
            message=str(exception),
            operation=context.get("operation", "unknown") if context else "unknown",
            timeout_seconds=context.get("timeout", 0.0) if context else 0.0
        )
    elif isinstance(exception, MemoryError):
        return MemoryError(
            message=str(exception),
            current_usage_mb=context.get("current_usage", 0.0) if context else 0.0,
            limit_mb=context.get("limit", 0.0) if context else 0.0
        )
    else:
        return KGRAGException(
            message=str(exception),
            error_code="KGRAG_WRAPPED_EXCEPTION",
            details=context or {},
            cause=exception
        )