"""
MCP (Model Context Protocol) Server Framework for Knowledge Graph-RAG.

Provides streamlined integration between AI agents and system components
through specialized MCP servers for different domains.
"""

from kg_rag.mcp_servers.orchestrator import MCPOrchestrator
from kg_rag.mcp_servers.knowledge_graph_mcp import KnowledgeGraphMCP
from kg_rag.mcp_servers.vector_search_mcp import VectorSearchMCP
from kg_rag.mcp_servers.document_processing_mcp import DocumentProcessingMCP
from kg_rag.mcp_servers.analytics_mcp import AnalyticsMCP

__all__ = [
    "MCPOrchestrator",
    "KnowledgeGraphMCP",
    "VectorSearchMCP", 
    "DocumentProcessingMCP",
    "AnalyticsMCP"
]