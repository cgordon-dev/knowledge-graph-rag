"""
Knowledge Graph-RAG System with AI Digital Twins

An offline, secure knowledge graph system that combines graph databases with
vector search and AI Digital Twins for intelligent document analysis and
compliance management.

Features:
- Offline-first architecture for data security
- AI Digital Twins for persona-driven interactions
- Vector embeddings for every graph component
- MCP server orchestration for streamlined operations
- Google ADK AI agent framework integration
- Neo4j graph database with vector indexes
- FedRAMP compliance support
"""

__version__ = "1.0.0"
__author__ = "AI Systems Team"
__email__ = "ai-systems@company.com"
__license__ = "MIT"
__description__ = "Offline Knowledge Graph-RAG System with AI Digital Twins"

# Core exports
from kg_rag.core.config import Settings, get_settings
from kg_rag.core.logger import get_logger
from kg_rag.core.exceptions import KGRAGException

# AI Digital Twins exports
from kg_rag.ai_twins.persona_twin import PersonaTwin
from kg_rag.ai_twins.expert_twin import ExpertTwin
from kg_rag.ai_twins.user_journey_twin import UserJourneyTwin

# MCP Server exports
from kg_rag.mcp_servers.orchestrator import MCPOrchestrator
from kg_rag.mcp_servers.knowledge_graph_mcp import KnowledgeGraphMCP
from kg_rag.mcp_servers.vector_search_mcp import VectorSearchMCP

# Agent exports
from kg_rag.agents.query_understanding_agent import QueryUnderstandingAgent
from kg_rag.agents.knowledge_synthesis_agent import KnowledgeSynthesisAgent

__all__ = [
    # Core
    "Settings",
    "get_settings", 
    "get_logger",
    "KGRAGException",
    
    # AI Digital Twins
    "PersonaTwin",
    "ExpertTwin", 
    "UserJourneyTwin",
    
    # MCP Servers
    "MCPOrchestrator",
    "KnowledgeGraphMCP",
    "VectorSearchMCP",
    
    # Agents
    "QueryUnderstandingAgent",
    "KnowledgeSynthesisAgent",
]

# Version info tuple
VERSION_INFO = tuple(map(int, __version__.split('.')))

# Package metadata
PACKAGE_DATA = {
    "name": "knowledge-graph-rag",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "author_email": __email__,
    "license": __license__,
    "url": "https://github.com/company/knowledge-graph-rag",
    "requires_python": ">=3.11",
}

def get_version() -> str:
    """Get the package version."""
    return __version__

def get_package_info() -> dict:
    """Get package metadata."""
    return PACKAGE_DATA.copy()