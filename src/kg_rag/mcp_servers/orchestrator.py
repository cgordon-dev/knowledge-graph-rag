"""
MCP Server Orchestrator for Knowledge Graph-RAG system.

Coordinates multiple MCP servers, provides unified interface,
and handles server lifecycle management with health monitoring.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import structlog

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import MCPServerError, MCPServerNotAvailableError
from kg_rag.core.logger import get_performance_logger
from kg_rag.mcp_servers.base_mcp import MCPToolResult
from kg_rag.mcp_servers.knowledge_graph_mcp import KnowledgeGraphMCP
from kg_rag.mcp_servers.vector_search_mcp import VectorSearchMCP


class MCPOrchestrator:
    """
    Orchestrates multiple MCP servers and provides unified access.
    
    Manages server lifecycle, health monitoring, load balancing,
    and provides a single interface for AI agents.
    """
    
    def __init__(self):
        """Initialize MCP Orchestrator."""
        self.settings = get_settings()
        self.logger = structlog.get_logger("mcp_orchestrator")
        self.performance_logger = get_performance_logger()
        
        # Server instances
        self.servers: Dict[str, Any] = {}
        self.server_health: Dict[str, Dict[str, Any]] = {}
        
        # Orchestrator state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        
        # Health monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
        # Initialize servers
        self._initialize_servers()
    
    def _initialize_servers(self) -> None:
        """Initialize all MCP servers."""
        try:
            # Core servers
            self.servers = {
                'knowledge_graph': KnowledgeGraphMCP(),
                'vector_search': VectorSearchMCP(),
                # Document processing and analytics servers would be added here
                # 'document_processing': DocumentProcessingMCP(),
                # 'analytics': AnalyticsMCP(),
            }
            
            # Initialize health status
            for server_name in self.servers:
                self.server_health[server_name] = {
                    "status": "stopped",
                    "last_check": None,
                    "error_count": 0,
                    "response_time_ms": 0.0
                }
            
            self.logger.info(
                "MCP servers initialized",
                server_count=len(self.servers),
                servers=list(self.servers.keys())
            )
            
        except Exception as e:
            self.logger.error("Failed to initialize MCP servers", error=str(e))
            raise MCPServerError(f"Server initialization failed: {e}")
    
    async def start_all_servers(self) -> None:
        """Start all MCP servers."""
        if self.is_running:
            self.logger.warning("Orchestrator already running")
            return
        
        self.logger.info("Starting MCP server orchestrator")
        
        try:
            # Start servers concurrently
            start_tasks = []
            for server_name, server in self.servers.items():
                task = asyncio.create_task(
                    self._start_server_with_retry(server_name, server)
                )
                start_tasks.append(task)
            
            # Wait for all servers to start
            await asyncio.gather(*start_tasks, return_exceptions=True)
            
            # Check which servers started successfully
            healthy_servers = []
            for server_name, server in self.servers.items():
                if server.is_running:
                    healthy_servers.append(server_name)
                    self.server_health[server_name]["status"] = "healthy"
            
            if not healthy_servers:
                raise MCPServerError("No servers started successfully")
            
            self.is_running = True
            self.start_time = datetime.utcnow()
            
            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_monitor_loop())
            
            self.logger.info(
                "MCP orchestrator started",
                healthy_servers=healthy_servers,
                failed_servers=[name for name in self.servers if name not in healthy_servers]
            )
            
        except Exception as e:
            self.logger.error("Failed to start MCP orchestrator", error=str(e))
            raise MCPServerError(f"Orchestrator startup failed: {e}")
    
    async def _start_server_with_retry(self, server_name: str, server: Any, max_retries: int = 3) -> None:
        """Start a server with retry logic."""
        for attempt in range(max_retries):
            try:
                await server.start()
                self.logger.info(f"Server {server_name} started successfully")
                return
            except Exception as e:
                self.logger.warning(
                    f"Server {server_name} start attempt {attempt + 1} failed",
                    error=str(e)
                )
                if attempt == max_retries - 1:
                    self.logger.error(f"Server {server_name} failed to start after {max_retries} attempts")
                    self.server_health[server_name]["status"] = "failed"
                else:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def stop_all_servers(self) -> None:
        """Stop all MCP servers."""
        if not self.is_running:
            self.logger.warning("Orchestrator not running")
            return
        
        self.logger.info("Stopping MCP server orchestrator")
        
        try:
            # Stop health monitoring
            if self.health_check_task:
                self.health_check_task.cancel()
                try:
                    await self.health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop servers concurrently
            stop_tasks = []
            for server_name, server in self.servers.items():
                if server.is_running:
                    task = asyncio.create_task(server.stop())
                    stop_tasks.append(task)
            
            if stop_tasks:
                await asyncio.gather(*stop_tasks, return_exceptions=True)
            
            self.is_running = False
            
            # Update health status
            for server_name in self.servers:
                self.server_health[server_name]["status"] = "stopped"
            
            self.logger.info("MCP orchestrator stopped successfully")
            
        except Exception as e:
            self.logger.error("Error stopping MCP orchestrator", error=str(e))
    
    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> MCPToolResult:
        """
        Execute a tool on a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            timeout: Execution timeout in seconds
            
        Returns:
            Tool execution result
        """
        start_time = datetime.utcnow()
        self.request_count += 1
        
        try:
            # Validate server availability
            if server_name not in self.servers:
                raise MCPServerNotAvailableError(server_name)
            
            server = self.servers[server_name]
            
            if not server.is_running:
                raise MCPServerNotAvailableError(server_name)
            
            # Execute tool with timeout
            if timeout:
                result = await asyncio.wait_for(
                    server.execute_tool(tool_name, parameters),
                    timeout=timeout
                )
            else:
                result = await server.execute_tool(tool_name, parameters)
            
            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.total_response_time += execution_time
            
            # Log performance
            self.performance_logger.log_query_performance(
                query_type=f"mcp_{server_name}_{tool_name}",
                duration_ms=execution_time,
                result_count=1,
                additional_metrics={
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "orchestrator": True
                }
            )
            
            return result
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(
                "Tool execution failed",
                server_name=server_name,
                tool_name=tool_name,
                error=str(e)
            )
            
            # Return error result
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            return MCPToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
                metadata={
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "orchestrator_error": True
                }
            )
    
    async def get_server_health(self, server_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health status for servers.
        
        Args:
            server_name: Specific server name, or None for all servers
            
        Returns:
            Health status information
        """
        if server_name:
            if server_name not in self.servers:
                return {"error": f"Server '{server_name}' not found"}
            
            server = self.servers[server_name]
            try:
                health = await server.health_check()
                self.server_health[server_name].update({
                    "status": "healthy",
                    "last_check": datetime.utcnow().isoformat(),
                    "details": health
                })
                return self.server_health[server_name]
            except Exception as e:
                self.server_health[server_name].update({
                    "status": "unhealthy",
                    "last_check": datetime.utcnow().isoformat(),
                    "error": str(e)
                })
                return self.server_health[server_name]
        else:
            return {
                "orchestrator_status": "running" if self.is_running else "stopped",
                "uptime_seconds": self.get_uptime_seconds(),
                "servers": self.server_health.copy(),
                "metrics": {
                    "total_requests": self.request_count,
                    "error_rate": self.error_count / max(self.request_count, 1) * 100,
                    "average_response_time_ms": self.total_response_time / max(self.request_count, 1)
                }
            }
    
    async def get_available_tools(self, server_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get available tools from servers.
        
        Args:
            server_name: Specific server name, or None for all servers
            
        Returns:
            Available tools information
        """
        if server_name:
            if server_name not in self.servers:
                return {"error": f"Server '{server_name}' not found"}
            
            server = self.servers[server_name]
            return {
                "server_name": server_name,
                "tools": server.get_tools_info()
            }
        else:
            all_tools = {}
            for name, server in self.servers.items():
                if server.is_running:
                    all_tools[name] = server.get_tools_info()
            
            return {
                "total_servers": len(self.servers),
                "running_servers": len([s for s in self.servers.values() if s.is_running]),
                "tools_by_server": all_tools
            }
    
    async def _health_monitor_loop(self) -> None:
        """Background task for monitoring server health."""
        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # Check each server's health
                for server_name, server in self.servers.items():
                    try:
                        start_time = datetime.utcnow()
                        health = await server.health_check()
                        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                        
                        self.server_health[server_name].update({
                            "status": "healthy",
                            "last_check": datetime.utcnow().isoformat(),
                            "response_time_ms": response_time,
                            "error_count": 0
                        })
                        
                    except Exception as e:
                        error_count = self.server_health[server_name].get("error_count", 0) + 1
                        self.server_health[server_name].update({
                            "status": "unhealthy",
                            "last_check": datetime.utcnow().isoformat(),
                            "error": str(e),
                            "error_count": error_count
                        })
                        
                        self.logger.warning(
                            "Server health check failed",
                            server_name=server_name,
                            error=str(e),
                            error_count=error_count
                        )
                        
                        # Restart server if it failed multiple times
                        if error_count >= 3:
                            self.logger.info(f"Attempting to restart unhealthy server: {server_name}")
                            try:
                                await server.stop()
                                await server.start()
                                self.server_health[server_name]["error_count"] = 0
                            except Exception as restart_error:
                                self.logger.error(
                                    f"Failed to restart server {server_name}",
                                    error=str(restart_error)
                                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("Health monitor error", error=str(e))
    
    def get_uptime_seconds(self) -> float:
        """Get orchestrator uptime in seconds."""
        if not self.start_time:
            return 0.0
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    async def graceful_shutdown(self, timeout: float = 30.0) -> None:
        """
        Perform graceful shutdown of the orchestrator.
        
        Args:
            timeout: Maximum time to wait for shutdown
        """
        self.logger.info("Starting graceful shutdown")
        
        try:
            # Stop accepting new requests (this would be implemented in a full server)
            # self.accepting_requests = False
            
            # Wait for current requests to complete or timeout
            await asyncio.wait_for(self.stop_all_servers(), timeout=timeout)
            
        except asyncio.TimeoutError:
            self.logger.warning("Graceful shutdown timeout, forcing shutdown")
            # Force stop servers
            for server in self.servers.values():
                try:
                    await server.stop()
                except Exception as e:
                    self.logger.error(f"Error force stopping server: {e}")
        
        self.logger.info("Graceful shutdown completed")
    
    def __repr__(self) -> str:
        """String representation of the orchestrator."""
        return f"<MCPOrchestrator(servers={len(self.servers)}, running={self.is_running})>"


# Global orchestrator instance
_orchestrator: Optional[MCPOrchestrator] = None


def get_orchestrator() -> MCPOrchestrator:
    """
    Get the global MCP orchestrator instance.
    
    Returns:
        MCPOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MCPOrchestrator()
    return _orchestrator


async def startup_orchestrator() -> MCPOrchestrator:
    """
    Start the MCP orchestrator.
    
    Returns:
        Started orchestrator instance
    """
    orchestrator = get_orchestrator()
    if not orchestrator.is_running:
        await orchestrator.start_all_servers()
    return orchestrator


async def shutdown_orchestrator() -> None:
    """Shutdown the MCP orchestrator."""
    global _orchestrator
    if _orchestrator and _orchestrator.is_running:
        await _orchestrator.graceful_shutdown()
        _orchestrator = None