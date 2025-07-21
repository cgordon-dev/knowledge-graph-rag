"""
Base MCP server implementation for Knowledge Graph-RAG system.

Provides common functionality and patterns for all MCP servers with
security, monitoring, and error handling.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

import structlog
from pydantic import BaseModel, Field

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import MCPServerError, MCPToolError, handle_exception
from kg_rag.core.logger import get_performance_logger


class MCPTool(BaseModel):
    """MCP tool definition."""
    
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters schema")
    handler: Optional[Callable] = Field(None, description="Tool handler function")
    
    class Config:
        arbitrary_types_allowed = True


class MCPToolResult(BaseModel):
    """MCP tool execution result."""
    
    success: bool = Field(..., description="Whether tool execution succeeded")
    result: Any = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class MCPServerMetrics(BaseModel):
    """MCP server performance metrics."""
    
    total_requests: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    average_response_time_ms: float = Field(default=0.0, description="Average response time")
    uptime_seconds: float = Field(default=0.0, description="Server uptime")
    last_request_time: Optional[datetime] = Field(None, description="Last request timestamp")
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class BaseMCPServer(ABC):
    """Base class for all MCP servers."""
    
    def __init__(self, name: str, description: str, port: Optional[int] = None):
        """
        Initialize base MCP server.
        
        Args:
            name: Server name
            description: Server description
            port: Server port (optional)
        """
        self.name = name
        self.description = description
        self.port = port
        self.settings = get_settings()
        self.logger = structlog.get_logger(f"mcp.{name}")
        self.performance_logger = get_performance_logger()
        
        # Server state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.tools: Dict[str, MCPTool] = {}
        self.metrics = MCPServerMetrics()
        
        # Register tools on initialization
        self.register_tools()
    
    @abstractmethod
    def register_tools(self) -> None:
        """Register available tools for this server."""
        pass
    
    def tool(self, name: str, description: str = "", parameters: Optional[Dict] = None):
        """Decorator to register a tool method."""
        def decorator(func: Callable) -> Callable:
            tool_def = MCPTool(
                name=name,
                description=description or func.__doc__ or f"Tool: {name}",
                parameters=parameters or {},
                handler=func
            )
            self.tools[name] = tool_def
            return func
        return decorator
    
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> MCPToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        start_time = datetime.utcnow()
        
        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.last_request_time = start_time
        
        try:
            # Validate tool exists
            if tool_name not in self.tools:
                raise MCPToolError(
                    f"Tool '{tool_name}' not found in server '{self.name}'",
                    tool_name=tool_name,
                    server_name=self.name
                )
            
            tool = self.tools[tool_name]
            
            # Validate parameters
            validated_params = self._validate_parameters(tool, parameters or {})
            
            # Execute tool
            self.logger.info(
                "Executing tool",
                tool_name=tool_name,
                server_name=self.name,
                parameters=validated_params
            )
            
            # Call the tool handler
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**validated_params)
            else:
                result = tool.handler(**validated_params)
            
            # Calculate execution time
            end_time = datetime.utcnow()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update metrics
            self.metrics.successful_requests += 1
            self._update_average_response_time(execution_time_ms)
            
            # Log performance
            self.performance_logger.log_query_performance(
                query_type=f"mcp_tool_{tool_name}",
                duration_ms=execution_time_ms,
                result_count=1 if result else 0,
                additional_metrics={
                    "server_name": self.name,
                    "tool_name": tool_name
                }
            )
            
            return MCPToolResult(
                success=True,
                result=result,
                execution_time_ms=execution_time_ms,
                metadata={
                    "server_name": self.name,
                    "tool_name": tool_name,
                    "timestamp": start_time.isoformat()
                }
            )
            
        except Exception as e:
            # Calculate execution time for failed requests
            end_time = datetime.utcnow()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update metrics
            self.metrics.failed_requests += 1
            self._update_average_response_time(execution_time_ms)
            
            # Handle and log error
            kg_error = handle_exception(e, {
                "server_name": self.name,
                "tool_name": tool_name,
                "parameters": parameters
            })
            
            self.logger.error(
                "Tool execution failed",
                tool_name=tool_name,
                server_name=self.name,
                error=str(kg_error),
                execution_time_ms=execution_time_ms
            )
            
            return MCPToolResult(
                success=False,
                error=str(kg_error),
                execution_time_ms=execution_time_ms,
                metadata={
                    "server_name": self.name,
                    "tool_name": tool_name,
                    "timestamp": start_time.isoformat(),
                    "error_code": kg_error.error_code
                }
            )
    
    def _validate_parameters(self, tool: MCPTool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool parameters against schema.
        
        Args:
            tool: Tool definition
            parameters: Parameters to validate
            
        Returns:
            Validated parameters
        """
        # Basic validation - in a full implementation, use JSON Schema or Pydantic
        if not tool.parameters:
            return parameters
        
        validated = {}
        
        for param_name, param_config in tool.parameters.items():
            if param_config.get('required', False) and param_name not in parameters:
                raise MCPToolError(
                    f"Required parameter '{param_name}' missing for tool '{tool.name}'",
                    tool_name=tool.name,
                    server_name=self.name
                )
            
            if param_name in parameters:
                validated[param_name] = parameters[param_name]
        
        return validated
    
    def _update_average_response_time(self, execution_time_ms: float) -> None:
        """Update average response time metric."""
        total_requests = self.metrics.total_requests
        current_avg = self.metrics.average_response_time_ms
        
        # Calculate new average
        self.metrics.average_response_time_ms = (
            (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
        )
    
    async def start(self) -> None:
        """Start the MCP server."""
        if self.is_running:
            self.logger.warning("Server already running", server_name=self.name)
            return
        
        self.logger.info("Starting MCP server", server_name=self.name, port=self.port)
        
        try:
            await self._initialize()
            self.is_running = True
            self.start_time = datetime.utcnow()
            
            self.logger.info(
                "MCP server started successfully",
                server_name=self.name,
                port=self.port,
                tools_count=len(self.tools)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to start MCP server",
                server_name=self.name,
                error=str(e)
            )
            raise MCPServerError(f"Failed to start server '{self.name}': {e}")
    
    async def stop(self) -> None:
        """Stop the MCP server."""
        if not self.is_running:
            self.logger.warning("Server not running", server_name=self.name)
            return
        
        self.logger.info("Stopping MCP server", server_name=self.name)
        
        try:
            await self._cleanup()
            self.is_running = False
            
            self.logger.info(
                "MCP server stopped successfully",
                server_name=self.name,
                uptime_seconds=self.get_uptime_seconds()
            )
            
        except Exception as e:
            self.logger.error(
                "Error stopping MCP server",
                server_name=self.name,
                error=str(e)
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status information
        """
        health_status = {
            "server_name": self.name,
            "status": "healthy" if self.is_running else "stopped",
            "uptime_seconds": self.get_uptime_seconds(),
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate(),
                "average_response_time_ms": self.metrics.average_response_time_ms
            },
            "tools": list(self.tools.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add custom health checks
        try:
            custom_health = await self._custom_health_check()
            health_status.update(custom_health)
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status
    
    def get_uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        if not self.start_time:
            return 0.0
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def get_tools_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available tools."""
        return {
            name: {
                "description": tool.description,
                "parameters": tool.parameters
            }
            for name, tool in self.tools.items()
        }
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize server-specific resources."""
        pass
    
    @abstractmethod
    async def _cleanup(self) -> None:
        """Cleanup server-specific resources."""
        pass
    
    async def _custom_health_check(self) -> Dict[str, Any]:
        """Override for custom health checks."""
        return {}
    
    def __repr__(self) -> str:
        """String representation of the server."""
        return f"<{self.__class__.__name__}(name='{self.name}', port={self.port}, running={self.is_running})>"