"""
Structured logging configuration for Knowledge Graph-RAG system.

Provides secure, compliant logging with audit trail capabilities and
structured output for monitoring and debugging.
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import structlog
from rich.console import Console
from rich.logging import RichHandler

from kg_rag.core.config import get_settings


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs."""
    
    SENSITIVE_FIELDS = {
        'password', 'token', 'secret', 'key', 'api_key', 'jwt',
        'authorization', 'auth', 'credential', 'private_key'
    }
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive information from log records."""
        if hasattr(record, 'msg') and isinstance(record.msg, (dict, str)):
            record.msg = self._sanitize_message(record.msg)
        
        if hasattr(record, 'args') and record.args:
            record.args = tuple(self._sanitize_message(arg) for arg in record.args)
        
        return True
    
    def _sanitize_message(self, message: Any) -> Any:
        """Sanitize sensitive information from message."""
        if isinstance(message, dict):
            return {
                key: "***REDACTED***" if any(sensitive in key.lower() 
                                           for sensitive in self.SENSITIVE_FIELDS)
                else value
                for key, value in message.items()
            }
        elif isinstance(message, str):
            # Basic string sanitization for common patterns
            for field in self.SENSITIVE_FIELDS:
                if field.lower() in message.lower():
                    return message.replace(message, "***REDACTED***")
        
        return message


class AuditLogger:
    """Specialized logger for audit events with compliance features."""
    
    def __init__(self, logger_name: str = "audit"):
        """Initialize audit logger."""
        self.logger = structlog.get_logger(logger_name)
        self.settings = get_settings()
    
    def log_authentication(
        self, 
        user_id: str, 
        action: str, 
        success: bool, 
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log authentication events."""
        self.logger.info(
            "authentication_event",
            user_id=user_id,
            action=action,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.utcnow().isoformat(),
            event_type="authentication",
            **(additional_data or {})
        )
    
    def log_data_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        success: bool,
        resource_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log data access events."""
        self.logger.info(
            "data_access_event",
            user_id=user_id,
            resource=resource,
            resource_id=resource_id,
            action=action,
            success=success,
            timestamp=datetime.utcnow().isoformat(),
            event_type="data_access",
            **(additional_data or {})
        )
    
    def log_system_event(
        self,
        event_type: str,
        description: str,
        severity: str = "info",
        component: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log system events."""
        self.logger.info(
            "system_event",
            event_type=event_type,
            description=description,
            severity=severity,
            component=component,
            timestamp=datetime.utcnow().isoformat(),
            **(additional_data or {})
        )
    
    def log_compliance_event(
        self,
        control_id: str,
        action: str,
        result: str,
        user_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log compliance-related events."""
        self.logger.info(
            "compliance_event",
            control_id=control_id,
            action=action,
            result=result,
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            event_type="compliance",
            **(additional_data or {})
        )


class PerformanceLogger:
    """Logger for performance metrics and monitoring."""
    
    def __init__(self, logger_name: str = "performance"):
        """Initialize performance logger."""
        self.logger = structlog.get_logger(logger_name)
    
    def log_query_performance(
        self,
        query_type: str,
        duration_ms: float,
        result_count: int,
        user_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log query performance metrics."""
        self.logger.info(
            "query_performance",
            query_type=query_type,
            duration_ms=duration_ms,
            result_count=result_count,
            user_id=user_id,
            persona_id=persona_id,
            timestamp=datetime.utcnow().isoformat(),
            **(additional_metrics or {})
        )
    
    def log_model_performance(
        self,
        model_name: str,
        operation: str,
        duration_ms: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log AI model performance metrics."""
        self.logger.info(
            "model_performance",
            model_name=model_name,
            operation=operation,
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timestamp=datetime.utcnow().isoformat(),
            **(additional_metrics or {})
        )
    
    def log_system_performance(
        self,
        component: str,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        disk_usage: Optional[float] = None,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log system performance metrics."""
        self.logger.info(
            "system_performance",
            component=component,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            timestamp=datetime.utcnow().isoformat(),
            **(additional_metrics or {})
        )


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure structlog
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
            structlog.processors.JSONRenderer() if settings.monitoring.log_format == "json"
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.monitoring.log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure handlers based on environment
    if settings.is_development():
        _setup_development_logging(root_logger, settings)
    else:
        _setup_production_logging(root_logger, settings)
    
    # Add security filter to all handlers
    security_filter = SecurityFilter()
    for handler in root_logger.handlers:
        handler.addFilter(security_filter)


def _setup_development_logging(root_logger: logging.Logger, settings) -> None:
    """Setup logging for development environment."""
    # Console handler with rich formatting
    console = Console()
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=settings.debug
    )
    console_handler.setLevel(logging.DEBUG if settings.debug else logging.INFO)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)


def _setup_production_logging(root_logger: logging.Logger, settings) -> None:
    """Setup logging for production environment."""
    log_dir = Path("/app/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Application log handler
    app_log_file = log_dir / "application.log"
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_file,
        maxBytes=100 * 1024 * 1024,  # 100MB
        backupCount=10
    )
    app_handler.setLevel(getattr(logging, settings.monitoring.log_level.upper()))
    
    # Audit log handler (separate file for compliance)
    if settings.monitoring.audit_log_enabled:
        audit_log_file = log_dir / "audit.log"
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=20
        )
        audit_handler.setLevel(logging.INFO)
        
        # Add audit handler for audit logger only
        audit_logger = logging.getLogger("audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.propagate = False
    
    # JSON formatter for production
    if settings.monitoring.log_format == "json":
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                if record.exc_info:
                    log_entry["exception"] = self.formatException(record.exc_info)
                
                return json.dumps(log_entry)
        
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    app_handler.setFormatter(formatter)
    root_logger.addHandler(app_handler)
    
    # Console handler for critical errors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def get_audit_logger() -> AuditLogger:
    """
    Get an audit logger instance.
    
    Returns:
        Audit logger instance
    """
    return AuditLogger()


def get_performance_logger() -> PerformanceLogger:
    """
    Get a performance logger instance.
    
    Returns:
        Performance logger instance
    """
    return PerformanceLogger()


# Setup logging on module import
setup_logging()

# Create default loggers
logger = get_logger(__name__)
audit_logger = get_audit_logger()
performance_logger = get_performance_logger()