"""
Configuration management for Knowledge Graph-RAG system.

Handles environment variables, settings validation, and configuration loading
with security and compliance considerations.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings, Field, validator
from pydantic_settings import SettingsConfigDict


class SecuritySettings(BaseSettings):
    """Security-related configuration settings."""
    
    # JWT Configuration
    jwt_secret_key: str = Field(..., description="Secret key for JWT tokens")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(default=1440, description="Access token expiration")
    jwt_refresh_token_expire_days: int = Field(default=7, description="Refresh token expiration")
    
    # Encryption Configuration
    encryption_key: str = Field(..., description="Encryption key for sensitive data")
    salt_rounds: int = Field(default=12, description="BCrypt salt rounds")
    
    # API Security
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    # Compliance
    fedramp_compliance_mode: bool = Field(default=True, description="Enable FedRAMP compliance")
    audit_log_encryption: bool = Field(default=True, description="Encrypt audit logs")
    data_retention_years: int = Field(default=7, description="Data retention period")
    
    @validator('jwt_secret_key')
    def validate_jwt_secret(cls, v):
        if len(v) < 32:
            raise ValueError('JWT secret key must be at least 32 characters')
        return v
    
    @validator('encryption_key')
    def validate_encryption_key(cls, v):
        if len(v.encode()) != 32:
            raise ValueError('Encryption key must be exactly 32 bytes')
        return v


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(..., description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    neo4j_max_connection_lifetime: int = Field(default=3600, description="Max connection lifetime")
    neo4j_max_connection_pool_size: int = Field(default=50, description="Max connection pool size")
    neo4j_connection_timeout: int = Field(default=30, description="Connection timeout")
    
    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    cache_ttl: int = Field(default=3600, description="Default cache TTL")
    query_cache_size: int = Field(default=1000, description="Query cache size")
    result_cache_size: int = Field(default=5000, description="Result cache size")


class AIModelSettings(BaseSettings):
    """AI/ML model configuration settings."""
    
    # Embedding Model
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", description="Embedding model name")
    embedding_device: str = Field(default="cpu", description="Device for embedding model")
    embedding_batch_size: int = Field(default=32, description="Embedding batch size")
    embedding_max_length: int = Field(default=512, description="Max embedding length")
    
    # Google ADK Configuration
    google_adk_model_path: str = Field(default="/app/models/gemini-offline", description="ADK model path")
    google_adk_api_key: str = Field(default="offline_placeholder", description="ADK API key")
    google_adk_project_id: str = Field(default="offline_project", description="ADK project ID")
    google_adk_temperature: float = Field(default=0.1, description="ADK temperature")
    google_adk_max_tokens: int = Field(default=2048, description="ADK max tokens")
    
    # Model Cache
    model_cache_dir: str = Field(default="/app/models", description="Model cache directory")
    embedding_cache_size: int = Field(default=10000, description="Embedding cache size")
    vector_index_cache_size: int = Field(default=50000, description="Vector index cache size")


class MCPServerSettings(BaseSettings):
    """MCP server configuration settings."""
    
    # Main MCP Server
    mcp_server_host: str = Field(default="localhost", description="MCP server host")
    mcp_server_port: int = Field(default=8001, description="MCP server port")
    mcp_server_config: str = Field(default="/app/config/mcp_servers.yaml", description="MCP config path")
    mcp_log_level: str = Field(default="INFO", description="MCP log level")
    
    # Individual MCP Servers
    knowledge_graph_mcp_port: int = Field(default=8002, description="Knowledge Graph MCP port")
    vector_search_mcp_port: int = Field(default=8003, description="Vector Search MCP port")
    document_processing_mcp_port: int = Field(default=8004, description="Document Processing MCP port")
    analytics_mcp_port: int = Field(default=8005, description="Analytics MCP port")


class PersonaSettings(BaseSettings):
    """AI Digital Twins persona configuration."""
    
    # Persona Configuration
    persona_config_path: str = Field(default="/app/config/personas", description="Persona config path")
    default_persona: str = Field(default="compliance_officer", description="Default persona")
    persona_learning_rate: float = Field(default=0.1, description="Persona learning rate")
    persona_adaptation_threshold: float = Field(default=0.8, description="Adaptation threshold")
    
    # Expert Twins
    expert_twins_enabled: bool = Field(default=True, description="Enable expert twins")
    expert_validation_threshold: float = Field(default=0.85, description="Expert validation threshold")
    expert_consensus_required: int = Field(default=2, description="Required expert consensus")
    
    # User Journey Twins
    user_journey_tracking: bool = Field(default=True, description="Enable user journey tracking")
    journey_analytics_enabled: bool = Field(default=True, description="Enable journey analytics")
    persona_interaction_logging: bool = Field(default=True, description="Log persona interactions")


class VectorSearchSettings(BaseSettings):
    """Vector search configuration settings."""
    
    # FAISS Configuration
    faiss_index_type: str = Field(default="IVFFlat", description="FAISS index type")
    faiss_nlist: int = Field(default=100, description="FAISS nlist parameter")
    faiss_nprobe: int = Field(default=10, description="FAISS nprobe parameter")
    faiss_metric_type: str = Field(default="INNER_PRODUCT", description="FAISS metric type")
    
    # Search Parameters
    vector_search_top_k: int = Field(default=20, description="Vector search top K results")
    vector_search_threshold: float = Field(default=0.7, description="Vector search threshold")
    hybrid_search_weight_vector: float = Field(default=0.6, description="Hybrid search vector weight")
    hybrid_search_weight_graph: float = Field(default=0.4, description="Hybrid search graph weight")


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_workers: int = Field(default=4, description="API worker processes")
    api_timeout: int = Field(default=300, description="API request timeout")
    api_max_request_size: str = Field(default="100MB", description="Max request size")
    
    # CORS Configuration
    cors_origins: List[str] = Field(default=["http://localhost:3000"], description="CORS origins")
    cors_credentials: bool = Field(default=True, description="CORS credentials")
    cors_methods: List[str] = Field(default=["GET", "POST", "PUT", "DELETE"], description="CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="CORS headers")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Prometheus
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=9090, description="Prometheus port")
    metrics_endpoint: str = Field(default="/metrics", description="Metrics endpoint")
    
    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format")
    log_rotation: str = Field(default="daily", description="Log rotation")
    log_retention_days: int = Field(default=90, description="Log retention days")
    audit_log_enabled: bool = Field(default=True, description="Enable audit logging")
    
    # Health Checks
    health_check_interval: int = Field(default=30, description="Health check interval")
    health_check_timeout: int = Field(default=10, description="Health check timeout")
    health_check_retries: int = Field(default=3, description="Health check retries")


class PerformanceSettings(BaseSettings):
    """Performance and resource configuration."""
    
    # System Resources
    max_memory_usage: str = Field(default="8GB", description="Maximum memory usage")
    max_cpu_usage: int = Field(default=80, description="Maximum CPU usage percentage")
    thread_pool_size: int = Field(default=10, description="Thread pool size")
    async_worker_count: int = Field(default=20, description="Async worker count")
    
    # Data Processing
    max_document_size: str = Field(default="50MB", description="Maximum document size")
    chunk_size: int = Field(default=512, description="Document chunk size")
    chunk_overlap: int = Field(default=50, description="Chunk overlap size")
    chunk_quality_threshold: float = Field(default=0.7, description="Chunk quality threshold")


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    offline_mode: bool = Field(default=True, description="Offline mode")
    
    # Sub-settings
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ai_models: AIModelSettings = Field(default_factory=AIModelSettings)
    mcp_servers: MCPServerSettings = Field(default_factory=MCPServerSettings)
    personas: PersonaSettings = Field(default_factory=PersonaSettings)
    vector_search: VectorSearchSettings = Field(default_factory=VectorSearchSettings)
    api: APISettings = Field(default_factory=APISettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f'Environment must be one of {valid_envs}')
        return v
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    def get_neo4j_uri(self) -> str:
        """Get the complete Neo4j connection URI."""
        return self.database.neo4j_uri
    
    def get_model_cache_path(self) -> Path:
        """Get the model cache directory path."""
        return Path(self.ai_models.model_cache_dir)
    
    def get_persona_config_path(self) -> Path:
        """Get the persona configuration directory path."""
        return Path(self.personas.persona_config_path)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    import json
    import yaml
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")


def validate_settings(settings: Settings) -> None:
    """
    Validate settings for security and compliance.
    
    Args:
        settings: Settings instance to validate
        
    Raises:
        ValueError: If settings are invalid
    """
    # Validate offline mode consistency
    if settings.offline_mode:
        if settings.ai_models.google_adk_api_key != "offline_placeholder":
            raise ValueError("Offline mode requires placeholder API key")
    
    # Validate security settings in production
    if settings.is_production():
        if settings.security.jwt_secret_key == "your_jwt_secret_key_here_change_in_production":
            raise ValueError("Production JWT secret key must be changed from default")
        
        if not settings.security.fedramp_compliance_mode:
            raise ValueError("FedRAMP compliance mode required in production")
    
    # Validate resource limits
    if settings.performance.max_cpu_usage > 90:
        raise ValueError("CPU usage limit cannot exceed 90%")
    
    # Validate model paths exist in offline mode
    if settings.offline_mode:
        model_cache_path = settings.get_model_cache_path()
        if not model_cache_path.exists():
            raise ValueError(f"Model cache directory does not exist: {model_cache_path}")


# Global settings instance
settings = get_settings()