# Knowledge Graph-RAG Deployment Guide

Complete deployment guide for the offline Knowledge Graph-RAG system with AI Digital Twins.

## 📋 Prerequisites

### System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS 10.15+, Windows 10+ with WSL2
- **CPU**: 8+ cores (16+ recommended)
- **RAM**: 16GB minimum (32GB+ recommended for production)
- **Storage**: 100GB+ available space
- **Network**: Offline deployment capability (air-gapped)

### Software Dependencies
- **Docker**: 20.10+ with Docker Compose
- **Python**: 3.11+ (for development)
- **Git**: For repository management

## 🚀 Quick Start Deployment

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/cgordon-dev/knowledge-graph-rag.git
cd knowledge-graph-rag

# Copy environment configuration
cp .env.example .env
```

### 2. Environment Configuration

Edit `.env` file with your settings:

```bash
# Core Configuration
ENVIRONMENT=production
SECRET_KEY=your-super-secure-secret-key-here
DEBUG=false

# Database Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-secure-neo4j-password

# AI Models Configuration
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DEVICE=cpu  # or 'cuda' for GPU
EMBEDDING_CACHE_SIZE=1000

# Security Settings
ENABLE_PII_FILTER=true
ENABLE_AUDIT_LOGGING=true
LOG_LEVEL=INFO

# Performance Settings
MAX_CONCURRENT_QUERIES=10
QUERY_TIMEOUT_SECONDS=30
CACHE_SIZE_MB=1024

# MCP Server Configuration
KNOWLEDGE_GRAPH_MCP_PORT=8001
VECTOR_SEARCH_MCP_PORT=8002
```

### 3. Docker Deployment

```bash
# Build and start all services
docker-compose up --build -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Verify Deployment

```bash
# Check Neo4j
curl http://localhost:7474

# Check application health (when API is implemented)
curl http://localhost:8000/health

# Run basic test
python scripts/test_deployment.py
```

## 🏗️ Production Deployment

### Air-Gapped Environment Setup

For secure, offline deployment in FedRAMP environments:

#### 1. Prepare Offline Package

```bash
# On internet-connected machine
./scripts/prepare_offline_package.sh

# This creates: kg-rag-offline-package.tar.gz
# Transfer this package to target environment
```

#### 2. Deploy in Air-Gapped Environment

```bash
# Extract package
tar -xzf kg-rag-offline-package.tar.gz
cd knowledge-graph-rag

# Load Docker images
docker load < docker-images.tar

# Deploy without external dependencies
docker-compose -f docker-compose.offline.yml up -d
```

### Production Configuration

#### Environment Variables for Production

```bash
# Production Environment
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-production-secret-key

# Security Hardening
ENABLE_PII_FILTER=true
ENABLE_AUDIT_LOGGING=true
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=100

# Performance Optimization
MAX_CONCURRENT_QUERIES=50
WORKER_PROCESSES=8
CACHE_SIZE_MB=4096
CONNECTION_POOL_SIZE=20

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_INTERVAL=30

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
```

#### Resource Allocation

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
        reservations:
          cpus: '4'
          memory: 8G
  
  neo4j:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

## 🔧 Service Configuration

### Neo4j Configuration

#### Memory Settings

```bash
# neo4j.conf
# Memory allocation (adjust based on available RAM)
dbms.memory.heap.initial_size=4G
dbms.memory.heap.max_size=8G
dbms.memory.pagecache.size=4G

# Security settings
dbms.security.auth_enabled=true
dbms.connector.bolt.listen_address=0.0.0.0:7687
dbms.connector.http.listen_address=0.0.0.0:7474

# Backup settings
dbms.backup.enabled=true
dbms.backup.address=0.0.0.0:6362
```

#### Vector Index Configuration

```cypher
// Create vector indexes for embeddings
CREATE VECTOR INDEX entity_embeddings 
FOR (n:Entity) ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX chunk_embeddings 
FOR (n:Chunk) ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};
```

### Application Configuration

#### Logging Configuration

```yaml
# logging.yml
version: 1
formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    formatter: detailed
    level: INFO
  
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/kg-rag/application.log
    maxBytes: 10485760  # 10MB
    backupCount: 5
    formatter: json
    level: INFO
  
  audit:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/kg-rag/audit.log
    maxBytes: 10485760
    backupCount: 10
    formatter: json
    level: INFO

loggers:
  kg_rag:
    level: INFO
    handlers: [console, file]
  
  audit:
    level: INFO
    handlers: [audit]
    propagate: false
```

## 🔐 Security Configuration

### SSL/TLS Configuration

```bash
# Generate certificates (for production)
./scripts/generate_certificates.sh

# Configure SSL in docker-compose
services:
  app:
    volumes:
      - ./certs:/app/certs:ro
    environment:
      - SSL_CERT_PATH=/app/certs/server.crt
      - SSL_KEY_PATH=/app/certs/server.key
```

### Access Control

```yaml
# rbac.yml - Role-Based Access Control
roles:
  admin:
    permissions:
      - twin:create
      - twin:delete
      - twin:configure
      - system:manage
  
  analyst:
    permissions:
      - twin:interact
      - twin:view
      - query:execute
  
  viewer:
    permissions:
      - twin:view
      - query:read_only
```

### Audit Configuration

```python
# audit.py
AUDIT_EVENTS = [
    'twin_interaction',
    'twin_creation',
    'twin_modification',
    'user_authentication',
    'data_access',
    'configuration_change'
]

AUDIT_FIELDS = [
    'timestamp',
    'user_id',
    'action',
    'resource',
    'result',
    'ip_address',
    'user_agent'
]
```

## 📊 Monitoring & Observability

### Health Checks

```python
# healthcheck.py
HEALTH_CHECKS = {
    'database': 'neo4j_connection',
    'mcp_servers': 'mcp_server_status',
    'ai_twins': 'twin_orchestrator_status',
    'memory': 'memory_usage',
    'disk': 'disk_usage'
}
```

### Metrics Collection

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kg-rag'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    metrics_path: /metrics
```

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: kg-rag-alerts
    rules:
      - alert: HighErrorRate
        expr: rate(kg_rag_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: High error rate detected
      
      - alert: HighMemoryUsage
        expr: kg_rag_memory_usage > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: Memory usage above 90%
```

## 🔄 Backup & Recovery

### Automated Backup

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/kg-rag"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Neo4j database
docker exec neo4j neo4j-admin dump --database=neo4j --to=/backups/neo4j_$DATE.dump

# Backup application data
docker run --rm -v kg-rag_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/app_data_$DATE.tar.gz -C /data .

# Backup configuration
cp -r ./config $BACKUP_DIR/config_$DATE

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.dump" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Recovery Procedures

```bash
#!/bin/bash
# restore.sh

BACKUP_FILE=$1
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh <backup_file>"
    exit 1
fi

# Stop services
docker-compose down

# Restore Neo4j database
docker run --rm -v neo4j_data:/data -v $PWD:/backup neo4j:5.15 \
    neo4j-admin load --from=/backup/$BACKUP_FILE --database=neo4j --force

# Restore application data
docker run --rm -v kg-rag_data:/data -v $PWD:/backup alpine \
    tar xzf /backup/app_data_backup.tar.gz -C /data

# Restart services
docker-compose up -d
```

## 🔧 Troubleshooting

### Common Issues

#### Memory Issues

```bash
# Check memory usage
docker stats

# Increase memory limits
# Edit docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          memory: 32G
```

#### Database Connection Issues

```bash
# Check Neo4j logs
docker-compose logs neo4j

# Test connection
docker exec -it neo4j cypher-shell -u neo4j -p password

# Restart database
docker-compose restart neo4j
```

#### Performance Issues

```bash
# Check performance metrics
curl http://localhost:9090/metrics

# Analyze slow queries
# Check logs for query performance
grep "slow_query" /var/log/kg-rag/application.log
```

### Diagnostic Commands

```bash
# System diagnostics
./scripts/diagnose.sh

# Performance test
./scripts/performance_test.sh

# Configuration validation
./scripts/validate_config.sh
```

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  app:
    deploy:
      replicas: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app
```

### Load Balancing Configuration

```nginx
# nginx.conf
upstream kg_rag_backend {
    server app_1:8000;
    server app_2:8000;
    server app_3:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://kg_rag_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔒 Security Hardening

### Container Security

```dockerfile
# Use non-root user
RUN addgroup -g 1001 appgroup && \
    adduser -u 1001 -G appgroup -s /bin/sh -D appuser

USER appuser

# Read-only filesystem
--read-only --tmpfs /tmp --tmpfs /run
```

### Network Security

```yaml
# docker-compose.security.yml
networks:
  kg-rag-internal:
    driver: bridge
    internal: true
  kg-rag-external:
    driver: bridge

services:
  app:
    networks:
      - kg-rag-internal
      - kg-rag-external
  
  neo4j:
    networks:
      - kg-rag-internal
```

### Secret Management

```bash
# Use Docker secrets
echo "neo4j_password" | docker secret create neo4j_password -

# In docker-compose.yml
services:
  neo4j:
    secrets:
      - neo4j_password
    environment:
      - NEO4J_AUTH=neo4j/run/secrets/neo4j_password

secrets:
  neo4j_password:
    external: true
```

## 📞 Support & Maintenance

### Log Locations

```bash
# Application logs
/var/log/kg-rag/application.log
/var/log/kg-rag/audit.log
/var/log/kg-rag/error.log

# Container logs
docker-compose logs app
docker-compose logs neo4j
```

### Maintenance Schedule

- **Daily**: Log rotation, health checks
- **Weekly**: Performance analysis, security scans
- **Monthly**: Backup verification, dependency updates
- **Quarterly**: Security audit, capacity planning

### Contact Information

- **Technical Issues**: Create GitHub issue
- **Security Issues**: security@your-domain.com
- **Emergency Support**: emergency@your-domain.com

---

This deployment guide provides comprehensive instructions for deploying the Knowledge Graph-RAG system in various environments, from development to production air-gapped deployments.