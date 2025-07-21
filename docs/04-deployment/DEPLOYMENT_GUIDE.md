# Deployment Guide - Google ADK Integration

## Overview

This guide provides comprehensive deployment instructions for the Knowledge Graph-RAG system with Google ADK integration. The system is production-ready and supports multiple deployment scenarios.

## Prerequisites

### System Requirements

**Minimum Requirements**:
```yaml
Compute:
  CPU: 4 cores (2.5GHz+)
  RAM: 8GB
  Storage: 50GB SSD
  Network: 1Gbps

Software:
  OS: Ubuntu 20.04 LTS / RHEL 8+ / Windows Server 2019+
  Python: 3.11+
  Docker: 24.0+
  Docker Compose: 2.20+
```

**Recommended Production**:
```yaml
Compute:
  CPU: 8 cores (3.0GHz+)
  RAM: 16GB
  Storage: 100GB NVMe SSD
  Network: 10Gbps

Additional:
  Load Balancer: HAProxy / NGINX
  Monitoring: Prometheus + Grafana
  Logging: ELK Stack / Fluentd
```

### Google Cloud Requirements

1. **GCP Project Setup**:
   ```bash
   # Create or select project
   gcloud projects create your-project-id
   gcloud config set project your-project-id
   ```

2. **Required APIs**:
   ```bash
   # Enable necessary APIs
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable generativeai.googleapis.com
   gcloud services enable storage.googleapis.com
   ```

3. **Service Account**:
   ```bash
   # Create service account
   gcloud iam service-accounts create kg-rag-sa \
     --description="Knowledge Graph RAG Service Account" \
     --display-name="KG-RAG Service Account"
   
   # Grant necessary roles
   gcloud projects add-iam-policy-binding your-project-id \
     --member="serviceAccount:kg-rag-sa@your-project-id.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   
   gcloud projects add-iam-policy-binding your-project-id \
     --member="serviceAccount:kg-rag-sa@your-project-id.iam.gserviceaccount.com" \
     --role="roles/generativeai.user"
   
   # Create and download key
   gcloud iam service-accounts keys create credentials.json \
     --iam-account=kg-rag-sa@your-project-id.iam.gserviceaccount.com
   ```

### Neo4j Setup

1. **Neo4j Installation**:
   ```bash
   # Docker installation (recommended)
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -v neo4j-data:/data \
     -v neo4j-logs:/logs \
     -e NEO4J_AUTH=neo4j/your-password \
     -e NEO4J_PLUGINS=["apoc", "gds"] \
     neo4j:5.15
   ```

2. **Vector Index Setup**:
   ```cypher
   // Connect to Neo4j and run setup
   CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.node_id IS UNIQUE;
   CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.node_id IS UNIQUE;
   CREATE CONSTRAINT entity_canonical IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE;
   
   CALL db.index.vector.createNodeIndex(
     'document_embeddings',
     'Document', 
     'embedding',
     1536,
     'cosine'
   );
   ```

## Deployment Options

### Option 1: Docker Compose (Recommended for Development/Testing)

1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-org/knowledge-graph-rag.git
   cd knowledge-graph-rag
   ```

2. **Environment Configuration**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit configuration
   nano .env
   ```

   ```env
   # .env configuration
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_CLOUD_LOCATION=us-central1
   GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/credentials.json
   
   NEO4J_URI=neo4j://neo4j:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your-neo4j-password
   
   REDIS_URL=redis://redis:6379
   
   # ADK Configuration
   ADK_MODEL_NAME=gemini-1.5-pro
   ADK_TEMPERATURE=0.3
   ADK_MAX_OUTPUT_TOKENS=8192
   
   # Orchestration Configuration
   ORCHESTRATOR_ROUTING_STRATEGY=automatic
   ORCHESTRATOR_MAX_CONCURRENT_AGENTS=3
   ORCHESTRATOR_MIN_CONFIDENCE_THRESHOLD=0.7
   ```

3. **Docker Compose Setup**:
   ```yaml
   # docker-compose.yml
   version: '3.8'
   
   services:
     kg-rag:
       build: .
       ports:
         - "8000:8000"
       environment:
         - GOOGLE_CLOUD_PROJECT=${GOOGLE_CLOUD_PROJECT}
         - GOOGLE_CLOUD_LOCATION=${GOOGLE_CLOUD_LOCATION}
         - NEO4J_URI=${NEO4J_URI}
         - NEO4J_USERNAME=${NEO4J_USERNAME}
         - NEO4J_PASSWORD=${NEO4J_PASSWORD}
         - REDIS_URL=${REDIS_URL}
       volumes:
         - ./credentials.json:/app/credentials/credentials.json:ro
         - ./logs:/app/logs
       depends_on:
         - neo4j
         - redis
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   
     neo4j:
       image: neo4j:5.15
       ports:
         - "7474:7474"
         - "7687:7687"
       environment:
         - NEO4J_AUTH=${NEO4J_USERNAME}/${NEO4J_PASSWORD}
         - NEO4J_PLUGINS=["apoc", "gds"]
         - NEO4J_dbms_memory_heap_initial__size=2G
         - NEO4J_dbms_memory_heap_max__size=4G
       volumes:
         - neo4j-data:/data
         - neo4j-logs:/logs
   
     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"
       volumes:
         - redis-data:/data
   
     prometheus:
       image: prom/prometheus:latest
       ports:
         - "9090:9090"
       volumes:
         - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
         - prometheus-data:/prometheus
   
     grafana:
       image: grafana/grafana:latest
       ports:
         - "3000:3000"
       environment:
         - GF_SECURITY_ADMIN_PASSWORD=admin
       volumes:
         - grafana-data:/var/lib/grafana
         - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
   
   volumes:
     neo4j-data:
     neo4j-logs:
     redis-data:
     prometheus-data:
     grafana-data:
   ```

4. **Launch Services**:
   ```bash
   # Build and start services
   docker-compose up -d
   
   # Check service health
   docker-compose ps
   docker-compose logs kg-rag
   
   # Test endpoint
   curl http://localhost:8000/health
   ```

### Option 2: Kubernetes Deployment (Production)

1. **Namespace Setup**:
   ```yaml
   # namespace.yaml
   apiVersion: v1
   kind: Namespace
   metadata:
     name: kg-rag
     labels:
       name: kg-rag
   ```

2. **ConfigMap**:
   ```yaml
   # configmap.yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: kg-rag-config
     namespace: kg-rag
   data:
     GOOGLE_CLOUD_PROJECT: "your-project-id"
     GOOGLE_CLOUD_LOCATION: "us-central1"
     NEO4J_URI: "neo4j://neo4j-service:7687"
     NEO4J_USERNAME: "neo4j"
     REDIS_URL: "redis://redis-service:6379"
     ADK_MODEL_NAME: "gemini-1.5-pro"
     ADK_TEMPERATURE: "0.3"
     ADK_MAX_OUTPUT_TOKENS: "8192"
     ORCHESTRATOR_ROUTING_STRATEGY: "automatic"
     ORCHESTRATOR_MAX_CONCURRENT_AGENTS: "3"
   ```

3. **Secret**:
   ```yaml
   # secret.yaml
   apiVersion: v1
   kind: Secret
   metadata:
     name: kg-rag-secrets
     namespace: kg-rag
   type: Opaque
   data:
     NEO4J_PASSWORD: <base64-encoded-password>
     GOOGLE_APPLICATION_CREDENTIALS: <base64-encoded-credentials-json>
   ```

4. **Deployment**:
   ```yaml
   # deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: kg-rag
     namespace: kg-rag
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: kg-rag
     template:
       metadata:
         labels:
           app: kg-rag
       spec:
         containers:
         - name: kg-rag
           image: your-registry/kg-rag:latest
           ports:
           - containerPort: 8000
           env:
           - name: GOOGLE_CLOUD_PROJECT
             valueFrom:
               configMapKeyRef:
                 name: kg-rag-config
                 key: GOOGLE_CLOUD_PROJECT
           - name: NEO4J_PASSWORD
             valueFrom:
               secretKeyRef:
                 name: kg-rag-secrets
                 key: NEO4J_PASSWORD
           envFrom:
           - configMapRef:
               name: kg-rag-config
           volumeMounts:
           - name: credentials
             mountPath: /app/credentials
             readOnly: true
           resources:
             requests:
               cpu: 2000m
               memory: 4Gi
             limits:
               cpu: 4000m
               memory: 8Gi
           livenessProbe:
             httpGet:
               path: /health/live
               port: 8000
             initialDelaySeconds: 60
             periodSeconds: 30
           readinessProbe:
             httpGet:
               path: /health/ready
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
         volumes:
         - name: credentials
           secret:
             secretName: kg-rag-secrets
             items:
             - key: GOOGLE_APPLICATION_CREDENTIALS
               path: credentials.json
   ```

5. **Service**:
   ```yaml
   # service.yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: kg-rag-service
     namespace: kg-rag
   spec:
     selector:
       app: kg-rag
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: ClusterIP
   ```

6. **Horizontal Pod Autoscaler**:
   ```yaml
   # hpa.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: kg-rag-hpa
     namespace: kg-rag
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: kg-rag
     minReplicas: 3
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80
   ```

7. **Deploy to Kubernetes**:
   ```bash
   # Apply configurations
   kubectl apply -f namespace.yaml
   kubectl apply -f configmap.yaml
   kubectl apply -f secret.yaml
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f hpa.yaml
   
   # Check deployment
   kubectl get pods -n kg-rag
   kubectl get services -n kg-rag
   kubectl logs -f deployment/kg-rag -n kg-rag
   ```

### Option 3: Cloud-Native Deployment (GKE)

1. **GKE Cluster Creation**:
   ```bash
   # Create GKE cluster
   gcloud container clusters create kg-rag-cluster \
     --zone=us-central1-a \
     --num-nodes=3 \
     --machine-type=n1-standard-4 \
     --enable-autoscaling \
     --min-nodes=3 \
     --max-nodes=10 \
     --enable-autorepair \
     --enable-autoupgrade
   
   # Get credentials
   gcloud container clusters get-credentials kg-rag-cluster --zone=us-central1-a
   ```

2. **Workload Identity Setup**:
   ```bash
   # Enable Workload Identity
   gcloud container clusters update kg-rag-cluster \
     --workload-pool=your-project-id.svc.id.goog \
     --zone=us-central1-a
   
   # Create Kubernetes Service Account
   kubectl create serviceaccount kg-rag-ksa --namespace kg-rag
   
   # Bind to Google Service Account
   gcloud iam service-accounts add-iam-policy-binding \
     --role roles/iam.workloadIdentityUser \
     --member "serviceAccount:your-project-id.svc.id.goog[kg-rag/kg-rag-ksa]" \
     kg-rag-sa@your-project-id.iam.gserviceaccount.com
   
   # Annotate Kubernetes Service Account
   kubectl annotate serviceaccount kg-rag-ksa \
     --namespace kg-rag \
     iam.gke.io/gcp-service-account=kg-rag-sa@your-project-id.iam.gserviceaccount.com
   ```

3. **Cloud SQL for Neo4j Alternative** (Optional):
   ```bash
   # If using managed Neo4j alternative
   gcloud sql instances create kg-rag-db \
     --database-version=POSTGRES_14 \
     --tier=db-n1-standard-2 \
     --region=us-central1 \
     --enable-ip-alias
   ```

## Configuration Management

### Environment Variables Reference

```bash
# Google Cloud Configuration
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Neo4j Configuration
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"

# Redis Configuration (Optional)
export REDIS_URL="redis://localhost:6379"
export REDIS_PASSWORD="your-redis-password"

# ADK Agent Configuration
export ADK_MODEL_NAME="gemini-1.5-pro"
export ADK_TEMPERATURE="0.3"
export ADK_MAX_OUTPUT_TOKENS="8192"
export ADK_ENABLE_SAFETY_FILTERING="true"
export ADK_ENABLE_CACHING="true"
export ADK_CACHE_TTL_SECONDS="3600"

# RAG Agent Configuration
export RAG_RETRIEVAL_STRATEGY="hybrid"
export RAG_MAX_RETRIEVAL_DOCS="10"
export RAG_MIN_SIMILARITY_THRESHOLD="0.7"
export RAG_ENABLE_EXPERT_CONSULTATION="true"

# Orchestrator Configuration
export ORCHESTRATOR_ROUTING_STRATEGY="automatic"
export ORCHESTRATOR_ORCHESTRATION_MODE="single_agent"
export ORCHESTRATOR_MAX_CONCURRENT_AGENTS="3"
export ORCHESTRATOR_REQUEST_TIMEOUT_SECONDS="60"
export ORCHESTRATOR_MIN_CONFIDENCE_THRESHOLD="0.7"
export ORCHESTRATOR_ENABLE_VALIDATION="true"
export ORCHESTRATOR_ENABLE_FALLBACK="true"

# Performance Configuration
export ENABLE_PARALLEL_PROCESSING="true"
export MAX_WORKER_THREADS="10"
export CONNECTION_POOL_SIZE="20"

# Logging Configuration
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"
export ENABLE_STRUCTURED_LOGGING="true"
export LOG_CORRELATION_ID="true"

# Monitoring Configuration
export ENABLE_PROMETHEUS_METRICS="true"
export PROMETHEUS_PORT="9090"
export ENABLE_HEALTH_CHECKS="true"
export HEALTH_CHECK_TIMEOUT="30"
```

### Configuration Validation

```bash
# Validate configuration script
#!/bin/bash

echo "Validating Knowledge Graph-RAG Configuration..."

# Check required environment variables
required_vars=(
  "GOOGLE_CLOUD_PROJECT"
  "NEO4J_URI"
  "NEO4J_USERNAME"
  "NEO4J_PASSWORD"
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "❌ Missing required variable: $var"
    exit 1
  else
    echo "✅ $var is set"
  fi
done

# Test Google Cloud connectivity
echo "Testing Google Cloud connectivity..."
gcloud auth application-default print-access-token > /dev/null
if [ $? -eq 0 ]; then
  echo "✅ Google Cloud authentication successful"
else
  echo "❌ Google Cloud authentication failed"
  exit 1
fi

# Test Neo4j connectivity
echo "Testing Neo4j connectivity..."
python3 -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('$NEO4J_URI', auth=('$NEO4J_USERNAME', '$NEO4J_PASSWORD'))
with driver.session() as session:
    result = session.run('RETURN 1 as test')
    print('✅ Neo4j connectivity successful')
driver.close()
" 2>/dev/null || echo "❌ Neo4j connectivity failed"

echo "Configuration validation complete!"
```

## Monitoring and Observability

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kg-rag'
    static_configs:
      - targets: ['kg-rag:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'neo4j'
    static_configs:
      - targets: ['neo4j:7474']
    metrics_path: '/metrics'

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "title": "Knowledge Graph-RAG Dashboard",
    "panels": [
      {
        "title": "Query Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, kg_rag_query_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Agent Performance", 
        "type": "table",
        "targets": [
          {
            "expr": "rate(kg_rag_agent_requests_total[5m])",
            "legendFormat": "{{agent}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(kg_rag_errors_total[5m])",
            "legendFormat": "Errors/sec"
          }
        ]
      }
    ]
  }
}
```

### Health Check Endpoints

The application provides comprehensive health check endpoints:

```python
# Health check endpoints
GET /health          # Overall system health
GET /health/live     # Liveness probe (Kubernetes)
GET /health/ready    # Readiness probe (Kubernetes)
GET /health/startup  # Startup probe (Kubernetes)

# Detailed status endpoints
GET /status/agents   # Individual agent status
GET /status/database # Database connectivity
GET /status/gcp      # Google Cloud connectivity
GET /metrics         # Prometheus metrics
```

## Security Hardening

### SSL/TLS Configuration

```nginx
# nginx.conf for SSL termination
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/private/your-domain.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    location / {
        proxy_pass http://kg-rag-service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: kg-rag-netpol
  namespace: kg-rag
spec:
  podSelector:
    matchLabels:
      app: kg-rag
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS to Google APIs
    - protocol: TCP
      port: 7687 # Neo4j
    - protocol: TCP
      port: 6379 # Redis
```

### RBAC Configuration

```yaml
# rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: kg-rag
  name: kg-rag-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: kg-rag-rolebinding
  namespace: kg-rag
subjects:
- kind: ServiceAccount
  name: kg-rag-ksa
  namespace: kg-rag
roleRef:
  kind: Role
  name: kg-rag-role
  apiGroup: rbac.authorization.k8s.io
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale deployment
kubectl scale deployment kg-rag --replicas=5 -n kg-rag

# Auto-scaling based on custom metrics
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: kg-rag-hpa-custom
  namespace: kg-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: kg-rag
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: kg_rag_query_queue_length
      target:
        type: AverageValue
        averageValue: "5"
EOF
```

### Performance Tuning

```yaml
# performance-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kg-rag-performance-config
  namespace: kg-rag
data:
  # Connection pooling
  NEO4J_POOL_SIZE: "20"
  NEO4J_POOL_TIMEOUT: "30"
  
  # Async configuration
  MAX_CONCURRENT_QUERIES: "50"
  QUERY_TIMEOUT_SECONDS: "300"
  
  # Caching configuration
  REDIS_MAX_CONNECTIONS: "100"
  CACHE_DEFAULT_TTL: "3600"
  
  # Model optimization
  ADK_MAX_CONCURRENT_REQUESTS: "10"
  ADK_REQUEST_TIMEOUT: "30"
  
  # Resource limits
  MAX_MEMORY_USAGE_MB: "6144"
  MAX_CPU_USAGE_PERCENT: "80"
```

## Backup and Disaster Recovery

### Neo4j Backup Strategy

```bash
#!/bin/bash
# backup-neo4j.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/neo4j"
NEO4J_CONTAINER="neo4j"

# Create backup directory
mkdir -p $BACKUP_DIR

# Export database
docker exec $NEO4J_CONTAINER neo4j-admin database dump \
  --to-path=/var/backups neo4j > $BACKUP_DIR/neo4j_backup_$BACKUP_DATE.dump

# Copy backup file
docker cp $NEO4J_CONTAINER:/var/backups/neo4j.dump \
  $BACKUP_DIR/neo4j_backup_$BACKUP_DATE.dump

# Compress backup
gzip $BACKUP_DIR/neo4j_backup_$BACKUP_DATE.dump

# Upload to cloud storage (optional)
gsutil cp $BACKUP_DIR/neo4j_backup_$BACKUP_DATE.dump.gz \
  gs://your-backup-bucket/neo4j/

# Cleanup old backups (keep last 7 days)
find $BACKUP_DIR -name "neo4j_backup_*.dump.gz" -mtime +7 -delete

echo "Backup completed: neo4j_backup_$BACKUP_DATE.dump.gz"
```

### Application State Backup

```bash
#!/bin/bash
# backup-application-state.sh

# Backup Redis cache
redis-cli --rdb /backup/redis/dump_$(date +%Y%m%d_%H%M%S).rdb

# Backup configuration
kubectl get configmap kg-rag-config -n kg-rag -o yaml > \
  /backup/config/configmap_$(date +%Y%m%d_%H%M%S).yaml

kubectl get secret kg-rag-secrets -n kg-rag -o yaml > \
  /backup/config/secrets_$(date +%Y%m%d_%H%M%S).yaml

# Backup monitoring data
docker exec prometheus tar czf /backup/prometheus_$(date +%Y%m%d_%H%M%S).tar.gz \
  /prometheus/data/
```

## Troubleshooting

### Common Issues and Solutions

1. **Google Cloud Authentication Failed**:
   ```bash
   # Check credentials
   gcloud auth list
   gcloud auth application-default login
   
   # Verify service account permissions
   gcloud projects get-iam-policy your-project-id \
     --filter="bindings.members:serviceAccount:kg-rag-sa@your-project-id.iam.gserviceaccount.com"
   ```

2. **Neo4j Connection Issues**:
   ```bash
   # Test connectivity
   docker exec -it neo4j cypher-shell -u neo4j -p your-password
   
   # Check logs
   docker logs neo4j
   
   # Verify network connectivity
   telnet neo4j-host 7687
   ```

3. **High Memory Usage**:
   ```bash
   # Monitor resource usage
   kubectl top pods -n kg-rag
   
   # Check memory limits
   kubectl describe pod kg-rag-pod -n kg-rag
   
   # Adjust JVM settings for Neo4j
   docker exec neo4j neo4j-admin server memory-recommendation
   ```

4. **Slow Query Performance**:
   ```bash
   # Check query performance
   curl -X POST http://localhost:8000/debug/query-stats
   
   # Monitor database performance
   docker exec neo4j cypher-shell -c "CALL dbms.listQueries()"
   
   # Review indexes
   docker exec neo4j cypher-shell -c "SHOW INDEXES"
   ```

### Debugging Commands

```bash
# Application debugging
kubectl logs -f deployment/kg-rag -n kg-rag
kubectl exec -it deployment/kg-rag -n kg-rag -- /bin/bash

# Database debugging
docker exec -it neo4j cypher-shell -u neo4j -p password
docker exec -it neo4j tail -f /logs/debug.log

# Network debugging
kubectl exec -it deployment/kg-rag -n kg-rag -- nslookup neo4j-service
kubectl exec -it deployment/kg-rag -n kg-rag -- curl -v http://neo4j-service:7474

# Performance debugging
kubectl top nodes
kubectl top pods -n kg-rag --sort-by=cpu
kubectl top pods -n kg-rag --sort-by=memory
```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**:
   - Review application logs for errors
   - Check resource utilization
   - Verify backup integrity
   - Update security patches

2. **Monthly**:
   - Rotate service account keys
   - Review and update monitoring alerts
   - Performance optimization review
   - Capacity planning assessment

3. **Quarterly**:
   - Dependency updates and security audit
   - Disaster recovery testing
   - Performance benchmarking
   - Documentation updates

### Upgrade Procedures

```bash
# Rolling update deployment
kubectl set image deployment/kg-rag \
  kg-rag=your-registry/kg-rag:v2.0.0 \
  -n kg-rag

# Monitor rollout
kubectl rollout status deployment/kg-rag -n kg-rag

# Rollback if needed
kubectl rollout undo deployment/kg-rag -n kg-rag
```

This comprehensive deployment guide provides everything needed to successfully deploy and maintain the Knowledge Graph-RAG system with Google ADK integration in production environments.