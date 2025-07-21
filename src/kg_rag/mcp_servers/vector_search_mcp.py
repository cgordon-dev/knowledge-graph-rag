"""
Vector Search MCP Server for semantic similarity operations.

Provides tools for vector similarity search, clustering, and embedding
operations with FAISS and sentence transformers integration.
"""

import asyncio
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import structlog

from kg_rag.core.config import get_settings
from kg_rag.core.exceptions import VectorSearchError, EmbeddingGenerationError, ModelLoadError
from kg_rag.mcp_servers.base_mcp import BaseMCPServer


class VectorSearchMCP(BaseMCPServer):
    """MCP server for Vector Search and Embedding operations."""
    
    def __init__(self):
        """Initialize Vector Search MCP server."""
        settings = get_settings()
        super().__init__(
            name="vector_search_mcp",
            description="Vector similarity search and embedding operations",
            port=settings.mcp_servers.vector_search_mcp_port
        )
        
        # Vector search components
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vector_indexes: Dict[str, faiss.Index] = {}
        self.index_metadata: Dict[str, Dict[str, Any]] = {}
        self.vector_dimension: int = 1024  # BGE-Large-EN-v1.5 dimension
        
        # Cache for embeddings
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_size_limit = self.settings.ai_models.embedding_cache_size
    
    async def _initialize(self) -> None:
        """Initialize embedding model and vector indexes."""
        try:
            # Load embedding model
            await self._load_embedding_model()
            
            # Initialize FAISS indexes
            await self._initialize_vector_indexes()
            
            # Load existing indexes if available
            await self._load_existing_indexes()
            
            self.logger.info(
                "Vector search MCP initialized",
                model=self.settings.ai_models.embedding_model,
                vector_dimension=self.vector_dimension,
                indexes_loaded=len(self.vector_indexes)
            )
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize vector search MCP: {e}",
                model_name=self.settings.ai_models.embedding_model
            )
    
    async def _cleanup(self) -> None:
        """Cleanup vector search resources."""
        # Save indexes before cleanup
        await self._save_indexes()
        
        # Clear caches
        self.embedding_cache.clear()
        self.vector_indexes.clear()
        self.index_metadata.clear()
        
        self.embedding_model = None
    
    async def _load_embedding_model(self) -> None:
        """Load the sentence transformer embedding model."""
        try:
            model_name = self.settings.ai_models.embedding_model
            device = self.settings.ai_models.embedding_device
            cache_dir = self.settings.get_model_cache_path()
            
            # Load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    model_name,
                    device=device,
                    cache_folder=str(cache_dir)
                )
            )
            
            # Update vector dimension from model
            if hasattr(self.embedding_model, 'get_sentence_embedding_dimension'):
                self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
            
            self.logger.info(
                "Embedding model loaded",
                model=model_name,
                device=device,
                dimension=self.vector_dimension
            )
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load embedding model: {e}",
                model_name=self.settings.ai_models.embedding_model,
                model_path=str(self.settings.get_model_cache_path())
            )
    
    async def _initialize_vector_indexes(self) -> None:
        """Initialize FAISS vector indexes."""
        try:
            # Create different index types based on configuration
            faiss_config = self.settings.vector_search
            
            if faiss_config.faiss_index_type == "IVFFlat":
                # IVF (Inverted File) index for large datasets
                quantizer = faiss.IndexFlatIP(self.vector_dimension)  # Inner product for cosine similarity
                index = faiss.IndexIVFFlat(quantizer, self.vector_dimension, faiss_config.faiss_nlist)
                index.nprobe = faiss_config.faiss_nprobe
            elif faiss_config.faiss_index_type == "HNSW":
                # HNSW index for fast approximate search
                index = faiss.IndexHNSWFlat(self.vector_dimension, 32)
                index.hnsw.efSearch = 64
            else:
                # Default flat index
                index = faiss.IndexFlatIP(self.vector_dimension)
            
            # Initialize default indexes
            default_indexes = [
                "control_content_index",
                "chunk_content_index",
                "entity_capability_index",
                "persona_expertise_index"
            ]
            
            for index_name in default_indexes:
                self.vector_indexes[index_name] = index.clone() if hasattr(index, 'clone') else faiss.IndexFlatIP(self.vector_dimension)
                self.index_metadata[index_name] = {
                    "dimension": self.vector_dimension,
                    "metric": "inner_product",
                    "size": 0,
                    "created_at": self.start_time.isoformat() if self.start_time else None
                }
            
        except Exception as e:
            raise VectorSearchError(f"Failed to initialize FAISS indexes: {e}")
    
    async def _load_existing_indexes(self) -> None:
        """Load existing vector indexes from disk."""
        try:
            index_dir = Path(self.settings.performance.max_document_size).parent / "indexes"
            if not index_dir.exists():
                return
            
            for index_file in index_dir.glob("*.faiss"):
                index_name = index_file.stem
                try:
                    # Load FAISS index
                    index = faiss.read_index(str(index_file))
                    self.vector_indexes[index_name] = index
                    
                    # Load metadata
                    metadata_file = index_dir / f"{index_name}.metadata"
                    if metadata_file.exists():
                        with open(metadata_file, 'rb') as f:
                            self.index_metadata[index_name] = pickle.load(f)
                    
                    self.logger.info(f"Loaded vector index: {index_name}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load index {index_name}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error loading existing indexes: {e}")
    
    async def _save_indexes(self) -> None:
        """Save vector indexes to disk."""
        try:
            index_dir = Path(self.settings.performance.max_document_size).parent / "indexes"
            index_dir.mkdir(exist_ok=True)
            
            for index_name, index in self.vector_indexes.items():
                # Save FAISS index
                index_file = index_dir / f"{index_name}.faiss"
                faiss.write_index(index, str(index_file))
                
                # Save metadata
                metadata_file = index_dir / f"{index_name}.metadata"
                with open(metadata_file, 'wb') as f:
                    pickle.dump(self.index_metadata[index_name], f)
                    
        except Exception as e:
            self.logger.error(f"Error saving indexes: {e}")
    
    def register_tools(self) -> None:
        """Register Vector Search MCP tools."""
        
        @self.tool(
            "generate_embedding",
            "Generate vector embedding for text",
            {
                "text": {"type": "string", "required": True, "description": "Text to embed"},
                "normalize": {"type": "boolean", "required": False, "description": "Normalize embedding vector"}
            }
        )
        async def generate_embedding(
            text: str,
            normalize: bool = True
        ) -> List[float]:
            """Generate vector embedding for text."""
            if not self.embedding_model:
                raise EmbeddingGenerationError("Embedding model not loaded")
            
            # Check cache first
            cache_key = f"{text}_{normalize}"
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key].tolist()
            
            try:
                # Generate embedding in thread
                loop = asyncio.get_event_loop()
                embedding = await loop.run_in_executor(
                    None,
                    lambda: self.embedding_model.encode(
                        text,
                        normalize_embeddings=normalize,
                        show_progress_bar=False
                    )
                )
                
                # Cache embedding
                if len(self.embedding_cache) < self.cache_size_limit:
                    self.embedding_cache[cache_key] = embedding
                
                return embedding.tolist()
                
            except Exception as e:
                raise EmbeddingGenerationError(
                    f"Failed to generate embedding: {e}",
                    text_length=len(text),
                    model_name=self.settings.ai_models.embedding_model
                )
        
        @self.tool(
            "similarity_search",
            "Perform vector similarity search",
            {
                "text": {"type": "string", "required": False, "description": "Text query"},
                "vector": {"type": "array", "required": False, "description": "Query vector"},
                "index_names": {"type": "array", "required": False, "description": "Index names to search"},
                "threshold": {"type": "number", "required": False, "description": "Similarity threshold"},
                "limit": {"type": "integer", "required": False, "description": "Maximum results"}
            }
        )
        async def similarity_search(
            text: Optional[str] = None,
            vector: Optional[List[float]] = None,
            index_names: Optional[List[str]] = None,
            threshold: float = 0.7,
            limit: int = 20
        ) -> List[Dict[str, Any]]:
            """Perform vector similarity search across multiple indexes."""
            
            # Generate embedding if text provided
            if text and not vector:
                vector = await generate_embedding(text)
            
            if not vector:
                raise VectorSearchError("Either text or vector must be provided")
            
            # Default to all indexes if none specified
            if not index_names:
                index_names = list(self.vector_indexes.keys())
            
            query_vector = np.array(vector, dtype=np.float32).reshape(1, -1)
            all_results = []
            
            try:
                for index_name in index_names:
                    if index_name not in self.vector_indexes:
                        self.logger.warning(f"Index not found: {index_name}")
                        continue
                    
                    index = self.vector_indexes[index_name]
                    if index.ntotal == 0:
                        continue  # Skip empty indexes
                    
                    # Search in index
                    scores, indices = index.search(query_vector, min(limit, index.ntotal))
                    
                    # Filter by threshold and format results
                    for score, idx in zip(scores[0], indices[0]):
                        if score >= threshold:
                            all_results.append({
                                "index_name": index_name,
                                "document_id": int(idx),
                                "similarity_score": float(score),
                                "metadata": self.index_metadata.get(index_name, {})
                            })
                
                # Sort by similarity score and limit results
                all_results.sort(key=lambda x: x["similarity_score"], reverse=True)
                return all_results[:limit]
                
            except Exception as e:
                raise VectorSearchError(f"Similarity search failed: {e}")
        
        @self.tool(
            "add_vectors_to_index",
            "Add vectors to a specific index",
            {
                "index_name": {"type": "string", "required": True, "description": "Index name"},
                "vectors": {"type": "array", "required": True, "description": "Array of vectors"},
                "document_ids": {"type": "array", "required": False, "description": "Document IDs for vectors"}
            }
        )
        async def add_vectors_to_index(
            index_name: str,
            vectors: List[List[float]],
            document_ids: Optional[List[int]] = None
        ) -> Dict[str, Any]:
            """Add vectors to a specific index."""
            try:
                if index_name not in self.vector_indexes:
                    # Create new index if it doesn't exist
                    self.vector_indexes[index_name] = faiss.IndexFlatIP(self.vector_dimension)
                    self.index_metadata[index_name] = {
                        "dimension": self.vector_dimension,
                        "metric": "inner_product", 
                        "size": 0,
                        "created_at": self.start_time.isoformat() if self.start_time else None
                    }
                
                index = self.vector_indexes[index_name]
                
                # Convert vectors to numpy array
                vector_array = np.array(vectors, dtype=np.float32)
                
                # Validate vector dimensions
                if vector_array.shape[1] != self.vector_dimension:
                    raise VectorSearchError(
                        f"Vector dimension mismatch: expected {self.vector_dimension}, got {vector_array.shape[1]}"
                    )
                
                # Add vectors to index
                if document_ids:
                    # If we have IDs, we need to use IndexIDMap
                    if not hasattr(index, 'id_map'):
                        id_map_index = faiss.IndexIDMap(index)
                        self.vector_indexes[index_name] = id_map_index
                        index = id_map_index
                    
                    index.add_with_ids(vector_array, np.array(document_ids, dtype=np.int64))
                else:
                    index.add(vector_array)
                
                # Update metadata
                self.index_metadata[index_name]["size"] = index.ntotal
                self.index_metadata[index_name]["last_updated"] = self.start_time.isoformat() if self.start_time else None
                
                return {
                    "index_name": index_name,
                    "vectors_added": len(vectors),
                    "total_vectors": index.ntotal,
                    "success": True
                }
                
            except Exception as e:
                raise VectorSearchError(f"Failed to add vectors to index: {e}")
        
        @self.tool(
            "create_index",
            "Create a new vector index",
            {
                "index_name": {"type": "string", "required": True, "description": "Index name"},
                "index_type": {"type": "string", "required": False, "description": "Index type (flat, ivf, hnsw)"},
                "metric": {"type": "string", "required": False, "description": "Distance metric"}
            }
        )
        async def create_index(
            index_name: str,
            index_type: str = "flat",
            metric: str = "inner_product"
        ) -> Dict[str, Any]:
            """Create a new vector index."""
            try:
                if index_name in self.vector_indexes:
                    raise VectorSearchError(f"Index '{index_name}' already exists")
                
                # Create index based on type
                if index_type.lower() == "ivf":
                    quantizer = faiss.IndexFlatIP(self.vector_dimension)
                    index = faiss.IndexIVFFlat(quantizer, self.vector_dimension, 100)
                    index.nprobe = 10
                elif index_type.lower() == "hnsw":
                    index = faiss.IndexHNSWFlat(self.vector_dimension, 32)
                    index.hnsw.efSearch = 64
                else:  # flat
                    if metric == "inner_product":
                        index = faiss.IndexFlatIP(self.vector_dimension)
                    else:
                        index = faiss.IndexFlatL2(self.vector_dimension)
                
                self.vector_indexes[index_name] = index
                self.index_metadata[index_name] = {
                    "dimension": self.vector_dimension,
                    "metric": metric,
                    "index_type": index_type,
                    "size": 0,
                    "created_at": self.start_time.isoformat() if self.start_time else None
                }
                
                return {
                    "index_name": index_name,
                    "index_type": index_type,
                    "metric": metric,
                    "dimension": self.vector_dimension,
                    "created": True
                }
                
            except Exception as e:
                raise VectorSearchError(f"Failed to create index: {e}")
        
        @self.tool(
            "get_index_info",
            "Get information about vector indexes",
            {
                "index_name": {"type": "string", "required": False, "description": "Specific index name"}
            }
        )
        async def get_index_info(index_name: Optional[str] = None) -> Dict[str, Any]:
            """Get information about vector indexes."""
            if index_name:
                if index_name not in self.vector_indexes:
                    raise VectorSearchError(f"Index '{index_name}' not found")
                
                index = self.vector_indexes[index_name]
                metadata = self.index_metadata.get(index_name, {})
                
                return {
                    "index_name": index_name,
                    "size": index.ntotal,
                    "dimension": metadata.get("dimension", self.vector_dimension),
                    "metric": metadata.get("metric", "unknown"),
                    "metadata": metadata
                }
            else:
                # Return info for all indexes
                all_indexes = {}
                for name, index in self.vector_indexes.items():
                    metadata = self.index_metadata.get(name, {})
                    all_indexes[name] = {
                        "size": index.ntotal,
                        "dimension": metadata.get("dimension", self.vector_dimension),
                        "metric": metadata.get("metric", "unknown"),
                        "metadata": metadata
                    }
                
                return {
                    "total_indexes": len(self.vector_indexes),
                    "indexes": all_indexes,
                    "cache_size": len(self.embedding_cache)
                }
        
        @self.tool(
            "semantic_clustering", 
            "Perform semantic clustering on vectors",
            {
                "index_name": {"type": "string", "required": True, "description": "Index to cluster"},
                "num_clusters": {"type": "integer", "required": False, "description": "Number of clusters"},
                "sample_size": {"type": "integer", "required": False, "description": "Sample size for clustering"}
            }
        )
        async def semantic_clustering(
            index_name: str,
            num_clusters: int = 5,
            sample_size: Optional[int] = None
        ) -> Dict[str, Any]:
            """Perform semantic clustering on vectors in an index."""
            try:
                if index_name not in self.vector_indexes:
                    raise VectorSearchError(f"Index '{index_name}' not found")
                
                index = self.vector_indexes[index_name]
                if index.ntotal == 0:
                    return {"clusters": [], "message": "Empty index"}
                
                # Get vectors from index
                vectors = index.reconstruct_n(0, min(sample_size or index.ntotal, index.ntotal))
                
                if len(vectors) < num_clusters:
                    num_clusters = len(vectors)
                
                # Perform k-means clustering
                kmeans = faiss.Kmeans(self.vector_dimension, num_clusters, niter=20, verbose=False)
                kmeans.train(vectors)
                
                # Assign vectors to clusters
                _, cluster_assignments = kmeans.index.search(vectors, 1)
                
                # Group results by cluster
                clusters = {}
                for i, cluster_id in enumerate(cluster_assignments.flatten()):
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append({
                        "vector_id": i,
                        "distance_to_centroid": float(kmeans.index.search(vectors[i:i+1], 1)[0][0])
                    })
                
                return {
                    "index_name": index_name,
                    "num_clusters": num_clusters,
                    "total_vectors": len(vectors),
                    "clusters": [{"cluster_id": k, "members": v} for k, v in clusters.items()],
                    "centroids": kmeans.centroids.tolist()
                }
                
            except Exception as e:
                raise VectorSearchError(f"Clustering failed: {e}")
    
    async def _custom_health_check(self) -> Dict[str, Any]:
        """Custom health check for vector search components."""
        try:
            health_info = {
                "embedding_model_loaded": self.embedding_model is not None,
                "vector_dimension": self.vector_dimension,
                "total_indexes": len(self.vector_indexes),
                "cache_size": len(self.embedding_cache),
                "cache_limit": self.cache_size_limit
            }
            
            # Check index sizes
            index_info = {}
            for name, index in self.vector_indexes.items():
                index_info[name] = {
                    "size": index.ntotal,
                    "trained": getattr(index, 'is_trained', True)
                }
            
            health_info["indexes"] = index_info
            
            return health_info
            
        except Exception as e:
            return {"error": str(e)}