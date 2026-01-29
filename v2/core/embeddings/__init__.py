"""Embedding system for JARVIS v2.

Provides:
- Message indexing and similarity search
- Style learning from your past messages
- Cached embeddings for performance
- Relationship-aware cross-conversation RAG
"""

from .cache import EmbeddingCache, get_embedding_cache, content_hash
from .model import EmbeddingModel, get_embedding_model
from .similarity import (
    cosine_similarity,
    find_most_similar,
    semantic_match,
    batch_cosine_similarity,
)
from .store import (
    EmbeddingStore,
    get_embedding_store,
    SimilarMessage,
    ConversationContext,
    StyleProfile,
)
from .indexer import MessageIndexer, run_indexing, IndexingStats
from .contact_profiler import ContactProfiler, ContactProfile, TopicCluster, get_contact_profile
from .relationship_registry import (
    RelationshipRegistry,
    RelationshipInfo,
    get_relationship_registry,
    reset_relationship_registry,
)

__all__ = [
    # Store (main interface)
    "EmbeddingStore",
    "get_embedding_store",
    "SimilarMessage",
    "ConversationContext",
    "StyleProfile",
    # Contact profiles
    "ContactProfiler",
    "ContactProfile",
    "TopicCluster",
    "get_contact_profile",
    # Relationship registry
    "RelationshipRegistry",
    "RelationshipInfo",
    "get_relationship_registry",
    "reset_relationship_registry",
    # Cache
    "EmbeddingCache",
    "get_embedding_cache",
    "content_hash",
    # Model
    "EmbeddingModel",
    "get_embedding_model",
    # Similarity
    "cosine_similarity",
    "find_most_similar",
    "semantic_match",
    "batch_cosine_similarity",
]
