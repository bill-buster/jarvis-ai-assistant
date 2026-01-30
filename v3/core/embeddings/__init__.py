"""Embedding system for JARVIS v2.

Provides:
- Message indexing and similarity search
- Style learning from your past messages
- Cached embeddings for performance
- Relationship-aware cross-conversation RAG
"""

from .contact_profiler import ContactProfile, ContactProfiler, TopicCluster, get_contact_profile
from .indexer import IndexingStats, MessageIndexer, run_indexing
from .model import EmbeddingModel, get_embedding_model
from .relationship_registry import (
    RelationshipInfo,
    RelationshipRegistry,
    get_relationship_registry,
    reset_relationship_registry,
)
from .store import (
    ConversationContext,
    EmbeddingStore,
    SimilarMessage,
    StyleProfile,
    get_embedding_store,
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
    # Model
    "EmbeddingModel",
    "get_embedding_model",
    # Indexer
    "MessageIndexer",
    "run_indexing",
    "IndexingStats",
]
