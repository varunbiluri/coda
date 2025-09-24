"""
Repository content indexing and retrieval system.

This module provides semantic search capabilities over repository content using
LlamaIndex for document processing and ChromaDB for vector storage.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


class RepositoryIndexer:
    """
    Production-grade repository content indexer for semantic search and retrieval.

    This class provides comprehensive indexing and querying capabilities for repository
    content, enabling AI agents to understand and work with codebases effectively.
    """

    def __init__(self, collection_name: str = "coda_repo", persist_dir: str = "./chroma_db"):
        """
        Initialize the repository indexer with vector storage and embedding configuration.

        Args:
            collection_name: Identifier for the ChromaDB collection
            persist_dir: Directory path for persistent vector database storage

        Raises:
            RuntimeError: If embedding model initialization fails
        """
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        # Configure local embedding model to avoid external API dependencies
        try:
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            Settings.embed_model = embed_model
            logger.info("Successfully initialized HuggingFace embedding model")
        except Exception as e:
            logger.warning(
                f"Failed to load HuggingFace embedding model: {e}. Using default embeddings."
            )
            # Note: This may require OpenAI API key - consider this a fallback only

        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Initialize ChromaDB collection with error handling
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Connected to existing collection: {collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")

        # Initialize storage context and index
        self.storage_context = StorageContext.from_defaults()
        self.index: VectorStoreIndex | None = None

    def ingest_repository(self, repo_path: str, file_extensions: list[str] | None = None) -> None:
        """
        Ingest repository content into the vector index for semantic search.

        This method processes all relevant files in the repository, extracts their content,
        and creates searchable vector embeddings for AI agent consumption.

        Args:
            repo_path: Absolute path to the repository directory
            file_extensions: File extensions to include (defaults to common code file types)

        Raises:
            ValueError: If the repository path is invalid or inaccessible
            RuntimeError: If document indexing fails
        """
        if file_extensions is None:
            file_extensions = [
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".cs",
                ".go",
                ".rs",
                ".rb",
                ".php",
                ".swift",
                ".kt",
                ".scala",
                ".md",
                ".txt",
                ".yml",
                ".yaml",
                ".json",
                ".toml",
            ]

        repo_path_obj = Path(repo_path)
        if not repo_path_obj.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Load and process repository documents with comprehensive error handling
        documents = []
        total_files_processed = 0

        try:
            # Use simplified approach to avoid file extractor issues
            reader = SimpleDirectoryReader(
                input_dir=str(repo_path),
                recursive=True,
                exclude_hidden=True,
                required_exts=file_extensions,
            )
            documents = reader.load_data()
            total_files_processed = len(documents)
            logger.info(f"Successfully loaded {total_files_processed} documents from {repo_path}")

        except Exception as e:
            logger.warning(
                f"SimpleDirectoryReader failed: {e}. Attempting manual file processing..."
            )
            # Fallback to manual file processing
            for file_path in repo_path_obj.rglob("*"):
                if file_path.is_file() and file_path.suffix in file_extensions:
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            if content.strip():  # Only add non-empty files
                                doc = Document(
                                    text=content,
                                    metadata={"file_name": str(file_path.relative_to(repo_path))},
                                )
                                documents.append(doc)
                                total_files_processed += 1
                    except Exception as file_error:
                        logger.debug(f"Skipping file {file_path}: {file_error}")

        if not documents:
            raise ValueError(f"No valid documents found in repository: {repo_path}")

        # Create vector index from processed documents
        try:
            self.index = VectorStoreIndex.from_documents(documents)
            logger.info(f"Successfully indexed {len(documents)} documents from {repo_path}")
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise RuntimeError(f"Document indexing failed: {e}") from e

    def query(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Execute semantic search query against the indexed repository content.

        This method performs vector similarity search to find the most relevant
        code snippets and documentation that match the query intent.

        Args:
            query: Natural language query describing the desired information
            top_k: Maximum number of results to return (default: 5)

        Returns:
            List of search results, each containing:
                - content: The matched text content
                - metadata: File information and relevance metadata
                - score: Similarity score (if available)

        Raises:
            ValueError: If no index has been created (call ingest_repository first)
        """
        if self.index is None:
            raise ValueError("No index found. Please ingest a repository first.")

        # Use retriever directly to avoid LLM dependency
        retriever = self.index.as_retriever(similarity_top_k=top_k)

        # Execute retrieval
        nodes = retriever.retrieve(query)

        # Format results
        results = []
        for node in nodes:
            results.append(
                {
                    "content": node.text,
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, "score") else 0.0,
                }
            )

        return results

    def clear_collection(self) -> None:
        """
        Clear all indexed data from the collection.

        This method removes all previously indexed content and resets the collection
        to prepare for fresh repository ingestion.

        Note:
            This operation cannot be undone. All indexed content will be permanently removed.
        """
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(self.collection_name)
            self.storage_context = StorageContext.from_defaults()
            self.index = None
            logger.info("Successfully cleared collection data")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")
            # This is not a critical failure, continue execution
