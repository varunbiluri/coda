"""
Repository content indexing and retrieval system.

This module provides semantic search capabilities over repository content using
LlamaIndex for document processing and ChromaDB for vector storage with Azure OpenAI embeddings.
"""

import logging
import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import Document
from openai import AzureOpenAI
from pydantic import Field

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class AzureOpenAIEmbedding(BaseEmbedding):
    """Custom Azure OpenAI embedding class for LlamaIndex integration."""

    model_name: str = Field(description="The model name")
    deployment_name: str = Field(description="The deployment name")
    client: AzureOpenAI = Field(description="The Azure OpenAI client")

    def __init__(
        self,
        model: str,
        deployment_name: str,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
    ):
        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
        super().__init__(
            model_name=model, deployment_name=deployment_name, client=client
        )

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        response = self.client.embeddings.create(model=self.deployment_name, input=text)
        return list(response.data[0].embedding)

    def _get_text_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        response = self.client.embeddings.create(
            model=self.deployment_name, input=texts
        )
        return [data.embedding for data in response.data]

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query (same as text embedding)."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Get embedding for a query asynchronously."""
        return self._get_query_embedding(query)


class RepositoryIndexer:
    """
    Production-grade repository content indexer for semantic search and retrieval.

    This class provides comprehensive indexing and querying capabilities for repository
    content, enabling AI agents to understand and work with codebases effectively.
    """

    def __init__(
        self,
        collection_name: str = "coda_repo",
        persist_dir: str = "./chroma_db",
        embedding_provider: str = "azure",
    ):
        """
        Initialize the repository indexer with vector storage and embedding configuration.

        Args:
            collection_name: Identifier for the ChromaDB collection
            persist_dir: Directory path for persistent vector database storage
            embedding_provider: Embedding provider to use ("azure", "openai", "local")

        Raises:
            RuntimeError: If embedding model initialization fails
            ValueError: If invalid embedding provider is specified
        """
        self.collection_name = collection_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)
        self.embedding_provider = embedding_provider.lower()

        # Initialize embedding model (single provider approach)
        self.embed_model: BaseEmbedding | None = None
        self._setup_embedding_model()

        # Initialize ChromaDB
        self._setup_chromadb()

    def _setup_embedding_model(self) -> None:
        """Setup embedding model based on specified provider."""
        if self.embedding_provider == "azure":
            self._setup_azure_embeddings()
        elif self.embedding_provider == "openai":
            self._setup_openai_embeddings()
        elif self.embedding_provider == "local":
            self._setup_local_embeddings()
        else:
            raise ValueError(
                f"Invalid embedding provider: {self.embedding_provider}. Must be 'azure', 'openai', or 'local'"
            )

        # Set global LlamaIndex settings to use our embedding model
        Settings.embed_model = self.embed_model
        logger.info(
            f"Set global LlamaIndex embedding model to: {self.embedding_provider}"
        )

    def _setup_azure_embeddings(self) -> None:
        """Setup Azure OpenAI embeddings."""
        try:
            # Get Azure OpenAI configuration from environment
            api_key = os.getenv("AZURE_API_KEY")
            azure_endpoint = os.getenv("AZURE_API_BASE")
            api_version = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")

            if not api_key or not azure_endpoint:
                raise ValueError(
                    "Azure API credentials not found. Set AZURE_API_KEY and AZURE_API_BASE environment variables."
                )

            # For Azure AI Foundry, extract the base URL
            if "services.ai.azure.com" in azure_endpoint:
                base_url = azure_endpoint.split("/api/projects/")[0]
                logger.info(f"Using Azure AI Foundry endpoint: {base_url}")
            else:
                base_url = azure_endpoint

            # Initialize Azure OpenAI embeddings
            self.embed_model = AzureOpenAIEmbedding(
                model="text-embedding-3-small",
                deployment_name="text-embedding-3-small",
                api_key=api_key,
                azure_endpoint=base_url,
                api_version=api_version,
            )
            logger.info(
                "Successfully initialized Azure OpenAI embedding model: text-embedding-3-small"
            )

        except Exception as e:
            logger.error(f"Failed to initialize Azure embeddings: {e}")
            raise RuntimeError(f"Azure embedding initialization failed: {e}") from e

    def _setup_openai_embeddings(self) -> None:
        """Setup OpenAI embeddings."""
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
                )

            self.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small", api_key=api_key
            )
            logger.info(
                "Successfully initialized OpenAI embedding model: text-embedding-3-small"
            )

        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise RuntimeError(f"OpenAI embedding initialization failed: {e}") from e

    def _setup_local_embeddings(self) -> None:
        """Setup local HuggingFace embeddings."""
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            self.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info(
                "Successfully initialized HuggingFace embedding model: all-MiniLM-L6-v2"
            )

        except Exception as e:
            logger.error(f"Failed to initialize local embeddings: {e}")
            raise RuntimeError(f"Local embedding initialization failed: {e}") from e

    def _setup_chromadb(self) -> None:
        """Setup ChromaDB client and collection."""
        try:
            # Initialize Chroma client
            self.chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))

            # Initialize ChromaDB collection with error handling
            try:
                self.collection = self.chroma_client.get_collection(
                    self.collection_name
                )
                logger.info(f"Connected to existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    self.collection_name
                )
                logger.info(f"Created new collection: {self.collection_name}")

            # Initialize storage context and index
            self.storage_context = StorageContext.from_defaults()
            self.index: VectorStoreIndex | None = None

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB initialization failed: {e}") from e

    def get_embedding(self, text: str) -> list[float]:
        """
        Get embedding for text using the configured provider.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            RuntimeError: If embedding model is not available
        """
        if not self.embed_model:
            raise RuntimeError(
                "No embedding model available. Check provider configuration."
            )

        try:
            result = self.embed_model._get_text_embedding(text)
            if hasattr(result, "__iter__") and not isinstance(result, str):
                return list(result)
            else:
                return [float(result)]
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e

    def get_embedding_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts using the configured provider.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If embedding model is not available
        """
        if not self.embed_model:
            raise RuntimeError(
                "No embedding model available. Check provider configuration."
            )

        try:
            if hasattr(self.embed_model, "_get_text_embedding_batch"):
                result = self.embed_model._get_text_embedding_batch(texts)
                return (
                    list(result) if hasattr(result, "__iter__") else [[float(result)]]
                )
            else:
                # Fallback to individual embeddings
                return [self.get_embedding(text) for text in texts]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise RuntimeError(f"Batch embedding generation failed: {e}") from e

    def ingest_repository(
        self, repo_path: str, file_extensions: list[str] | None = None
    ) -> None:
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
            logger.info(
                f"Successfully loaded {total_files_processed} documents from {repo_path}"
            )

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
                                    metadata={
                                        "file_name": str(
                                            file_path.relative_to(repo_path)
                                        )
                                    },
                                )
                                documents.append(doc)
                                total_files_processed += 1
                    except Exception as file_error:
                        logger.debug(f"Skipping file {file_path}: {file_error}")

        if not documents:
            raise ValueError(f"No valid documents found in repository: {repo_path}")

        # Create vector index from processed documents
        try:
            # Explicitly pass the embedding model to avoid default OpenAI usage
            self.index = VectorStoreIndex.from_documents(
                documents, embed_model=self.embed_model
            )
            logger.info(
                f"Successfully indexed {len(documents)} documents from {repo_path}"
            )
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
