"""
Repository Gist Agent for Git Repo Summary Use Case.

This agent focuses on the core git repo summary use case with:
- Intelligent Git checkout strategies
- Basic semantic chunking (with fallback)
- Enhanced LLM summarization
- Multiple export formats
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import yaml
from git import Repo

from config.settings import (
    CHROMA_DB_PATH,
    LITELLM_MODEL,
    LITELLM_PROVIDER,
    USE_MOCK_LLM,
)

from ..core.git_strategy import GitStrategy
from ..core.indexer import RepositoryIndexer
from ..core.llm_client import LLMClient, create_llm_client

logger = logging.getLogger(__name__)


class RepoGistAgent:
    """
    Agent for git repo summary use case.

    Focuses on core functionality: intelligent Git strategies,
    basic chunking, and enhanced LLM summarization.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        use_embeddings: bool = True,
        embedding_provider: str = "azure",
    ):
        """
        Initialize the RepoGistAgent.

        Args:
            llm_client: LLM client for generating summaries
            use_embeddings: Whether to use vector embeddings
            embedding_provider: Embedding provider to use ("azure", "openai", "local")
        """
        self.llm_client = llm_client
        self.use_embeddings = use_embeddings
        self.embedding_provider = embedding_provider

        # Initialize components
        self.git_strategy = GitStrategy()
        self.indexer: RepositoryIndexer | None = None

        if use_embeddings:
            self._initialize_indexer()

    def _initialize_indexer(self) -> None:
        """Initialize the repository indexer for vector embeddings."""
        try:
            self.indexer = RepositoryIndexer(
                collection_name="repo_gist_embeddings",
                persist_dir=str(CHROMA_DB_PATH),
                embedding_provider=self.embedding_provider,
            )
            logger.info(
                f"Repository indexer initialized with {self.embedding_provider} embeddings"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize indexer: {e}")
            self.indexer = None

    def analyze_repository(
        self, repo_url: str, branch: str = "main", query_context: str = ""
    ) -> dict[str, Any]:
        """
        Analyze a remote Git repository with simplified approach.

        Args:
            repo_url: URL of the remote Git repository
            branch: Branch to analyze (default: main)
            query_context: Context about what we're looking for

        Returns:
            Dictionary containing repository analysis
        """
        try:
            logger.info(f"Analyzing repository: {repo_url}")

            # Step 1: Analyze repository and determine optimal Git strategy
            strategy_analysis = self.git_strategy.analyze_repository(
                repo_url, query_context
            )
            logger.info(f"Git strategy: {strategy_analysis['strategy']['strategy']}")

            # Step 2: Execute intelligent checkout
            with tempfile.TemporaryDirectory() as temp_dir:
                target_path = Path(temp_dir) / "repo"
                checkout_path, checkout_metadata = self.git_strategy.execute_checkout(
                    repo_url, strategy_analysis["strategy"], target_path
                )

                # Switch to specified branch if needed
                if branch != "main":
                    self._switch_branch(checkout_path, branch)

                # Step 3: Basic file analysis and chunking
                file_analysis = self._analyze_repository_files(checkout_path)
                chunks = self._create_basic_chunks(checkout_path, file_analysis)
                logger.info(f"Generated {len(chunks)} chunks")

                # Step 4: Create embeddings if enabled
                embedding_analysis = None
                if self.use_embeddings and self.indexer:
                    embedding_analysis = self._create_simple_embeddings(chunks)

                # Step 5: Generate enhanced summary
                gist = self._generate_enhanced_gist(
                    file_analysis, chunks, repo_url, embedding_analysis
                )

                return {
                    "repository_url": repo_url,
                    "branch": branch,
                    "strategy_analysis": strategy_analysis,
                    "checkout_metadata": checkout_metadata,
                    "file_analysis": file_analysis,
                    "chunks_count": len(chunks),
                    "embedding_analysis": embedding_analysis,
                    "gist": gist,
                    "status": "success",
                }

        except Exception as e:
            logger.error(f"Repository analysis failed: {e}")
            return {
                "repository_url": repo_url,
                "branch": branch,
                "error": str(e),
                "status": "error",
            }

    def _switch_branch(self, repo_path: Path, branch: str) -> None:
        """Switch to specified branch."""
        try:
            repo = Repo(repo_path)
            if branch in [ref.name for ref in repo.remotes.origin.refs]:
                repo.git.checkout(branch)
                logger.info(f"Switched to branch: {branch}")
            else:
                logger.warning(f"Branch {branch} not found, staying on current branch")
        except Exception as e:
            logger.warning(f"Failed to switch to branch {branch}: {e}")

    def _analyze_repository_files(self, repo_path: Path) -> dict[str, Any]:
        """Analyze repository files and structure."""
        analysis: dict[str, Any] = {
            "total_files": 0,
            "total_lines": 0,
            "languages": {},
            "file_types": {},
            "key_files": [],
            "directories": [],
            "structure": {},
        }

        try:
            # Analyze files
            for file_path in repo_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    analysis["total_files"] += 1

                    # Count lines
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            analysis["total_lines"] += len(lines)
                    except Exception:
                        pass

                    # Track file types
                    ext = file_path.suffix.lower()
                    if ext:
                        analysis["file_types"][ext] = (
                            analysis["file_types"].get(ext, 0) + 1
                        )

                    # Identify key files
                    if self._is_key_file(file_path):
                        analysis["key_files"].append(
                            str(file_path.relative_to(repo_path))
                        )

            # Analyze directories
            analysis["directories"] = [
                str(d.relative_to(repo_path))
                for d in repo_path.rglob("*")
                if d.is_dir() and not d.name.startswith(".")
            ]

            # Detect languages
            analysis["languages"] = self._detect_languages(analysis["file_types"])

            # Generate structure overview
            analysis["structure"] = self._generate_structure_overview(repo_path)

            logger.info(
                f"Analyzed {analysis['total_files']} files with {analysis['total_lines']} total lines"
            )

        except Exception as e:
            logger.error(f"Repository file analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    def _is_key_file(self, file_path: Path) -> bool:
        """Check if a file is considered key/important."""
        key_files = [
            "README.md",
            "main.py",
            "app.py",
            "index.js",
            "package.json",
            "requirements.txt",
            "Dockerfile",
            "Makefile",
            "setup.py",
            "pyproject.toml",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
        ]

        # Check for key file names
        if file_path.name in key_files:
            return True

        # Check for key directories
        key_dirs = ["src", "app", "lib", "core", "main", "api"]
        for part in file_path.parts:
            if part in key_dirs:
                return True

        # Check for important extensions
        important_exts = [".py", ".js", ".ts", ".go", ".java", ".rs", ".md"]
        if file_path.suffix.lower() in important_exts:
            return True

        return False

    def _detect_languages(self, file_types: dict[str, int]) -> dict[str, int]:
        """Detect programming languages from file types."""
        language_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "JavaScript",
            ".tsx": "TypeScript",
            ".go": "Go",
            ".java": "Java",
            ".rs": "Rust",
            ".cpp": "C++",
            ".c": "C",
            ".md": "Markdown",
            ".json": "JSON",
            ".yaml": "YAML",
            ".yml": "YAML",
        }

        languages: dict[str, int] = {}
        for ext, count in file_types.items():
            lang = language_map.get(ext, "Other")
            languages[lang] = languages.get(lang, 0) + count

        return languages

    def _generate_structure_overview(self, repo_path: Path) -> dict[str, Any]:
        """Generate a structure overview of the repository."""
        structure: dict[str, Any] = {
            "top_level_files": [],
            "top_level_dirs": [],
            "has_tests": False,
            "has_docs": False,
            "has_config": False,
        }

        try:
            # Get top-level items
            for item in repo_path.iterdir():
                if item.is_file():
                    structure["top_level_files"].append(item.name)
                elif item.is_dir() and not item.name.startswith("."):
                    structure["top_level_dirs"].append(item.name)

            # Check for common patterns
            structure["has_tests"] = any(
                d.name.lower() in ["tests", "test", "__tests__"]
                for d in repo_path.rglob("*")
                if d.is_dir()
            )

            structure["has_docs"] = any(
                f.name.lower() in ["readme.md", "docs", "documentation"]
                for f in repo_path.rglob("*")
                if f.is_file()
            )

            structure["has_config"] = any(
                f.name.lower() in ["config", "settings", "conf"]
                for f in repo_path.rglob("*")
                if f.is_dir()
            )

        except Exception as e:
            logger.warning(f"Structure overview generation failed: {e}")

        return structure

    def _create_basic_chunks(
        self, repo_path: Path, file_analysis: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Create basic chunks from repository files."""
        chunks = []

        # Process key files
        key_files = file_analysis.get("key_files", [])[:20]  # Limit to 20 files

        for file_path_str in key_files:
            file_path = repo_path / file_path_str
            if file_path.exists() and file_path.is_file():
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    # Limit content size
                    if len(content) > 5000:  # 5KB limit
                        content = content[:5000] + "..."

                    # Create chunk
                    chunk = {
                        "content": content,
                        "metadata": {
                            "file_path": file_path_str,
                            "language": self._detect_language_from_path(file_path),
                            "start_line": 1,
                            "end_line": content.count("\n") + 1,
                            "chunk_hash": f"basic_{hash(content) % 10000}",
                            "node_type": "file",
                            "token_count": len(content) // 4,
                            "char_count": len(content),
                        },
                    }
                    chunks.append(chunk)

                except Exception as e:
                    logger.warning(f"Failed to process file {file_path}: {e}")

        return chunks

    def _detect_language_from_path(self, file_path: Path) -> str:
        """Detect programming language from file path."""
        ext = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".go": "go",
            ".java": "java",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".md": "markdown",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
        }
        return language_map.get(ext, "unknown")

    def _create_simple_embeddings(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """Create simple embeddings for chunks."""
        try:
            embedding_analysis: dict[str, Any] = {
                "chunk_embeddings": {},
                "embedding_stats": {},
            }

            # Process chunks for embeddings (limit to 10 for efficiency)
            for i, chunk in enumerate(chunks[:10]):
                try:
                    content = chunk["content"]
                    file_path = chunk["metadata"]["file_path"]

                    # Create embedding
                    if self.indexer is not None:
                        embedding = self.indexer.get_embedding(content)
                    else:
                        embedding = [0.0] * 1536  # Default embedding size

                    embedding_analysis["chunk_embeddings"][file_path] = embedding

                except Exception as e:
                    logger.warning(f"Failed to create embedding for chunk {i}: {e}")
                    continue

            # Generate embedding statistics
            embedding_analysis["embedding_stats"] = {
                "chunks_processed": len(embedding_analysis["chunk_embeddings"]),
                "total_chunks": len(chunks),
            }

            logger.info(
                f"Created embeddings for {embedding_analysis['embedding_stats']['chunks_processed']} chunks"
            )
            return embedding_analysis

        except Exception as e:
            logger.error(f"Simple embedding creation failed: {e}")
            return {"error": str(e)}

    def _generate_enhanced_gist(
        self,
        file_analysis: dict[str, Any],
        chunks: list[dict[str, Any]],
        repo_url: str,
        embedding_analysis: dict[str, Any] | None,
    ) -> str:
        """Generate enhanced gist using LLM."""
        try:
            # Prepare comprehensive context
            context_parts = [
                f"Repository: {repo_url}",
                f"Total Files: {file_analysis.get('total_files', 0)}",
                f"Total Lines: {file_analysis.get('total_lines', 0)}",
                f"Languages: {', '.join(file_analysis.get('languages', {}).keys())}",
                f"Key Files: {', '.join(file_analysis.get('key_files', [])[:10])}",
                f"Top-level Files: {', '.join(file_analysis.get('structure', {}).get('top_level_files', [])[:5])}",
                f"Top-level Directories: {', '.join(file_analysis.get('structure', {}).get('top_level_dirs', [])[:5])}",
            ]

            # Add structure information
            structure = file_analysis.get("structure", {})
            if structure.get("has_tests"):
                context_parts.append("Has test directory")
            if structure.get("has_docs"):
                context_parts.append("Has documentation")
            if structure.get("has_config"):
                context_parts.append("Has configuration files")

            # Add embedding information if available
            if embedding_analysis and not embedding_analysis.get("error"):
                context_parts.append(
                    f"Embeddings: {embedding_analysis['embedding_stats']['chunks_processed']} chunks processed"
                )

            # Add sample chunks
            context_parts.append("\nSample Code Content:")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                context_parts.append(
                    f"\nFile {i+1} ({chunk['metadata']['file_path']}):"
                )
                content = chunk["content"]
                if len(content) > 300:
                    content = content[:300] + "..."
                context_parts.append(content)

            context = "\n".join(context_parts)

            # Generate enhanced gist using LLM
            prompt = f"""
Analyze the following repository and generate a comprehensive, professional one-pager summary.

{context}

Please provide a clear, structured summary that covers:

1. **Project Overview**: What this repository is and its primary purpose
2. **Technology Stack**: Key technologies, frameworks, and languages used
3. **Architecture**: Main components, structure, and how they work together
4. **Key Features**: Important functionality and capabilities
5. **Getting Started**: How to set up, install, or use the project
6. **Project Structure**: Overview of important directories and files

Format as a professional technical document suitable for:
- New developers joining the project
- Technical documentation
- Project portfolio showcase

Be specific about file purposes, mention key technologies, and provide actionable insights.
"""

            if not hasattr(self.llm_client, "_call_llm"):
                raise AttributeError("LLMClient does not have _call_llm method")
            response = self.llm_client._call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert software architect and technical writer. Generate comprehensive, professional repository summaries that are clear, informative, and actionable.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            return str(response)

        except Exception as e:
            logger.error(f"Enhanced gist generation failed: {e}")
            return f"Repository analysis completed but gist generation failed: {e}"

    def export_analysis(self, analysis: dict[str, Any], format: str = "json") -> str:
        """
        Export analysis results in various formats.

        Args:
            analysis: Analysis results
            format: Export format ("json", "markdown", "yaml")

        Returns:
            Exported analysis as string
        """
        if format == "json":
            return json.dumps(analysis, indent=2, default=str)
        elif format == "markdown":
            return self._export_markdown(analysis)
        elif format == "yaml":
            return yaml.dump(analysis, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_markdown(self, analysis: dict[str, Any]) -> str:
        """Export analysis as Markdown."""
        md = f"""# Repository Analysis Report

## Repository Information
- **URL**: {analysis.get('repository_url', 'Unknown')}
- **Branch**: {analysis.get('branch', 'Unknown')}
- **Status**: {analysis.get('status', 'Unknown')}

## Analysis Summary
- **Git Strategy**: {analysis.get('strategy_analysis', {}).get('strategy', {}).get('strategy', 'Unknown')}
- **Total Files**: {analysis.get('file_analysis', {}).get('total_files', 0)}
- **Total Lines**: {analysis.get('file_analysis', {}).get('total_lines', 0)}
- **Languages**: {', '.join(analysis.get('file_analysis', {}).get('languages', {}).keys())}
- **Chunks Generated**: {analysis.get('chunks_count', 0)}

## Repository Gist
{analysis.get('gist', 'No gist available')}

## File Analysis
- **Key Files**: {', '.join(analysis.get('file_analysis', {}).get('key_files', [])[:10])}
- **File Types**: {analysis.get('file_analysis', {}).get('file_types', {})}
- **Structure**: {analysis.get('file_analysis', {}).get('structure', {})}
"""

        return md


# Example usage and testing
def demo_repo_gist() -> None:
    """Demonstrate the repository gist agent."""
    # Initialize components
    llm_client = create_llm_client(
        use_mock=USE_MOCK_LLM, model=LITELLM_MODEL, provider=LITELLM_PROVIDER
    )

    agent = RepoGistAgent(
        llm_client=llm_client, use_embeddings=True, embedding_provider="azure"
    )

    # Test repository
    test_repo = "https://github.com/varunbiluri/coda.git"

    print("Repository Gist Agent Demo")
    print("=" * 50)

    # Analyze repository
    result = agent.analyze_repository(
        test_repo, query_context="Analyze Python FastAPI application"
    )

    print(f"Status: {result['status']}")
    print(
        f"Strategy: {result.get('strategy_analysis', {}).get('strategy', {}).get('strategy', 'Unknown')}"
    )
    print(f"Files: {result.get('file_analysis', {}).get('total_files', 0)}")
    print(f"Chunks: {result.get('chunks_count', 0)}")

    if result.get("gist"):
        print(f"\nGist Preview: {result['gist'][:200]}...")

    # Export analysis
    markdown_export = agent.export_analysis(result, "markdown")
    print(f"\nMarkdown Export: {len(markdown_export)} characters")


if __name__ == "__main__":
    demo_repo_gist()
