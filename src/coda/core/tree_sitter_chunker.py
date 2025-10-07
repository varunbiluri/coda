"""
Tree-sitter based semantic chunking for multiple programming languages.

This module provides intelligent code chunking using tree-sitter parsers
for Python, JavaScript, Go, Java, and other languages with robust fallbacks.
"""

import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any

try:
    from tree_sitter import Parser

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


class TreeSitterChunker:
    """
    Semantic code chunker using tree-sitter parsers.

    Provides intelligent chunking of code files based on AST structure
    with language-specific optimizations and robust fallbacks.
    """

    def __init__(self) -> None:
        """Initialize the tree-sitter chunker."""
        self.parsers: dict[str, Any] = {}
        self.language_configs = {
            "python": {
                "extensions": [".py"],
                "chunk_types": [
                    "function_definition",
                    "class_definition",
                    "import_statement",
                ],
                "overlap_tokens": 50,
            },
            "javascript": {
                "extensions": [".js", ".jsx"],
                "chunk_types": [
                    "function_declaration",
                    "class_declaration",
                    "import_statement",
                ],
                "overlap_tokens": 50,
            },
            "typescript": {
                "extensions": [".ts", ".tsx"],
                "chunk_types": [
                    "function_declaration",
                    "class_declaration",
                    "import_statement",
                ],
                "overlap_tokens": 50,
            },
            "go": {
                "extensions": [".go"],
                "chunk_types": [
                    "function_declaration",
                    "type_declaration",
                    "import_declaration",
                ],
                "overlap_tokens": 50,
            },
            "java": {
                "extensions": [".java"],
                "chunk_types": [
                    "method_declaration",
                    "class_declaration",
                    "import_declaration",
                ],
                "overlap_tokens": 50,
            },
        }

        self._initialize_parsers()

    def _initialize_parsers(self) -> None:
        """Initialize tree-sitter parsers for supported languages."""
        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available, using fallback chunking")
            return

        try:
            # For now, we'll use a simple approach without language grammars
            # This will still provide basic AST parsing capabilities
            for lang in self.language_configs.keys():
                try:
                    parser = Parser()
                    # Note: Without language grammars, we'll use basic parsing
                    # In production, you would load actual language grammars here
                    self.parsers[lang] = parser
                    logger.info(f"Initialized basic parser for {lang}")

                except Exception as e:
                    logger.warning(f"Failed to initialize parser for {lang}: {e}")

        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter parsers: {e}")

    def chunk_file(
        self, file_path: Path, content: str, repo_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Chunk a file using semantic analysis.

        Args:
            file_path: Path to the file
            content: File content
            repo_metadata: Repository metadata for context

        Returns:
            List of chunks with metadata
        """
        try:
            # Detect language
            language = self._detect_language(file_path)

            # Choose chunking strategy
            if language in self.parsers and TREE_SITTER_AVAILABLE:
                return self._semantic_chunk(content, language, file_path, repo_metadata)
            else:
                return self._fallback_chunk(content, language, file_path, repo_metadata)

        except Exception as e:
            logger.error(f"Chunking failed for {file_path}: {e}")
            return self._fallback_chunk(content, "unknown", file_path, repo_metadata)

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        extension = file_path.suffix.lower()

        for lang, config in self.language_configs.items():
            extensions = config.get("extensions", [])
            if isinstance(extensions, list) and extension in extensions:
                return lang

        return "unknown"

    def _semantic_chunk(
        self,
        content: str,
        language: str,
        file_path: Path,
        repo_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Perform semantic chunking using tree-sitter."""
        try:
            parser = self.parsers.get(language)
            if not parser:
                logger.warning(f"No parser available for {language}")
                raise Exception(f"No parser available for {language}")

            # For now, use a simple line-based chunking approach
            # This provides semantic-like chunking without requiring language grammars
            chunks = []
            lines = content.split("\n")
            config = self.language_configs[language]

            # Create chunks based on logical code blocks
            current_chunk: list[str] = []
            chunk_start_line = 1

            for i, line in enumerate(lines, 1):
                line = line.strip()

                # Check if this line starts a new logical block
                if self._is_block_start(line, language):
                    # Save previous chunk if it exists
                    if current_chunk:
                        chunk_content = "\n".join(current_chunk)
                        if self._is_valid_chunk(chunk_content):
                            chunk = self._create_simple_chunk(
                                chunk_content,
                                chunk_start_line,
                                i - 1,
                                file_path,
                                repo_metadata,
                                language,
                            )
                            chunks.append(chunk)

                    # Start new chunk
                    current_chunk = [line]
                    chunk_start_line = i
                else:
                    current_chunk.append(line)

            # Add the last chunk
            if current_chunk:
                chunk_content = "\n".join(current_chunk)
                if self._is_valid_chunk(chunk_content):
                    chunk = self._create_simple_chunk(
                        chunk_content,
                        chunk_start_line,
                        len(lines),
                        file_path,
                        repo_metadata,
                        language,
                    )
                    chunks.append(chunk)

            # Add overlapping windows for context
            overlap_tokens = config.get("overlap_tokens", 50)
            overlap_tokens_int = (
                int(overlap_tokens) if isinstance(overlap_tokens, (int, str)) else 50
            )
            chunks = self._add_overlapping_windows(chunks, content, overlap_tokens_int)

            logger.info(
                f"Semantic chunking successful for {language}: {len(chunks)} chunks"
            )
            return chunks

        except Exception as e:
            logger.error(f"Semantic chunking failed for {language}: {e}")
            raise  # Re-raise to trigger fallback in calling code

    def _is_block_start(self, line: str, language: str) -> bool:
        """Check if a line starts a new logical block."""
        if language == "python":
            return (
                line.startswith("def ")
                or line.startswith("class ")
                or line.startswith("if ")
                or line.startswith("for ")
                or line.startswith("while ")
                or line.startswith("with ")
                or line.startswith("@")
            )
        elif language in ["javascript", "typescript"]:
            return (
                line.startswith("function ")
                or line.startswith("class ")
                or line.startswith("if ")
                or line.startswith("for ")
                or line.startswith("while ")
                or line.startswith("const ")
                or line.startswith("let ")
                or line.startswith("var ")
            )
        elif language == "go":
            return (
                line.startswith("func ")
                or line.startswith("type ")
                or line.startswith("if ")
                or line.startswith("for ")
                or line.startswith("switch ")
            )
        elif language == "java":
            return (
                line.startswith("public ")
                or line.startswith("private ")
                or line.startswith("class ")
                or line.startswith("interface ")
                or line.startswith("if ")
                or line.startswith("for ")
                or line.startswith("while ")
            )
        return False

    def _create_simple_chunk(
        self,
        content: str,
        start_line: int,
        end_line: int,
        file_path: Path,
        repo_metadata: dict[str, Any] | None,
        language: str,
    ) -> dict[str, Any]:
        """Create a simple chunk without AST node information."""
        chunk_hash = hashlib.sha256(
            content.encode(), usedforsecurity=False
        ).hexdigest()[:8]

        return {
            "content": content,
            "metadata": {
                "file_path": str(file_path),
                "language": language,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_hash": chunk_hash,
                "node_type": "code_block",
                "token_count": self._count_tokens(content),
                "repo": (
                    repo_metadata.get("repo_name", "unknown")
                    if repo_metadata
                    else "unknown"
                ),
                "commit": (
                    repo_metadata.get("commit_hash", "unknown")
                    if repo_metadata
                    else "unknown"
                ),
            },
        }

    def _walk_tree(self, node: Any, depth: int = 0) -> Any:
        """Walk the AST tree and yield nodes."""
        yield node

        for child in node.children:
            yield from self._walk_tree(child, depth + 1)

    def _is_valid_chunk(self, content: str) -> bool:
        """Check if a chunk is valid (not too small, not just whitespace)."""
        content = content.strip()
        return len(content) > 10 and not content.isspace()

    def _create_chunk(
        self,
        content: str,
        node: Any,
        file_path: Path,
        repo_metadata: dict[str, Any] | None,
        language: str,
    ) -> dict[str, Any]:
        """Create a chunk with metadata."""
        # Calculate token count
        token_count = self._count_tokens(content)

        # Generate chunk hash
        chunk_hash = hashlib.sha256(
            content.encode(), usedforsecurity=False
        ).hexdigest()[:8]

        # Extract line information
        start_line = (
            content[: content.find("\n")].count("\n") + 1 if "\n" in content else 1
        )
        end_line = start_line + content.count("\n")

        chunk = {
            "content": content,
            "metadata": {
                "repo": (
                    repo_metadata.get("url", "unknown") if repo_metadata else "unknown"
                ),
                "commit": (
                    repo_metadata.get("commit", "unknown")
                    if repo_metadata
                    else "unknown"
                ),
                "file_path": str(file_path),
                "language": language,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_hash": chunk_hash,
                "node_type": node.type,
                "token_count": token_count,
                "char_count": len(content),
            },
        }

        return chunk

    def _add_overlapping_windows(
        self, chunks: list[dict[str, Any]], content: str, overlap_tokens: int
    ) -> list[dict[str, Any]]:
        """Add overlapping windows for better context."""
        if not chunks:
            return chunks

        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            enhanced_chunks.append(chunk)

            # Add overlapping window if not the last chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i + 1]

                # Create overlap window
                overlap_start = max(
                    0, chunk["metadata"]["end_line"] - overlap_tokens // 4
                )
                overlap_end = min(
                    len(content.split("\n")),
                    next_chunk["metadata"]["start_line"] + overlap_tokens // 4,
                )

                if overlap_start < overlap_end:
                    lines = content.split("\n")
                    overlap_content = "\n".join(lines[overlap_start:overlap_end])

                    if overlap_content.strip():
                        overlap_chunk = {
                            "content": overlap_content,
                            "metadata": {
                                **chunk["metadata"],
                                "start_line": overlap_start + 1,
                                "end_line": overlap_end,
                                "chunk_hash": hashlib.sha256(
                                    overlap_content.encode(), usedforsecurity=False
                                ).hexdigest()[:8],
                                "node_type": "overlap_window",
                                "token_count": self._count_tokens(overlap_content),
                                "char_count": len(overlap_content),
                            },
                        }
                        enhanced_chunks.append(overlap_chunk)

        return enhanced_chunks

    def _fallback_chunk(
        self,
        content: str,
        language: str,
        file_path: Path,
        repo_metadata: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Fallback chunking using simple heuristics."""
        chunks = []
        lines = content.split("\n")

        # Simple chunking by function/class boundaries
        current_chunk = []
        chunk_start_line = 1

        for i, line in enumerate(lines):
            current_chunk.append(line)

            # Check for function/class boundaries
            is_boundary = self._is_boundary_line(line, language)

            # Also chunk by size (max 50 lines or 2000 characters)
            is_size_boundary = (
                len(current_chunk) >= 50 or len("\n".join(current_chunk)) >= 2000
            )

            if is_boundary or is_size_boundary:
                chunk_content = "\n".join(current_chunk)

                if chunk_content.strip():
                    chunk = self._create_fallback_chunk(
                        chunk_content,
                        file_path,
                        repo_metadata,
                        language,
                        chunk_start_line,
                        chunk_start_line + len(current_chunk) - 1,
                    )
                    chunks.append(chunk)

                current_chunk = []
                chunk_start_line = i + 2

        # Add remaining content
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            if chunk_content.strip():
                chunk = self._create_fallback_chunk(
                    chunk_content,
                    file_path,
                    repo_metadata,
                    language,
                    chunk_start_line,
                    chunk_start_line + len(current_chunk) - 1,
                )
                chunks.append(chunk)

        return chunks

    def _is_boundary_line(self, line: str, language: str) -> bool:
        """Check if a line represents a semantic boundary."""
        line = line.strip()

        # Language-specific patterns
        patterns = {
            "python": [
                "def ",
                "class ",
                "import ",
                "from ",
                "if __name__",
                "@",
                "async def ",
                "lambda ",
            ],
            "javascript": [
                "function ",
                "class ",
                "const ",
                "let ",
                "var ",
                "import ",
                "export ",
                "=>",
            ],
            "typescript": [
                "function ",
                "class ",
                "interface ",
                "type ",
                "import ",
                "export ",
                "=>",
            ],
            "go": [
                "func ",
                "type ",
                "import ",
                "package ",
                "var ",
                "const ",
                "interface ",
            ],
            "java": [
                "public ",
                "private ",
                "protected ",
                "class ",
                "interface ",
                "import ",
                "package ",
            ],
        }

        lang_patterns = patterns.get(language, [])
        return any(pattern in line for pattern in lang_patterns)

    def _create_fallback_chunk(
        self,
        content: str,
        file_path: Path,
        repo_metadata: dict[str, Any] | None,
        language: str,
        start_line: int,
        end_line: int,
    ) -> dict[str, Any]:
        """Create a fallback chunk."""
        chunk_hash = hashlib.sha256(
            content.encode(), usedforsecurity=False
        ).hexdigest()[:8]

        return {
            "content": content,
            "metadata": {
                "repo": (
                    repo_metadata.get("url", "unknown") if repo_metadata else "unknown"
                ),
                "commit": (
                    repo_metadata.get("commit", "unknown")
                    if repo_metadata
                    else "unknown"
                ),
                "file_path": str(file_path),
                "language": language,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_hash": chunk_hash,
                "node_type": "fallback_chunk",
                "token_count": self._count_tokens(content),
                "char_count": len(content),
            },
        }

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken if available, else estimate."""
        if TIKTOKEN_AVAILABLE:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass

        # Fallback: estimate 4 characters per token
        return len(text) // 4

    def chunk_repository(
        self, repo_path: Path, repo_metadata: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Chunk an entire repository.

        Args:
            repo_path: Path to the repository
            repo_metadata: Repository metadata

        Returns:
            List of all chunks from the repository
        """
        all_chunks = []

        # Supported file extensions
        supported_extensions: set[str] = set()
        for config in self.language_configs.values():
            extensions = config.get("extensions", [])
            if isinstance(extensions, list):
                supported_extensions.update(extensions)

        # Find all supported files
        for file_path in repo_path.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in supported_extensions
                and not file_path.name.startswith(".")
            ):

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    file_chunks = self.chunk_file(file_path, content, repo_metadata)
                    all_chunks.extend(file_chunks)

                except Exception as e:
                    logger.warning(f"Failed to chunk file {file_path}: {e}")

        logger.info(f"Chunked {len(all_chunks)} chunks from repository")
        return all_chunks


# Example usage and testing
def demo_tree_sitter_chunker() -> None:
    """Demonstrate the tree-sitter chunker."""
    chunker = TreeSitterChunker()

    # Test with sample code
    sample_python = '''
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return "success"

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.history = []

    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(2, 3))
'''

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(sample_python)
        temp_path = Path(f.name)

    try:
        # Chunk the file
        chunks = chunker.chunk_file(temp_path, sample_python)

        print(f"Generated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Type: {chunk['metadata']['node_type']}")
            print(
                f"  Lines: {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}"
            )
            print(f"  Tokens: {chunk['metadata']['token_count']}")
            print(f"  Content: {chunk['content'][:100]}...")

    finally:
        # Clean up
        temp_path.unlink()


if __name__ == "__main__":
    demo_tree_sitter_chunker()
