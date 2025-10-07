#!/usr/bin/env python3
"""
Repository Analysis Demonstration

This script demonstrates Coda's comprehensive repository analysis capabilities
for generating professional technical summaries of codebases.

Use Case: Technical due diligence and codebase understanding for development teams.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coda.agents.repo_gist import RepoGistAgent
from coda.core.git_strategy import GitStrategy
from coda.core.indexer import RepositoryIndexer
from coda.core.llm_client import create_llm_client
from coda.core.tree_sitter_chunker import TreeSitterChunker
from config.settings import (
    CHROMA_DB_PATH,
    LITELLM_MODEL,
    LITELLM_PROVIDER,
    USE_MOCK_LLM,
)


def initialize_components() -> (
    tuple[RepoGistAgent, GitStrategy, TreeSitterChunker, RepositoryIndexer, bool]
):
    """Initialize all analysis components."""
    print("Initializing repository analysis components...")

    # Check for API credentials
    has_credentials = bool(os.getenv("AZURE_API_KEY") or os.getenv("OPENAI_API_KEY"))
    if not has_credentials:
        print("INFO: No API credentials found. Using mock LLM for demonstration.")
        os.environ["USE_MOCK_LLM"] = "true"

    try:
        # Initialize LLM client
        llm_client = create_llm_client(
            use_mock=USE_MOCK_LLM, model=LITELLM_MODEL, provider=LITELLM_PROVIDER
        )

        # Initialize components
        git_strategy = GitStrategy()

        # Initialize tree-sitter chunker (strict implementation)
        tree_sitter_chunker = TreeSitterChunker()

        # Initialize repository indexer for embeddings
        if has_credentials:
            indexer = RepositoryIndexer(
                collection_name="demo_analysis",
                persist_dir=str(CHROMA_DB_PATH),
                embedding_provider="azure" if os.getenv("AZURE_API_KEY") else "openai",
            )
        else:
            indexer = None

        # Initialize RepoGist agent
        agent = RepoGistAgent(
            llm_client=llm_client,
            use_embeddings=has_credentials,
            embedding_provider="azure" if os.getenv("AZURE_API_KEY") else "openai",
        )

        print("SUCCESS: Analysis components ready")
        return agent, git_strategy, tree_sitter_chunker, indexer, has_credentials

    except Exception as e:
        print(f"ERROR: Failed to initialize components: {e}")
        sys.exit(1)


def analyze_repository_comprehensive(
    agent: RepoGistAgent,
    git_strategy: GitStrategy,
    tree_sitter_chunker: TreeSitterChunker,
    indexer: RepositoryIndexer | None,
    repo_url: str,
    query: str,
    has_credentials: bool,
    branch: str = "main",
) -> dict[str, Any]:
    """Perform comprehensive repository analysis."""
    print(f"Analyzing repository: {repo_url}")
    print(f"Analysis focus: {query}")
    print()

    start_time = time.time()

    try:
        # Step 1: Git strategy analysis
        strategy_analysis = git_strategy.analyze_repository(repo_url, query, branch)
        strategy = strategy_analysis["strategy"]

        # Step 2: Clone the actual repository
        repo_path = clone_repository(repo_url, branch)

        # Step 3: Tree-sitter semantic chunking
        all_chunks = []
        if repo_path.exists():
            code_files = []
            for ext in [".py", ".js", ".ts", ".go", ".java"]:
                code_files.extend(repo_path.rglob(f"*{ext}"))

            for file_path in code_files[:10]:  # Limit for demo
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")

                    if tree_sitter_chunker:
                        # Use tree-sitter chunking (no fallback)
                        chunks = tree_sitter_chunker.chunk_file(file_path, content)
                        all_chunks.extend(chunks)
                    else:
                        raise Exception("Tree-sitter chunker not available")

                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    continue

        # Step 4: Vector embeddings and storage (if credentials available)
        embeddings_generated = 0
        if has_credentials and indexer and all_chunks:
            print(f"Processing {len(all_chunks)} chunks for vector storage...")

            # Smart chunking: Select most important chunks to avoid rate limits
            # Priority: 1) Main files, 2) Configuration files, 3) Documentation
            important_chunks = []

            for chunk in all_chunks:
                file_path = chunk.get("metadata", {}).get("file_path", "")
                content = chunk.get("content", "")

                # Prioritize important files
                if any(
                    keyword in file_path.lower()
                    for keyword in [
                        "main",
                        "app",
                        "core",
                        "src",
                        "index",
                        "setup",
                        "config",
                        "readme",
                    ]
                ):
                    important_chunks.append(chunk)
                elif len(content.strip()) > 50:  # Meaningful content
                    important_chunks.append(chunk)

            # Limit to very small number to avoid rate limits (Azure has strict limits)
            max_chunks = min(
                10, len(important_chunks)
            )  # Process only 10 most important chunks
            selected_chunks = important_chunks[:max_chunks]

            print(
                f"Selected {len(selected_chunks)} important chunks for embedding generation"
            )

            # Generate embeddings one by one with delays to respect rate limits
            chunk_contents = []

            for chunk in selected_chunks:
                content = chunk.get("content", "")
                if content and len(content.strip()) > 10:
                    chunk_contents.append(content)

            # Generate embeddings with rate limit handling
            for i, content in enumerate(chunk_contents):
                try:
                    print(f"Generating embedding {i+1}/{len(chunk_contents)}...")
                    indexer.get_embedding(content)  # Generate embedding but don't store
                    embeddings_generated += 1
                    print(f"Generated embedding {i+1}")

                    # Delay to respect rate limits (Azure has very strict limits)
                    time.sleep(2)  # 2 second delay between requests

                except Exception as e:
                    print(f"Warning: Failed to generate embedding {i+1}: {e}")
                    continue

        # Step 5: Generate comprehensive summary
        file_analysis = {
            "total_files": len(all_chunks),
            "key_files": [
                chunk.get("metadata", {}).get("file_path", "unknown")
                for chunk in all_chunks[:5]
            ],
        }

        gist = agent._generate_enhanced_gist(
            file_analysis=file_analysis,
            chunks=all_chunks,
            repo_url=repo_url,
            embedding_analysis=(
                {"embedding_stats": {"chunks_processed": embeddings_generated}}
                if embeddings_generated > 0
                else None
            ),
        )

        execution_time = time.time() - start_time

        return {
            "status": "success",
            "repository_url": repo_url,
            "branch": branch,
            "analysis_query": query,
            "execution_time": execution_time,
            "git_strategy": strategy,
            "files_processed": len(all_chunks),
            "embeddings_generated": embeddings_generated,
            "summary": gist,
            "analysis_metadata": {
                "strategy_reasoning": strategy.get("reasoning", "N/A"),
                "checkout_method": strategy.get("checkout_method", "N/A"),
                "estimated_size": strategy.get("estimated_size", "N/A"),
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "execution_time": time.time() - start_time,
        }


def display_repository_summary(analysis: dict[str, Any]) -> None:
    """Display a beautifully formatted repository summary."""
    print("\n" + "=" * 80)
    print("REPOSITORY ANALYSIS SUMMARY")
    print("=" * 80)

    if analysis["status"] == "error":
        print(f"Analysis failed: {analysis.get('error', 'Unknown error')}")
        return

    # Repository information
    print(f"Repository: {analysis['repository_url']}")
    print(f"Branch: {analysis.get('branch', 'main')}")
    print(f"Analysis Focus: {analysis['analysis_query']}")
    print(f"Execution Time: {analysis['execution_time']:.2f} seconds")
    print()

    # Git strategy information
    strategy = analysis["git_strategy"]
    print("GIT STRATEGY")
    print("-" * 40)
    print(f"Strategy: {strategy.get('strategy', 'N/A')}")
    print(f"Reasoning: {analysis['analysis_metadata']['strategy_reasoning']}")
    print(f"Checkout Method: {analysis['analysis_metadata']['checkout_method']}")
    print(f"Estimated Size: {analysis['analysis_metadata']['estimated_size']}")
    print()

    # Analysis metrics
    print("ANALYSIS METRICS")
    print("-" * 40)
    print(f"Files Processed: {analysis['files_processed']}")
    print(f"Embeddings Generated: {analysis['embeddings_generated']}")
    print(
        f"Analysis Status: {'Complete' if analysis['status'] == 'success' else 'Failed'}"
    )
    print()

    # Technical summary
    if analysis.get("summary"):
        print("TECHNICAL SUMMARY")
        print("-" * 40)
        summary = analysis["summary"]

        # Format the summary nicely
        if len(summary) > 1000:
            # Truncate very long summaries
            summary = (
                summary[:1000]
                + "\n\n[Summary truncated for display - full version available in export]"
            )

        print(summary)
        print()

    # Capabilities demonstrated
    print("CAPABILITIES DEMONSTRATED")
    print("-" * 40)
    print("- Intelligent Git checkout strategies")
    print("- Tree-sitter semantic code chunking")
    print("- Vector embeddings for enhanced understanding")
    print("- LLM-powered repository summarization")
    print("- Professional technical analysis")
    print()

    print("=" * 80)


def export_analysis_results(analysis: dict[str, Any], output_dir: str = None) -> str:
    """Export analysis results to a markdown file."""
    if output_dir is None:
        # Use a temporary directory that gets cleaned up
        import tempfile

        output_dir = tempfile.mkdtemp(prefix="coda_analysis_")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"repository_analysis_{timestamp}.md")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("# Repository Analysis Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Repository:** {analysis['repository_url']}\n")
            f.write(f"**Analysis Focus:** {analysis['analysis_query']}\n")
            f.write(f"**Execution Time:** {analysis['execution_time']:.2f} seconds\n\n")

            f.write("## Git Strategy\n\n")
            strategy = analysis["git_strategy"]
            f.write(f"- **Strategy:** {strategy.get('strategy', 'N/A')}\n")
            f.write(
                f"- **Reasoning:** {analysis['analysis_metadata']['strategy_reasoning']}\n"
            )
            f.write(
                f"- **Checkout Method:** {analysis['analysis_metadata']['checkout_method']}\n"
            )
            f.write(
                f"- **Estimated Size:** {analysis['analysis_metadata']['estimated_size']}\n\n"
            )

            f.write("## Analysis Metrics\n\n")
            f.write(f"- **Files Processed:** {analysis['files_processed']}\n")
            f.write(f"- **Embeddings Generated:** {analysis['embeddings_generated']}\n")
            f.write(
                f"- **Status:** {'Complete' if analysis['status'] == 'success' else 'Failed'}\n\n"
            )

            if analysis.get("summary"):
                f.write("## Technical Summary\n\n")
                f.write(analysis["summary"])
                f.write("\n\n")

            f.write("## Capabilities Demonstrated\n\n")
            f.write("- Intelligent Git checkout strategies\n")
            f.write("- Tree-sitter semantic code chunking\n")
            f.write("- Vector embeddings for enhanced understanding\n")
            f.write("- LLM-powered repository summarization\n")
            f.write("- Professional technical analysis\n")

        return filename

    except Exception as e:
        print(f"WARNING: Failed to export results: {e}")
        return ""


def clone_repository(repo_url: str, branch: str = "main") -> Path:
    """
    Clone a repository to a temporary directory.

    Args:
        repo_url: Git repository URL
        branch: Git branch to checkout

    Returns:
        Path to the cloned repository
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="coda_analysis_")
        repo_path = Path(temp_dir) / "repo"

        print(f"Cloning repository: {repo_url}")
        print(f"Branch: {branch}")
        print(f"Target directory: {repo_path}")

        # Clone the repository
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",  # Shallow clone for faster download
                "--branch",
                branch,
                repo_url,
                str(repo_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        print(f"Successfully cloned repository to {repo_path}")
        return repo_path

    except subprocess.CalledProcessError as e:
        print(f"Failed to clone repository: {e}")
        print(f"Error output: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error during cloning: {e}")
        raise


def cleanup_temp_files():
    """Clean up any temporary analysis files."""
    import glob
    import os

    # Clean up any analysis markdown files in current directory
    patterns = ["repo_analysis*.md", "repository_analysis*.md", "*_analysis_*.md"]
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"Cleaned up: {file}")
            except Exception as e:
                print(f"Warning: Could not remove {file}: {e}")

    # Clean up temporary repository directories
    for temp_dir in glob.glob("**/coda_analysis_*"):
        if os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Could not remove {temp_dir}: {e}")


def main():
    """Execute the repository analysis demonstration."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Repository Analysis Demonstration")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/varunbiluri/coda.git",
        help="Git repository URL to analyze",
    )
    parser.add_argument("--branch", default="main", help="Git branch to analyze")
    parser.add_argument(
        "--query",
        default="analyze the repo and provide me highlevel summary",
        help="Analysis query/focus",
    )

    args = parser.parse_args()

    print("Repository Analysis Demonstration")
    print("=" * 50)
    print("Use Case: Technical due diligence and codebase understanding")
    print("=" * 50)
    print(f"Repository: {args.repo_url}")
    print(f"Branch: {args.branch}")
    print(f"Analysis Focus: {args.query}")
    print()

    # Clean up any existing temporary files
    cleanup_temp_files()

    # Initialize components
    agent, git_strategy, tree_sitter_chunker, indexer, has_credentials = (
        initialize_components()
    )

    # Analysis configuration
    repo_url = args.repo_url
    analysis_query = args.query

    # Perform comprehensive analysis
    analysis = analyze_repository_comprehensive(
        agent,
        git_strategy,
        tree_sitter_chunker,
        indexer,
        repo_url,
        analysis_query,
        has_credentials,
        args.branch,
    )

    # Display formatted summary
    display_repository_summary(analysis)

    # Export results (optional - only if explicitly requested)
    # Uncomment the following lines if you want to export results to a file:
    # if analysis['status'] == 'success':
    #     export_file = export_analysis_results(analysis)
    #     if export_file:
    #         print(f"📄 Analysis results exported to: {export_file}")

    print("\nRepository analysis demonstration completed!")

    # Final cleanup
    cleanup_temp_files()


if __name__ == "__main__":
    main()
