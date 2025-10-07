"""
Git Strategy Module for Efficient Repository Checkouts.

This module implements intelligent algorithms to decide between sparse vs dense
git checkouts based on repository characteristics and query requirements.
"""

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from git import Repo

logger = logging.getLogger(__name__)


class GitStrategy:
    """
    Intelligent Git checkout strategy selector.

    Decides between sparse checkouts, shallow clones, and API-based file fetching
    based on repository characteristics and analysis requirements.
    """

    def __init__(self) -> None:
        """Initialize the Git strategy selector."""
        self.supported_hosts = {
            "github.com": self._github_api_fetch,
            "gitlab.com": self._gitlab_api_fetch,
            "bitbucket.org": self._bitbucket_api_fetch,
        }

    def analyze_repository(
        self, repo_url: str, query_context: str = "", branch: str = "main"
    ) -> dict[str, Any]:
        """
        Analyze repository and determine optimal checkout strategy.

        Args:
            repo_url: Repository URL
            query_context: Context about what we're looking for
            branch: Git branch to analyze (default: main)

        Returns:
            Dictionary with strategy recommendation and metadata
        """
        try:
            # Parse repository URL
            parsed_url = urlparse(repo_url)
            host = parsed_url.netloc.lower()

            # Get repository metadata
            repo_metadata = self._get_repo_metadata(repo_url, host)

            # Analyze query requirements
            query_analysis = self._analyze_query_requirements(query_context)

            # Determine optimal strategy
            strategy = self._determine_strategy(repo_metadata, query_analysis)

            return {
                "strategy": strategy,
                "repo_metadata": repo_metadata,
                "query_analysis": query_analysis,
                "recommendation": self._get_strategy_recommendation(strategy),
            }

        except Exception as e:
            logger.error(f"Failed to analyze repository {repo_url}: {e}")
            return {
                "strategy": "dense",
                "error": str(e),
                "recommendation": "Use dense checkout as fallback",
            }

    def execute_checkout(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> tuple[Path, dict[str, Any]]:
        """
        Execute the recommended checkout strategy.

        Args:
            repo_url: Repository URL
            strategy: Strategy configuration
            target_path: Target directory for checkout

        Returns:
            Tuple of (checkout_path, execution_metadata)
        """
        strategy_type = strategy.get("strategy", "dense")

        try:
            if strategy_type == "sparse":
                return self._execute_sparse_checkout(repo_url, strategy, target_path)
            elif strategy_type == "shallow":
                return self._execute_shallow_checkout(repo_url, strategy, target_path)
            elif strategy_type == "api":
                return self._execute_api_fetch(repo_url, strategy, target_path)
            else:
                return self._execute_dense_checkout(repo_url, strategy, target_path)

        except Exception as e:
            logger.error(f"Checkout execution failed: {e}")
            # Fallback to dense checkout
            return self._execute_dense_checkout(repo_url, strategy, target_path)

    def _get_repo_metadata(self, repo_url: str, host: str) -> dict[str, Any]:
        """Get repository metadata for strategy decision."""
        metadata = {
            "url": repo_url,
            "host": host,
            "size": 0,
            "languages": [],
            "file_count": 0,
            "has_large_files": False,
            "api_available": host in self.supported_hosts,
        }

        try:
            # Try to get repository size and language info via API
            if host == "github.com":
                api_metadata = self._get_github_metadata(repo_url)
                if api_metadata:
                    metadata.update(api_metadata)
            elif host == "gitlab.com":
                api_metadata = self._get_gitlab_metadata(repo_url)
                if api_metadata:
                    metadata.update(api_metadata)
            elif host == "bitbucket.org":
                api_metadata = self._get_bitbucket_metadata(repo_url)
                if api_metadata:
                    metadata.update(api_metadata)

        except Exception as e:
            logger.warning(f"Failed to get repository metadata: {e}")

        return metadata

    def _get_github_metadata(self, repo_url: str) -> dict[str, Any]:
        """Get GitHub repository metadata."""
        try:
            # Extract owner/repo from URL
            path_parts = urlparse(repo_url).path.strip("/").split("/")
            if len(path_parts) >= 2:
                owner, repo = path_parts[0], path_parts[1]

                # GitHub API call
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                response = requests.get(api_url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    return {
                        "size": data.get("size", 0),
                        "languages": list(data.get("language", "")),
                        "file_count": data.get("size", 0),  # Approximate
                        "has_large_files": data.get("size", 0) > 100000,  # > 100MB
                        "default_branch": data.get("default_branch", "main"),
                    }
        except Exception as e:
            logger.warning(f"GitHub API call failed: {e}")

        return {}

    def _get_gitlab_metadata(self, repo_url: str) -> dict[str, Any]:
        """Get GitLab repository metadata."""
        # Similar implementation for GitLab
        return {}

    def _get_bitbucket_metadata(self, repo_url: str) -> dict[str, Any]:
        """Get Bitbucket repository metadata."""
        # Similar implementation for Bitbucket
        return {}

    def _analyze_query_requirements(self, query_context: str) -> dict[str, Any]:
        """Analyze query requirements to determine checkout needs."""
        requirements: dict[str, Any] = {
            "needs_full_history": False,
            "needs_all_files": False,
            "target_languages": [],
            "file_patterns": [],
            "depth_requirement": "shallow",
        }

        query_lower = query_context.lower()

        # Analyze query for requirements
        if any(
            keyword in query_lower for keyword in ["history", "commits", "blame", "log"]
        ):
            requirements["needs_full_history"] = True
            requirements["depth_requirement"] = "full"

        if any(
            keyword in query_lower
            for keyword in ["all files", "entire codebase", "complete"]
        ):
            requirements["needs_all_files"] = True

        # Extract language requirements
        languages = [
            "python",
            "javascript",
            "java",
            "go",
            "rust",
            "cpp",
            "c",
            "typescript",
        ]
        for lang in languages:
            if lang in query_lower:
                requirements["target_languages"].append(lang)

        # Extract file patterns
        if "test" in query_lower:
            requirements["file_patterns"].extend(["*test*", "*spec*"])
        if "config" in query_lower:
            requirements["file_patterns"].extend(
                ["*config*", "*.json", "*.yaml", "*.yml"]
            )

        return requirements

    def _determine_strategy(
        self, repo_metadata: dict[str, Any], query_analysis: dict[str, Any]
    ) -> dict[str, Any]:
        """Determine the optimal checkout strategy."""
        repo_size = repo_metadata.get("size", 0)
        if isinstance(repo_size, str):
            logger.warning(
                f"Repository size is a string ('{repo_size}'); defaulting to 0. Metadata: {repo_metadata}"
            )
            repo_size = 0  # Default to 0 if size is not numeric

        has_api = repo_metadata.get("api_available", False)
        needs_full_history = query_analysis.get("needs_full_history", False)
        needs_all_files = query_analysis.get("needs_all_files", False)
        target_languages = query_analysis.get("target_languages", [])
        file_patterns = query_analysis.get("file_patterns", [])

        # Decision logic
        if not needs_all_files and (target_languages or file_patterns):
            if has_api and repo_size > 50000:  # Large repo with API access
                return {
                    "strategy": "api",
                    "patterns": file_patterns,
                    "languages": target_languages,
                    "reason": "Large repo with specific file requirements",
                }
            elif repo_size > 10000:  # Medium-large repo
                return {
                    "strategy": "sparse",
                    "patterns": file_patterns,
                    "languages": target_languages,
                    "reason": "Medium repo with specific requirements",
                }

        if needs_full_history:
            return {"strategy": "dense", "reason": "Full history required"}

        if repo_size > 100000:  # Very large repo
            return {
                "strategy": "shallow",
                "depth": 1,
                "reason": "Very large repository",
            }

        # Default to dense for small repos
        return {
            "strategy": "dense",
            "reason": "Small repository, full checkout optimal",
        }

    def _execute_sparse_checkout(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> tuple[Path, dict[str, Any]]:
        """Execute sparse checkout strategy."""
        try:
            # Clone with sparse checkout
            repo = Repo.clone_from(repo_url, target_path, depth=1)

            # Configure sparse checkout
            repo.git.sparse_checkout_set("--cone")

            # Add patterns based on strategy
            patterns = strategy.get("patterns", [])
            languages = strategy.get("languages", [])

            # Add language-specific patterns
            for lang in languages:
                if lang == "python":
                    patterns.extend(["*.py", "**/tests/**", "**/test_*"])
                elif lang == "javascript":
                    patterns.extend(["*.js", "*.ts", "*.jsx", "*.tsx"])
                elif lang == "java":
                    patterns.extend(["*.java", "**/test/**"])
                elif lang == "go":
                    patterns.extend(["*.go", "**/*_test.go"])

            # Add common important files
            patterns.extend(
                [
                    "README*",
                    "*.md",
                    "*.json",
                    "*.yaml",
                    "*.yml",
                    "*.toml",
                    "*.cfg",
                    "*.ini",
                    "Dockerfile*",
                    "Makefile*",
                ]
            )

            # Set sparse checkout patterns
            if patterns:
                repo.git.sparse_checkout_set("--set", *patterns)

            # Apply sparse checkout
            repo.git.sparse_checkout_apply()

            return target_path, {
                "strategy": "sparse",
                "patterns": patterns,
                "files_checked_out": len(list(target_path.rglob("*"))),
            }

        except Exception as e:
            logger.error(f"Sparse checkout failed: {e}")
            raise

    def _execute_shallow_checkout(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> tuple[Path, dict[str, Any]]:
        """Execute shallow checkout strategy."""
        try:
            depth = strategy.get("depth", 1)
            Repo.clone_from(repo_url, target_path, depth=depth)

            return target_path, {
                "strategy": "shallow",
                "depth": depth,
                "files_checked_out": len(list(target_path.rglob("*"))),
            }

        except Exception as e:
            logger.error(f"Shallow checkout failed: {e}")
            raise

    def _execute_api_fetch(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> tuple[Path, dict[str, Any]]:
        """Execute API-based file fetching strategy."""
        try:
            target_path.mkdir(parents=True, exist_ok=True)

            # Determine API handler
            parsed_url = urlparse(repo_url)
            host = parsed_url.netloc.lower()

            if host in self.supported_hosts:
                api_handler = self.supported_hosts[host]
                files_fetched = api_handler(repo_url, strategy, target_path)

                return target_path, {
                    "strategy": "api",
                    "files_fetched": files_fetched,
                    "api_used": host,
                }
            else:
                raise ValueError(f"API not supported for host: {host}")

        except Exception as e:
            logger.error(f"API fetch failed: {e}")
            raise

    def _execute_dense_checkout(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> tuple[Path, dict[str, Any]]:
        """Execute dense checkout strategy (fallback)."""
        try:
            Repo.clone_from(repo_url, target_path)

            return target_path, {
                "strategy": "dense",
                "files_checked_out": len(list(target_path.rglob("*"))),
            }

        except Exception as e:
            logger.error(f"Dense checkout failed: {e}")
            raise

    def _github_api_fetch(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> int:
        """Fetch files using GitHub API."""
        try:
            # Extract owner/repo from URL
            path_parts = urlparse(repo_url).path.strip("/").split("/")
            owner, repo = path_parts[0], path_parts[1]

            # Get repository contents
            api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"
            response = requests.get(api_url, timeout=10)

            if response.status_code == 200:
                contents = response.json()
                files_fetched = 0

                for item in contents:
                    if item["type"] == "file":
                        # Download file content
                        file_response = requests.get(item["download_url"], timeout=5)
                        if file_response.status_code == 200:
                            file_path = target_path / item["path"]
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            file_path.write_text(file_response.text)
                            files_fetched += 1

                return files_fetched
            else:
                raise Exception(f"GitHub API error: {response.status_code}")

        except Exception as e:
            logger.error(f"GitHub API fetch failed: {e}")
            return 0

    def _gitlab_api_fetch(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> int:
        """Fetch files using GitLab API."""
        # Implementation for GitLab API
        return 0

    def _bitbucket_api_fetch(
        self, repo_url: str, strategy: dict[str, Any], target_path: Path
    ) -> int:
        """Fetch files using Bitbucket API."""
        # Implementation for Bitbucket API
        return 0

    def _get_strategy_recommendation(self, strategy: dict[str, Any]) -> str:
        """Get human-readable strategy recommendation."""
        strategy_type = strategy.get("strategy", "dense")
        reason = strategy.get("reason", "Default strategy")

        recommendations = {
            "sparse": f"Sparse checkout recommended: {reason}",
            "shallow": f"Shallow clone recommended: {reason}",
            "api": f"API-based fetch recommended: {reason}",
            "dense": f"Full checkout recommended: {reason}",
        }

        return recommendations.get(strategy_type, "Unknown strategy")


# Example usage and testing
def demo_git_strategy() -> None:
    """Demonstrate the Git strategy system."""
    strategy = GitStrategy()

    test_repos = [
        {
            "url": "https://github.com/tiangolo/fastapi.git",
            "query": "Analyze Python web framework structure and API patterns",
        },
        {
            "url": "https://github.com/microsoft/vscode.git",
            "query": "Find TypeScript configuration files and test patterns",
        },
        {
            "url": "https://github.com/varunbiluri/coda.git",
            "query": "Get all Python files and documentation",
        },
    ]

    for repo in test_repos:
        print(f"\n{'='*60}")
        print(f"Repository: {repo['url']}")
        print(f"Query: {repo['query']}")
        print(f"{'='*60}")

        # Analyze repository
        analysis = strategy.analyze_repository(repo["url"], repo["query"])

        print(f"Strategy: {analysis['strategy']['strategy']}")
        print(f"Reason: {analysis['strategy']['reason']}")
        print(f"Recommendation: {analysis['recommendation']}")

        if "patterns" in analysis["strategy"]:
            print(f"Patterns: {analysis['strategy']['patterns']}")
        if "languages" in analysis["strategy"]:
            print(f"Languages: {analysis['strategy']['languages']}")


if __name__ == "__main__":
    demo_git_strategy()
