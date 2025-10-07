"""
LLM Pipeline for Multi-level Repository Summarization.

This module defines prompt templates and reducer LLM pipeline to generate
file → module → repository summaries with proper citations and JSON schemas.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FileSummary(BaseModel):
    """Schema for file-level summaries."""

    file_path: str = Field(..., description="Path to the file")
    purpose: str = Field(..., description="Primary purpose of the file")
    key_functions: list[str] = Field(..., description="List of key functions/classes")
    dependencies: list[str] = Field(..., description="External dependencies used")
    complexity_score: int = Field(..., ge=1, le=10, description="Complexity score 1-10")
    lines_of_code: int = Field(..., description="Approximate lines of code")
    citations: list[dict[str, Any]] = Field(
        ..., description="Citations with line ranges"
    )


class ModuleSummary(BaseModel):
    """Schema for module-level summaries."""

    module_path: str = Field(..., description="Path to the module/directory")
    purpose: str = Field(..., description="Primary purpose of the module")
    key_files: list[str] = Field(..., description="List of key files in the module")
    architecture: str = Field(..., description="Architectural pattern or design")
    interfaces: list[str] = Field(..., description="Public interfaces/APIs")
    dependencies: list[str] = Field(..., description="Module dependencies")
    complexity_score: int = Field(
        ..., ge=1, le=10, description="Overall complexity score"
    )
    file_summaries: list[FileSummary] = Field(
        ..., description="Summaries of constituent files"
    )
    citations: list[dict[str, Any]] = Field(
        ..., description="Citations with file paths and line ranges"
    )


class RepositorySummary(BaseModel):
    """Schema for repository-level summaries."""

    repository_url: str = Field(..., description="Repository URL")
    purpose: str = Field(..., description="Primary purpose of the repository")
    technology_stack: list[str] = Field(
        ..., description="Technologies and frameworks used"
    )
    architecture: str = Field(..., description="Overall architecture description")
    key_modules: list[str] = Field(..., description="List of key modules/directories")
    entry_points: list[str] = Field(
        ..., description="Main entry points (main files, APIs)"
    )
    dependencies: list[str] = Field(..., description="External dependencies")
    complexity_score: int = Field(
        ..., ge=1, le=10, description="Overall complexity score"
    )
    module_summaries: list[ModuleSummary] = Field(
        ..., description="Summaries of constituent modules"
    )
    citations: list[dict[str, Any]] = Field(
        ..., description="Citations with file paths and line ranges"
    )


class LLMPipeline:
    """
    LLM Pipeline for multi-level repository summarization.

    Provides structured summarization from file → module → repository level
    with proper citations and JSON schema validation.
    """

    def __init__(self, llm_client: Any) -> None:
        """
        Initialize the LLM pipeline.

        Args:
            llm_client: LLM client for generating summaries
        """
        self.llm_client = llm_client

        # Prompt templates
        self.file_prompt_template = self._get_file_prompt_template()
        self.module_prompt_template = self._get_module_prompt_template()
        self.repository_prompt_template = self._get_repository_prompt_template()

    def process_repository(
        self, chunks: list[dict[str, Any]], repo_metadata: dict[str, Any]
    ) -> RepositorySummary:
        """
        Process repository and generate multi-level summaries.

        Args:
            chunks: List of code chunks
            repo_metadata: Repository metadata

        Returns:
            Complete repository summary
        """
        try:
            # Group chunks by file and module
            file_groups = self._group_chunks_by_file(chunks)
            module_groups = self._group_chunks_by_module(chunks)

            # Generate file summaries
            file_summaries = []
            for file_path, file_chunks in file_groups.items():
                file_summary = self._generate_file_summary(file_path, file_chunks)
                file_summaries.append(file_summary)

            # Generate module summaries
            module_summaries = []
            for module_path, module_chunks in module_groups.items():
                # Get file summaries for this module
                module_file_summaries = [
                    fs
                    for fs in file_summaries
                    if self._is_file_in_module(fs.file_path, module_path)
                ]

                module_summary = self._generate_module_summary(
                    module_path, module_chunks, module_file_summaries
                )
                module_summaries.append(module_summary)

            # Generate repository summary
            repository_summary = self._generate_repository_summary(
                repo_metadata, module_summaries
            )

            return repository_summary

        except Exception as e:
            logger.error(f"Repository processing failed: {e}")
            raise

    def _group_chunks_by_file(
        self, chunks: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group chunks by file path."""
        file_groups: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            file_path = chunk["metadata"]["file_path"]
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(chunk)
        return file_groups

    def _group_chunks_by_module(
        self, chunks: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group chunks by module (directory)."""
        module_groups: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            file_path = chunk["metadata"]["file_path"]
            module_path = self._determine_module_path(file_path)
            if module_path not in module_groups:
                module_groups[module_path] = []
            module_groups[module_path].append(chunk)
        return module_groups

    def _determine_module_path(self, file_path: str) -> str:
        """Determine module path from file path."""
        path = Path(file_path)
        if path.parent.name:
            return str(path.parent)
        return "root"

    def _is_file_in_module(self, file_path: str, module_path: str) -> bool:
        """Check if a file belongs to a module."""
        file_path_obj = Path(file_path)
        module_path_obj = Path(module_path)
        return (
            module_path_obj in file_path_obj.parents
            or str(file_path_obj.parent) == module_path
        )

    def _generate_file_summary(
        self, file_path: str, chunks: list[dict[str, Any]]
    ) -> FileSummary:
        """Generate file-level summary."""
        try:
            # Prepare file content and metadata
            file_content = "\n".join([chunk["content"] for chunk in chunks])
            lines_of_code = sum(
                chunk["metadata"]["end_line"] - chunk["metadata"]["start_line"] + 1
                for chunk in chunks
            )

            # Create context for LLM
            context = {
                "file_path": file_path,
                "content": file_content,
                "lines_of_code": lines_of_code,
                "chunks": [
                    {
                        "content": chunk["content"],
                        "start_line": chunk["metadata"]["start_line"],
                        "end_line": chunk["metadata"]["end_line"],
                        "node_type": chunk["metadata"]["node_type"],
                    }
                    for chunk in chunks
                ],
            }

            # Generate summary using LLM
            prompt = self.file_prompt_template.format(**context)
            response = self.llm_client._call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert code analyst. Generate detailed file summaries with proper citations.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            # Parse JSON response
            summary_data = json.loads(response)

            # Create FileSummary object
            return FileSummary(**summary_data)

        except Exception as e:
            logger.error(f"File summary generation failed for {file_path}: {e}")
            # Return minimal summary
            return FileSummary(
                file_path=file_path,
                purpose="Analysis failed",
                key_functions=[],
                dependencies=[],
                complexity_score=5,
                lines_of_code=0,
                citations=[],
            )

    def _generate_module_summary(
        self,
        module_path: str,
        chunks: list[dict[str, Any]],
        file_summaries: list[FileSummary],
    ) -> ModuleSummary:
        """Generate module-level summary."""
        try:
            # Prepare module context
            module_files = list({chunk["metadata"]["file_path"] for chunk in chunks})

            context = {
                "module_path": module_path,
                "files": module_files,
                "file_summaries": [fs.model_dump() for fs in file_summaries],
                "total_chunks": len(chunks),
            }

            # Generate summary using LLM
            prompt = self.module_prompt_template.format(**context)
            response = self.llm_client._call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert software architect. Generate detailed module summaries with proper citations.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            # Parse JSON response
            summary_data = json.loads(response)

            # Create ModuleSummary object
            return ModuleSummary(**summary_data)

        except Exception as e:
            logger.error(f"Module summary generation failed for {module_path}: {e}")
            # Return minimal summary
            return ModuleSummary(
                module_path=module_path,
                purpose="Analysis failed",
                key_files=[],
                architecture="Unknown",
                interfaces=[],
                dependencies=[],
                complexity_score=5,
                file_summaries=file_summaries,
                citations=[],
            )

    def _generate_repository_summary(
        self, repo_metadata: dict[str, Any], module_summaries: list[ModuleSummary]
    ) -> RepositorySummary:
        """Generate repository-level summary."""
        try:
            # Prepare repository context
            context = {
                "repository_url": repo_metadata.get("url", "unknown"),
                "module_summaries": [ms.dict() for ms in module_summaries],
                "total_modules": len(module_summaries),
            }

            # Generate summary using LLM
            prompt = self.repository_prompt_template.format(**context)
            response = self.llm_client._call_llm(
                [
                    {
                        "role": "system",
                        "content": "You are an expert software architect. Generate comprehensive repository summaries with proper citations.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )

            # Parse JSON response
            summary_data = json.loads(response)

            # Create RepositorySummary object
            return RepositorySummary(**summary_data)

        except Exception as e:
            logger.error(f"Repository summary generation failed: {e}")
            # Return minimal summary
            return RepositorySummary(
                repository_url=repo_metadata.get("url", "unknown"),
                purpose="Analysis failed",
                technology_stack=[],
                architecture="Unknown",
                key_modules=[],
                entry_points=[],
                dependencies=[],
                complexity_score=5,
                module_summaries=module_summaries,
                citations=[],
            )

    def _get_file_prompt_template(self) -> str:
        """Get file-level prompt template."""
        return """
Analyze the following file and generate a comprehensive summary.

File: {file_path}
Lines of Code: {lines_of_code}

File Content:
{content}

Code Chunks:
{chunks}

Please provide a JSON response with the following structure:
{{
    "file_path": "{file_path}",
    "purpose": "Brief description of the file's primary purpose",
    "key_functions": ["list", "of", "key", "functions", "or", "classes"],
    "dependencies": ["list", "of", "external", "dependencies"],
    "complexity_score": 5,
    "lines_of_code": {lines_of_code},
    "citations": [
        {{
            "type": "function",
            "name": "function_name",
            "start_line": 10,
            "end_line": 25,
            "description": "What this function does"
        }}
    ]
}}

IMPORTANT:
- Cite specific line ranges for all key functions and classes
- Be precise about the file's purpose and role
- Include all significant dependencies
- Provide accurate complexity assessment (1-10 scale)
"""

    def _get_module_prompt_template(self) -> str:
        """Get module-level prompt template."""
        return """
Analyze the following module and generate a comprehensive summary.

Module: {module_path}
Files: {files}
Total Chunks: {total_chunks}

File Summaries:
{file_summaries}

Please provide a JSON response with the following structure:
{{
    "module_path": "{module_path}",
    "purpose": "Brief description of the module's primary purpose",
    "key_files": ["list", "of", "key", "files"],
    "architecture": "Description of the architectural pattern or design",
    "interfaces": ["list", "of", "public", "interfaces", "or", "APIs"],
    "dependencies": ["list", "of", "module", "dependencies"],
    "complexity_score": 5,
    "file_summaries": {file_summaries},
    "citations": [
        {{
            "type": "file",
            "file_path": "path/to/file.py",
            "start_line": 1,
            "end_line": 50,
            "description": "What this file contributes to the module"
        }}
    ]
}}

IMPORTANT:
- Cite specific files and their contributions
- Describe the module's architectural role
- Identify public interfaces and APIs
- Assess overall complexity
"""

    def _get_repository_prompt_template(self) -> str:
        """Get repository-level prompt template."""
        return """
Analyze the following repository and generate a comprehensive summary.

Repository: {repository_url}
Modules: {total_modules}

Module Summaries:
{module_summaries}

Please provide a JSON response with the following structure:
{{
    "repository_url": "{repository_url}",
    "purpose": "Brief description of the repository's primary purpose",
    "technology_stack": ["list", "of", "technologies", "and", "frameworks"],
    "architecture": "Description of the overall architecture",
    "key_modules": ["list", "of", "key", "modules", "or", "directories"],
    "entry_points": ["list", "of", "main", "entry", "points"],
    "dependencies": ["list", "of", "external", "dependencies"],
    "complexity_score": 5,
    "module_summaries": {module_summaries},
    "citations": [
        {{
            "type": "module",
            "module_path": "src/main",
            "description": "What this module contributes to the repository"
        }}
    ]
}}

IMPORTANT:
- Provide a comprehensive overview of the repository
- Identify the technology stack and architecture
- List key modules and their roles
- Identify main entry points and APIs
- Assess overall complexity and dependencies
"""

    def export_summary(self, summary: RepositorySummary, format: str = "json") -> str:
        """
        Export summary in various formats.

        Args:
            summary: Repository summary to export
            format: Export format ("json", "markdown", "yaml")

        Returns:
            Exported summary as string
        """
        if format == "json":
            return str(summary.json(indent=2))
        elif format == "markdown":
            return self._export_markdown(summary)
        elif format == "yaml":
            return self._export_yaml(summary)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_markdown(self, summary: RepositorySummary) -> str:
        """Export summary as Markdown."""
        md = f"""# Repository Summary

## Overview
- **Repository**: {summary.repository_url}
- **Purpose**: {summary.purpose}
- **Architecture**: {summary.architecture}
- **Complexity Score**: {summary.complexity_score}/10

## Technology Stack
{chr(10).join(f"- {tech}" for tech in summary.technology_stack)}

## Key Modules
{chr(10).join(f"- {module}" for module in summary.key_modules)}

## Entry Points
{chr(10).join(f"- {entry}" for entry in summary.entry_points)}

## Dependencies
{chr(10).join(f"- {dep}" for dep in summary.dependencies)}

## Module Details
"""

        for module in summary.module_summaries:
            md += f"\n### {module.module_path}\n"
            md += f"- **Purpose**: {module.purpose}\n"
            md += f"- **Architecture**: {module.architecture}\n"
            md += f"- **Complexity**: {module.complexity_score}/10\n"
            md += f"- **Key Files**: {', '.join(module.key_files)}\n"

        return md

    def _export_yaml(self, summary: RepositorySummary) -> str:
        """Export summary as YAML."""
        return str(yaml.dump(summary.dict(), default_flow_style=False, indent=2))


# Example usage and testing
def demo_llm_pipeline() -> None:
    """Demonstrate the LLM pipeline."""

    # Mock LLM client
    class MockLLMClient:
        def _call_llm(self, messages: list[dict[str, Any]]) -> str:
            # Mock response for demo
            return json.dumps(
                {
                    "file_path": "src/main.py",
                    "purpose": "Main application entry point",
                    "key_functions": ["main", "hello_world"],
                    "dependencies": ["requests", "flask"],
                    "complexity_score": 3,
                    "lines_of_code": 50,
                    "citations": [
                        {
                            "type": "function",
                            "name": "main",
                            "start_line": 10,
                            "end_line": 25,
                            "description": "Main application function",
                        }
                    ],
                }
            )

    # Initialize pipeline
    llm_client = MockLLMClient()
    pipeline = LLMPipeline(llm_client)

    # Sample chunks
    sample_chunks = [
        {
            "content": "def main():\n    print('Hello, World!')",
            "metadata": {
                "file_path": "src/main.py",
                "start_line": 1,
                "end_line": 3,
                "node_type": "function_definition",
            },
        }
    ]

    repo_metadata = {"url": "https://github.com/example/repo.git"}

    # Process repository
    try:
        summary = pipeline.process_repository(sample_chunks, repo_metadata)

        print("LLM Pipeline Demo Results:")
        print(f"Repository: {summary.repository_url}")
        print(f"Purpose: {summary.purpose}")
        print(f"Technology Stack: {summary.technology_stack}")

        # Export in different formats
        json_export = pipeline.export_summary(summary, "json")
        print(f"\nJSON Export (first 200 chars): {json_export[:200]}...")

    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    demo_llm_pipeline()
