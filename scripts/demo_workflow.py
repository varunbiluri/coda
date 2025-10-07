#!/usr/bin/env python3
"""
Coda Multi-Agent Workflow Demonstration

This script demonstrates the complete Coda multi-agent workflow for automated
code generation and testing. It showcases how four specialized AI agents work
together to transform failing tests into working code.

Use Case: Automated API endpoint development with comprehensive testing.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import requests


def check_server_health(url: str) -> bool:
    """Check if the Coda server is running and healthy."""
    try:
        response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def display_workflow_status(result: dict[str, Any]) -> None:
    """Display the workflow execution status in a professional format."""
    print("\n" + "=" * 60)
    print("WORKFLOW EXECUTION SUMMARY")
    print("=" * 60)

    # Planner results
    if result.get("planner_spec"):
        planner = result["planner_spec"]
        tasks = planner.get("tasks", [])
        print(f"Planner Agent: {len(tasks)} tasks identified")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task.get('description', 'No description')}")

    # Coder results
    if result.get("coder_output"):
        coder = result["coder_output"]
        diff_length = len(coder.get("diff", ""))
        print(f"Coder Agent: Generated {diff_length} character code diff")
        print(f"  Commit Message: {coder.get('commit_message', 'N/A')}")

        # Display the generated diff
        if coder.get("diff"):
            print("\n  Generated Code Changes:")
            print("  " + "-" * 50)
            diff_lines = coder["diff"].split("\n")
            for line in diff_lines[:20]:  # Show first 20 lines
                if line.strip():
                    print(f"  {line}")
            if len(diff_lines) > 20:
                print(f"  ... ({len(diff_lines) - 20} more lines)")
            print("  " + "-" * 50)

    # ApplyPatch results
    if result.get("apply_patch_output"):
        patch = result["apply_patch_output"]
        status = "SUCCESS" if patch.get("success") else "FAILED"
        print(f"ApplyPatch Agent: {status}")
        if patch.get("commit_hash"):
            print(f"  Commit Hash: {patch['commit_hash']}")

    # Show final applied code if successful
    if result.get("apply_patch_output", {}).get("success"):
        print("\n  Applied Code Changes:")
        print("  " + "-" * 50)
        try:
            # Try to read the modified files to show what was actually applied
            from pathlib import Path

            # Look for the workspace directory in the runs folder
            run_id = result.get("run_id", "")
            if run_id:
                workspace_path = Path(f".runs/{run_id}/workspace")
                if workspace_path.exists():
                    main_py_path = workspace_path / "app" / "main.py"
                    if main_py_path.exists():
                        with open(main_py_path) as f:
                            content = f.read()

                        # Show the relevant parts of the file
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if (
                                "/health" in line
                                or "def read_health" in line
                                or 'return {"status"' in line
                            ):
                                # Show context around the health endpoint
                                start = max(0, i - 2)
                                end = min(len(lines), i + 5)
                                for j in range(start, end):
                                    marker = ">>> " if j == i else "    "
                                    print(f"  {marker}{lines[j]}")
                                break
        except Exception as e:
            print(f"  Could not display applied code: {e}")
        print("  " + "-" * 50)

    # Tester results
    if result.get("tester_output"):
        tester = result["tester_output"]
        status = "PASSED" if tester.get("success") else "FAILED"
        print(f"Tester Agent: Tests {status}")
        if tester.get("test_output"):
            print(f"  Test Output: {tester['test_output'][:100]}...")


def main():
    """Execute the Coda multi-agent workflow demonstration."""
    print("Coda Multi-Agent Workflow Demonstration")
    print("=" * 50)
    print("Use Case: Automated API endpoint development")
    print("=" * 50)

    # Configuration
    coda_url = "http://localhost:8000"
    repo_path = str(Path(__file__).parent.parent / "examples" / "sample_service")
    goal = "Add /health endpoint with proper error handling and validation"

    print(f"Repository: {repo_path}")
    print(f"Objective: {goal}")
    print()

    # Verify server availability
    print("Checking Coda server availability...")
    if not check_server_health(coda_url):
        print("ERROR: Coda server is not running or not responding")
        print("Please start the server with: python main.py")
        sys.exit(1)

    print("SUCCESS: Coda server is running")
    print()

    # Prepare workflow request
    run_request = {"goal": goal, "repo_path": repo_path, "branch": "main"}

    print("Initiating multi-agent workflow...")
    start_time = time.time()

    try:
        # Execute workflow
        response = requests.post(
            f"{coda_url}/runs", json=run_request, timeout=300  # 5 minutes timeout
        )

        if response.status_code not in [200, 201]:
            print(f"ERROR: Workflow request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)

        result = response.json()
        execution_time = time.time() - start_time

        print(f"SUCCESS: Workflow completed in {execution_time:.2f} seconds")
        print(f"Run ID: {result['run_id']}")
        print(f"Status: {result['status'].upper()}")

        # Display detailed results
        result_path = Path(result["result_path"])
        if result_path.exists():
            with open(result_path) as f:
                detailed_result = json.load(f)
            display_workflow_status(detailed_result)

        # Final status
        print("\n" + "=" * 60)
        if result["status"] == "success":
            print("WORKFLOW COMPLETED SUCCESSFULLY")
            print("All agents executed successfully and tests are passing.")
        else:
            print("WORKFLOW COMPLETED WITH ISSUES")
            print("Some agents encountered errors. Check logs for details.")
        print("=" * 60)

    except requests.exceptions.Timeout:
        print("ERROR: Workflow execution timed out")
        print("The operation may still be running in the background.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Workflow request failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
