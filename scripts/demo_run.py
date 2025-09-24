#!/usr/bin/env python3
"""Demo script to run the complete Coda flow."""

import json
import sys
from pathlib import Path

import requests

# Add the src directory to the path so we can import coda
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    """Run the multi-agent demo flow."""
    print("Starting Coda Multi-Agent Demo")
    print("=" * 60)
    print("This demo showcases four specialized agents working together:")
    print("   Planner Agent - Analyzes goals and creates plans")
    print("   Coder Agent - Generates code changes")
    print("   Apply Patch Agent - Applies changes to git")
    print("   Tester Agent - Runs tests in Docker sandbox")
    print("=" * 60)

    # Configuration
    coda_url = "http://localhost:8000"
    repo_path = str(Path(__file__).parent.parent / "examples" / "sample_service")
    goal = "Add /health endpoint"

    print(f"Repository: {repo_path}")
    print(f"Goal: {goal}")
    print(f"Coda URL: {coda_url}")

    # Check if Coda server is running
    try:
        response = requests.get(f"{coda_url}/health", timeout=5)
        if response.status_code != 200:
            print("ERROR: Coda server is not responding correctly")
            sys.exit(1)
        print("SUCCESS: Coda server is running")
    except requests.exceptions.RequestException:
        print("ERROR: Cannot connect to Coda server. Make sure it's running on port 8000")
        print("   Run: python main.py")
        sys.exit(1)

    # Create run request
    run_request = {"goal": goal, "repo_path": repo_path, "branch": "main"}

    print("\nSending run request...")
    print(f"Request: {json.dumps(run_request, indent=2)}")

    # Send request
    try:
        response = requests.post(
            f"{coda_url}/runs",
            json=run_request,
            timeout=300,  # 5 minutes timeout
        )

        if response.status_code not in [200, 201]:
            print(f"ERROR: Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            sys.exit(1)

        result = response.json()
        print("SUCCESS: Run created successfully!")
        print(f"Run ID: {result['run_id']}")
        print(f"Status: {result['status']}")
        print(f"Result file: {result['result_path']}")

        # Read and display the detailed result
        result_path = Path(result["result_path"])
        if result_path.exists():
            print("\nDetailed Results:")
            print("=" * 50)

            with open(result_path) as f:
                detailed_result = json.load(f)

            print(f"Status: {detailed_result['status']}")

            if detailed_result.get("error_message"):
                print(f"Error: {detailed_result['error_message']}")

            # Show planner spec
            if detailed_result.get("planner_spec"):
                print("\nPlanner Specification:")
                planner = detailed_result["planner_spec"]
                print(f"  Context: {planner['context']}")
                print(f"  Tasks: {len(planner['tasks'])}")
                for i, task in enumerate(planner["tasks"], 1):
                    print(f"    {i}. {task['description']}")

            # Show coder output
            if detailed_result.get("coder_output"):
                print("\nCoder Output:")
                coder = detailed_result["coder_output"]
                print(f"  Commit Message: {coder['commit_message']}")
                print(f"  Explanation: {coder['explanation']}")
                print(f"  Diff Length: {len(coder['diff'])} characters")

            # Show apply patch result
            if detailed_result.get("apply_patch_output"):
                print("\nApply Patch Result:")
                patch = detailed_result["apply_patch_output"]
                print(f"  Success: {patch['success']}")
                print(f"  Branch: {patch['branch_name']}")
                print(
                    f"  Commit Hash: {patch['commit_hash'][:8]}..."
                    if patch["commit_hash"]
                    else "None"
                )
                if patch.get("error_message"):
                    print(f"  Error: {patch['error_message']}")

            # Show test results
            if detailed_result.get("tester_output"):
                print("\nTest Results:")
                tester = detailed_result["tester_output"]
                print(f"  Success: {tester['success']}")
                print(f"  Exit Code: {tester['exit_code']}")
                if tester["stdout"]:
                    print(f"  Stdout: {tester['stdout']}")
                if tester["stderr"]:
                    print(f"  Stderr: {tester['stderr']}")

        print("\nDemo completed!")

        if result["status"] == "success":
            print("SUCCESS: All steps completed successfully!")
        else:
            print("WARNING: Some steps failed - check the logs above")

    except requests.exceptions.Timeout:
        print("ERROR: Request timed out - the operation may still be running")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
