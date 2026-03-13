import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(step_name: str, script_path: str) -> bool:
    """
    Run a Python script as a pipeline step.
    Returns True if successful, False otherwise.
    """
    full_path = PROJECT_ROOT / script_path

    print(f"\n=== Running: {step_name} ===")
    print(f"Script: {full_path}")

    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        duration = time.time() - start

        if result.stdout:
            print(result.stdout.strip())

        if result.stderr:
            print("STDERR:")
            print(result.stderr.strip())

        if result.returncode != 0:
            print(f"[FAIL] {step_name} failed in {duration:.2f}s")
            return False

        print(f"[OK] {step_name} completed in {duration:.2f}s")
        return True

    except Exception as e:
        duration = time.time() - start
        print(f"[ERROR] {step_name} crashed in {duration:.2f}s: {e}")
        return False


def run_pipeline(pipeline_name: str, steps: list[tuple[str, str]]) -> int:
    """
    Run a list of pipeline steps in order.
    Returns exit code 0 if all succeed, 1 if any fail.
    """
    print(f"\n==============================")
    print(f"Starting pipeline: {pipeline_name}")
    print(f"==============================")

    failures = []

    for step_name, script_path in steps:
        success = run_step(step_name, script_path)
        if not success:
            failures.append(step_name)

    print(f"\n==============================")
    print(f"Finished pipeline: {pipeline_name}")
    print(f"==============================")

    if failures:
        print("Failed steps:")
        for step in failures:
            print(f" - {step}")
        return 1

    print("All steps completed successfully.")
    return 0