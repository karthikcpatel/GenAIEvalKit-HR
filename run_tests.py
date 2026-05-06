import os
import shutil
import subprocess
import sys

import pytest

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

RESULTS_DIR = "reports/allure-results"

# Exit code descriptions from pytest docs
EXIT_CODE_SUMMARY = {
    0: "✓  All tests passed",
    1: "✗  Some tests failed",
    2: "✗  Test execution was interrupted",
    3: "✗  Internal pytest error",
    4: "✗  pytest command line usage error",
    5: "✓  No tests collected"
}


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def resolve_marker():
    """
    Resolve the pytest marker expression from the command line.

    Usage:
        python run_tests.py                   # default: all except performance
        python run_tests.py integration       # only integration tests
        python run_tests.py regression        # only regression tests
        python run_tests.py performance       # only performance tests
        python run_tests.py "not performance" # explicit default

    Returns:
        str: A pytest -m marker expression.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "not performance"


def should_clean():
    """
    Determine whether to wipe the Allure results directory before the run.

    Controlled by the CLEAN_ALLURE environment variable.
    Defaults to True so fresh runs don't mix old and new results.
    Set CLEAN_ALLURE=false to preserve history for trend comparison.

    Returns:
        bool
    """
    return os.environ.get("CLEAN_ALLURE", "true").strip().lower() != "false"


def serve_allure_report():
    """
    Launch the Allure report server if the Allure CLI is available.

    Prints a clear message if Allure is not installed so the user
    knows where the raw results are saved.
    """
    if shutil.which("allure") is None:
        print(
            "\n[run_tests] Allure CLI not found on PATH.\n"
            "            Install it to view the interactive report:\n"
            "            https://allurereport.org/docs/install/\n"
            f"            Raw results saved to: {RESULTS_DIR}/"
        )
        return

    print(f"\n[run_tests] Launching Allure report from {RESULTS_DIR}/ ...")
    subprocess.run(
        ["allure", "serve", RESULTS_DIR],
        check=False
    )


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":

    marker = resolve_marker()
    clean  = should_clean()

    print(f"[run_tests] Marker filter : {marker}")
    print(f"[run_tests] Clean results : {clean}")
    print(f"[run_tests] Results dir   : {RESULTS_DIR}/\n")

    # ---- Ensure results directory exists ------------------------- #
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Build pytest argument list ------------------------------ #
    args = [
        "tests/",               # explicit test path — not CWD-dependent
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-m", marker,
        f"--alluredir={RESULTS_DIR}",
    ]

    if clean:
        args.append("--clean-alluredir")

    # ---- Run tests ----------------------------------------------- #
    exit_code = pytest.main(args)

    # ---- Print summary ------------------------------------------- #
    description = EXIT_CODE_SUMMARY.get(exit_code, f"Unknown exit code: {exit_code}")
    print(f"\n[run_tests] Result: {description}  (exit code {exit_code})")

    # ---- Serve Allure report ------------------------------------- #
    serve_allure_report()

    # ---- Propagate exit code to the calling process -------------- #
    # This ensures CI/CD pipelines correctly detect test failures.
    sys.exit(exit_code)
