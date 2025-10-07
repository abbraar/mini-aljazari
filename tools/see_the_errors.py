
import traceback
import sys

def pytest_exception_interact(node, call, report):
    if report.failed:
        print(f"Error in {node.nodeid}: {report.longrepr}")

def report_error(exc: Exception) -> None:
    """Print exception details (used by tests)."""
    print("Reported error:", exc)
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stdout)