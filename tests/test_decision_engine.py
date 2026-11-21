import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "monitoring")
)

from decision_engine import make_decision


test_cases = [
    ("Healthy", 0.14, False, 0, "Healthy"),
    ("Low Drift", 0.02, False, 20, "Monitor"),
    ("Moderate Drift", 0.01, False, 40, "Investigate"),
    ("High Drift", 0.001, True, 70, "Retrain"),
    ("High Drift", 0.60, False, 0, "Manual Review"),
]

for status, pvalue, evidently, percentage, expected in test_cases:

    result = make_decision(
        status,
        pvalue,
        evidently,
        percentage
    )

    actual = result["decision"]

    print(
        f"Expected: {expected} | Actual: {actual}"
    )

    assert actual == expected, (
        f"Test failed: expected {expected}, got {actual}"
    )

print("\nAll Decision Engine tests passed!")