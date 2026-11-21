
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src" / "monitoring")
)

from decision_engine import make_feature_based_decision


test_cases = [
    (
        "Healthy",
        {
            "avg_temp": {
                "severity": "No Drift",
                "ks_pvalue": 0.8,
                "evidently_drift": False
            }
        },
        "Healthy"
    ),
    (
        "Low Drift",
        {
            "avg_temp": {
                "severity": "Low Drift",
                "ks_pvalue": 0.02,
                "evidently_drift": True
            }
        },
        "Monitor"
    ),
    (
        "Moderate Drift",
        {
            "avg_temp": {
                "severity": "Moderate Drift",
                "ks_pvalue": 0.01,
                "evidently_drift": True
            }
        },
        "Investigate"
    ),
    (
        "High Drift",
        {
            "avg_temp": {
                "severity": "High Drift",
                "ks_pvalue": 0.001,
                "evidently_drift": True
            }
        },
        "Retrain"
    ),
    (
        "Conflicting Evidence",
        {
            "avg_temp": {
                "severity": "High Drift",
                "ks_pvalue": 0.6,
                "evidently_drift": False
            }
        },
        "Manual Review"
    )
]


for name, features, expected in test_cases:

    result = make_feature_based_decision(features)

    actual = result["decision"]

    print(
        f"{name}: Expected={expected}, Actual={actual}"
    )

    assert actual == expected, (
        f"{name} failed: expected {expected}, got {actual}"
    )


print("\nAll feature-based Decision Engine tests passed!")


# Additional edge-case tests

invalid_cases = [
    (
        "Empty feature results",
        {}
    ),
    (
        "Invalid KS p-value",
        {
            "avg_temp": {
                "severity": "High Drift",
                "ks_pvalue": 1.5,
                "evidently_drift": True
            }
        }
    ),
    (
        "Invalid drift severity",
        {
            "avg_temp": {
                "severity": "Extreme Drift",
                "ks_pvalue": 0.01,
                "evidently_drift": True
            }
        }
    ),
    (
        "Invalid Evidently flag",
        {
            "avg_temp": {
                "severity": "Low Drift",
                "ks_pvalue": 0.02,
                "evidently_drift": "Yes"
            }
        }
    )
]

for name, features in invalid_cases:
    try:
        make_feature_based_decision(features)
    except ValueError:
        print(f"{name}: PASS")
    else:
        raise AssertionError(
            f"{name}: Expected ValueError"
        )

print("\nAll edge-case tests passed!")

