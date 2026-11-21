def make_decision(
    psi_status,
    ks_pvalue,
    evidently_dataset_drift,
    drifted_feature_percentage
):
    valid_statuses = {
        "Healthy",
        "Low Drift",
        "Moderate Drift",
        "High Drift"
    }

    if psi_status not in valid_statuses:
        raise ValueError("Unknown PSI drift status")

    if not 0 <= ks_pvalue <= 1:
        raise ValueError("Invalid KS p-value")

    if not 0 <= drifted_feature_percentage <= 100:
        raise ValueError("Invalid Evidently drift percentage")

    ks_significant = ks_pvalue < 0.05
    psi_drift = psi_status != "Healthy"
    evidently_drift = drifted_feature_percentage > 0

    # Dataset-level flag and feature-level share
    # must describe the same Evidently report.
    if evidently_dataset_drift and not evidently_drift:
        raise ValueError("Inconsistent Evidently results")

    # All monitoring methods agree on no drift.
    if (
        not psi_drift
        and not ks_significant
        and not evidently_drift
    ):
        return {
            "decision": "Healthy",
            "confidence": "High",
            "action": "No Action Required"
        }

    # High drift with corroborating evidence.
    if (
        psi_status == "High Drift"
        and ks_significant
        and evidently_drift
    ):
        return {
            "decision": "Retrain",
            "confidence": "High",
            "action": "Evaluate Model for Retraining"
        }

    # Moderate drift with corroborating evidence.
    if (
        psi_status == "Moderate Drift"
        and ks_significant
        and evidently_drift
    ):
        return {
            "decision": "Investigate",
            "confidence": "High",
            "action": "Investigate Drifted Features"
        }

    # Low drift with corroborating evidence.
    if (
        psi_status == "Low Drift"
        and ks_significant
        and evidently_drift
    ):
        return {
            "decision": "Monitor",
            "confidence": "Medium",
            "action": "Monitor Feature Distribution"
        }

    # Conflicting monitoring signals.
    return {
        "decision": "Manual Review",
        "confidence": "Low",
        "action": "Review Conflicting Drift Results"
    }


def make_feature_based_decision(feature_results):
    """
    Make decisions using feature-level PSI and KS results.

    feature_results example:
    {
        "avg_temp": {
            "severity": "High Drift",
            "ks_pvalue": 0.01,
            "evidently_drift": True
        }
    }
    """

    if not feature_results:
        raise ValueError("No feature results provided")

    valid_severities = {
        "No Drift",
        "Low Drift",
        "Moderate Drift",
        "High Drift"
    }

    confirmed = []
    conflicts = []
    stable = []

    for feature, metrics in feature_results.items():

        severity = metrics["severity"]
        pvalue = float(metrics["ks_pvalue"])
        evidently_drift = metrics["evidently_drift"]

        if severity not in valid_severities:
            raise ValueError(
                f"Invalid severity for {feature}"
            )

        if not 0 <= pvalue <= 1:
            raise ValueError(
                f"Invalid KS p-value for {feature}"
            )

        if not isinstance(evidently_drift, bool):
            raise ValueError(
                f"Invalid Evidently flag for {feature}"
            )

        psi_drift = severity != "No Drift"
        ks_drift = pvalue < 0.05

        # All three signals indicate drift
        if psi_drift and ks_drift and evidently_drift:
            confirmed.append({
                "feature": feature,
                "severity": severity
            })

        # All three signals indicate no drift
        elif (
            not psi_drift
            and not ks_drift
            and not evidently_drift
        ):
            stable.append(feature)

        # Monitoring signals disagree
        else:
            conflicts.append(feature)

    confirmed_severities = {
        item["severity"] for item in confirmed
    }

    if "High Drift" in confirmed_severities:
        decision = "Retrain"
        confidence = "High"
        action = "Evaluate Model for Retraining"

    elif "Moderate Drift" in confirmed_severities:
        decision = "Investigate"
        confidence = "High"
        action = "Investigate Drifted Features"

    elif "Low Drift" in confirmed_severities:
        decision = "Monitor"
        confidence = "Medium"
        action = "Monitor Feature Distribution"

    elif conflicts:
        decision = "Manual Review"
        confidence = "Low"
        action = "Review Conflicting Drift Results"

    else:
        decision = "Healthy"
        confidence = "High"
        action = "No Action Required"

    return {
        "decision": decision,
        "confidence": confidence,
        "action": action,
        "confirmed_drift_features": confirmed,
        "conflicting_features": conflicts,
        "stable_features": stable
    }
