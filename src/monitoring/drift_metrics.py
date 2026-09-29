import numpy as np

from scipy.stats import ks_2samp


def calculate_psi(
    expected,
    actual,
    bins=10
):

    expected = np.array(expected)

    actual = np.array(actual)

    breakpoints = np.linspace(
        0,
        100,
        bins + 1
    )

    breakpoints = np.percentile(
        expected,
        breakpoints
    )

    expected_counts = np.histogram(
        expected,
        bins=breakpoints
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=breakpoints
    )[0]

    expected_perc = expected_counts / len(expected)

    actual_perc = actual_counts / len(actual)

    expected_perc = np.where(
        expected_perc == 0,
        0.0001,
        expected_perc
    )

    actual_perc = np.where(
        actual_perc == 0,
        0.0001,
        actual_perc
    )

    psi = np.sum(

        (
            expected_perc -
            actual_perc
        )

        *

        np.log(
            expected_perc /
            actual_perc
        )

    )

    return psi


def calculate_ks_test(
    expected,
    actual
):

    statistic, p_value = ks_2samp(
        expected,
        actual
    )

    return statistic, p_value


def get_drift_severity(psi):

    if psi < 0.10:

        return "No Drift"

    elif psi < 0.20:

        return "Low Drift"

    elif psi < 0.30:

        return "Moderate Drift"

    else:

        return "High Drift"


def get_recommendation(
    psi,
    p_value
):

    severity = get_drift_severity(
        psi
    )

    if severity == "No Drift" and p_value >= 0.05:

        return "No Action Required"

    elif severity == "Low Drift":

        return "Monitor Feature"

    elif severity == "Moderate Drift":

        return "Investigate Feature"

    elif severity == "High Drift" or p_value < 0.05:

        return "Retrain Recommended"

    return "Monitor Feature"