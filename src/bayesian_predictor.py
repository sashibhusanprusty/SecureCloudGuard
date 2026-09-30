from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


def _build_model() -> DiscreteBayesianNetwork:
    model = DiscreteBayesianNetwork(
        [
            ("HighTrafficRate", "HighPacketRate"),
            ("HighPacketRate", "PeriodicConnection"),
            ("PeriodicConnection", "DDoSOccurrence"),
        ]
    )
    model.add_cpds(
        TabularCPD(
            variable="HighTrafficRate",
            variable_card=2,
            values=[[0.82], [0.18]],
            state_names={"HighTrafficRate": [False, True]},
        ),
        TabularCPD(
            variable="HighPacketRate",
            variable_card=2,
            values=[[0.90, 0.15], [0.10, 0.85]],
            evidence=["HighTrafficRate"],
            evidence_card=[2],
            state_names={
                "HighPacketRate": [False, True],
                "HighTrafficRate": [False, True],
            },
        ),
        TabularCPD(
            variable="PeriodicConnection",
            variable_card=2,
            values=[[0.90, 0.25], [0.10, 0.75]],
            evidence=["HighPacketRate"],
            evidence_card=[2],
            state_names={
                "PeriodicConnection": [False, True],
                "HighPacketRate": [False, True],
            },
        ),
        TabularCPD(
            variable="DDoSOccurrence",
            variable_card=2,
            values=[[0.98, 0.18], [0.02, 0.82]],
            evidence=["PeriodicConnection"],
            evidence_card=[2],
            state_names={
                "DDoSOccurrence": [False, True],
                "PeriodicConnection": [False, True],
            },
        ),
    )
    if not model.check_model():
        raise ValueError("Bayesian network CPDs are inconsistent")
    return model


def predict_attack_probability(evidence: Mapping[str, bool]) -> float:
    """Return P(DDoSOccurrence=True | evidence) from illustrative manual CPDs."""
    variables = {
        "HighTrafficRate",
        "HighPacketRate",
        "PeriodicConnection",
        "DDoSOccurrence",
    }
    unknown = set(evidence) - variables
    if unknown:
        raise ValueError(f"Unknown evidence variable(s): {', '.join(sorted(unknown))}")
    if any(not isinstance(value, bool) for value in evidence.values()):
        raise ValueError("Evidence values must be booleans")

    inference = VariableElimination(_build_model())
    posterior = inference.query(
        variables=["DDoSOccurrence"],
        evidence=dict(evidence),
        show_progress=False,
    )
    return float(posterior.values[1])


def evidence_from_traffic_row(
    traffic_row: Mapping[str, object],
    reference_data: pd.DataFrame,
) -> dict[str, bool]:
    """Map observed CICIDS flow rates and inter-arrival timing to network evidence."""
    row = {str(name).strip(): value for name, value in traffic_row.items()}
    if "Label" not in reference_data:
        return {}

    benign = reference_data[
        reference_data["Label"].astype(str).str.upper().eq("BENIGN")
    ]
    evidence: dict[str, bool] = {}
    for variable, feature in (
        ("HighTrafficRate", "Flow Bytes/s"),
        ("HighPacketRate", "Flow Packets/s"),
    ):
        if feature not in row or feature not in benign:
            continue
        try:
            observed = float(row[feature])
            threshold = float(benign[feature].quantile(0.95))
        except (TypeError, ValueError):
            continue
        if math.isfinite(observed) and math.isfinite(threshold):
            evidence[variable] = observed > threshold

    try:
        iat_mean = float(row["Flow IAT Mean"])
        iat_std = float(row["Flow IAT Std"])
    except (KeyError, TypeError, ValueError):
        return evidence
    if math.isfinite(iat_mean) and iat_mean > 0 and math.isfinite(iat_std) and iat_std >= 0:
        evidence["PeriodicConnection"] = iat_std / iat_mean <= 0.5

    return evidence


if __name__ == "__main__":
    sample_evidence = {"HighTrafficRate": True, "HighPacketRate": True}
    probability = predict_attack_probability(sample_evidence)
    print(f"Sample evidence: {sample_evidence}")
    print(f"Predicted DDoS probability: {probability:.1%}")