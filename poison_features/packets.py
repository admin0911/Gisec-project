"""Feature adapter for packet/flow records, not raw payload inspection."""

from typing import Any

import numpy as np

from .bundle import FeatureBundle
from .preprocessing import prepare_representations

PACKET_FIELDS = (
    "duration", "packet_count", "byte_count", "src_port", "dst_port",
    "protocol", "mean_packet_length", "std_packet_length",
    "mean_interarrival", "tcp_syn", "tcp_ack", "tcp_fin",
)


def extract_packet_features(
    records: list[dict[str, Any]],
    *,
    labels: Any = None,
    sample_ids: Any = None,
    dataset_name: str = "network_flows",
) -> FeatureBundle:
    """Convert structured packet/flow records to detector-ready features.

    Records should describe one packet or aggregated flow per row. Missing
    numeric fields are represented as zero; categorical protocol values are
    mapped deterministically.
    """
    protocol_map: dict[str, int] = {}
    rows: list[list[float]] = []
    for record in records:
        values: list[float] = []
        for field in PACKET_FIELDS:
            value = record.get(field, 0)
            if field == "protocol":
                key = str(value).lower()
                if key not in protocol_map:
                    protocol_map[key] = len(protocol_map) + 1
                value = protocol_map[key]
            try:
                values.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Packet field '{field}' must be numeric") from exc
        rows.append(values)
    if not rows:
        raise ValueError("records must not be empty")
    features = np.asarray(rows, dtype=np.float32)
    scaled, reduced = prepare_representations(features)
    return FeatureBundle(
        features=features,
        scaled_features=scaled,
        reduced_features=reduced,
        labels=np.full(len(records), -1) if labels is None else np.asarray(labels),
        sample_ids=np.arange(len(records)) if sample_ids is None else np.asarray(sample_ids),
        modality="network",
        encoder="engineered_flow_features",
        dataset_name=dataset_name,
        metadata={"fields": list(PACKET_FIELDS), "protocol_map": protocol_map},
    )
