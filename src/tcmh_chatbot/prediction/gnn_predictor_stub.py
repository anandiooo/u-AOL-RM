from __future__ import annotations

from typing import Any


class GNNRiskPredictorStub:
    """Stub module to mark where GNN-based early warning can be integrated."""

    def fit(self, graph_sequences: Any, labels: Any) -> None:
        raise NotImplementedError(
            "GNN predictor is intentionally left as a research extension. "
            "Implement training with PyTorch Geometric or DGL as needed."
        )

    def predict(self, graph_sequence: Any) -> float:
        raise NotImplementedError(
            "GNN predictor is intentionally left as a research extension."
        )
