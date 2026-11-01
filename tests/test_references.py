from panorama.protocols import LEGACY, SUPERPOINT
from panorama.references import comparison_rows


def test_paper_values_are_not_measured_or_added_to_partial_runs():
    measured = [{"dataset": "HPatches", "method": "sift-plain", "pairs": 580, "succeeded": 579}]
    protocol = {"protocol_id": SUPERPOINT, "is_subset": False, "pairs": 580}
    rows = comparison_rows(measured, protocol)
    assert len(rows) == 4
    assert rows[0]["source"] == "本地实测" and rows[0]["method"] == "SIFT"
    assert rows[2]["source"] == "论文报告" and rows[2]["h_accuracy_3px"] == 0.676
    assert "succeeded" not in rows[2] and "mean_seconds" not in rows[2]
    assert "source" not in measured[0]
    for other in [dict(protocol, is_subset=True), dict(protocol, protocol_id=LEGACY)]:
        assert len(comparison_rows(measured, other)) == 1
