"""scripts/hospital_inbox.py: one hospital's view of the shared predictions topic."""

from types import SimpleNamespace as Record

from scripts.hospital_inbox import follow_ups


def test_a_hospital_sees_only_its_own_high_risk_patients():
    records = [
        Record(key=b"rural-60plus", value={"high_risk": True, "patient_id": "a"}),
        Record(key=b"rural-60plus", value={"high_risk": False, "patient_id": "b"}),
        Record(key=b"urban-60plus", value={"high_risk": True, "patient_id": "c"}),
        Record(key=None, value={"high_risk": True, "patient_id": "d"}),  # keyless: ignored
    ]
    assert [p["patient_id"] for p in follow_ups("rural-60plus", records)] == ["a"]
