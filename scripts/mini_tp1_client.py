"""Mini-TP 1 client: one valid and one invalid prediction against the running API.

With the API up on localhost:8000 (compose, or uvicorn locally):

    API_KEY=<one of the API's API_KEYS> uv run --group api python scripts/mini_tp1_client.py

Expected: the valid call answers 200 with the probability and the model version; the
invalid one answers 422 naming the bad field; a third call with a wrong key answers 401.
Set API_URL to point at another host.
"""

import json
import os
import sys

import httpx

BASE_URL = os.getenv("API_URL", "http://localhost:8000")

VALID = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}
INVALID = {**VALID, "age": -5}  # the contract requires 0 < age <= 120

EXPECTED = [200, 422, 401]


def run(client: httpx.Client) -> list[int]:
    """Send the three calls, print each status and body, and return the statuses."""
    calls = [
        ("valid", VALID, {}),
        ("invalid (age=-5)", INVALID, {}),
        ("wrong API key", VALID, {"X-API-KEY": "not-a-real-key"}),
    ]
    statuses = []
    for name, body, headers in calls:
        response = client.post("/v1/predict", json=body, headers=headers)
        print(f"--- {name}: HTTP {response.status_code}")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        statuses.append(response.status_code)
    return statuses


def main() -> int:
    key = os.environ.get("API_KEY")
    if not key:
        print("Set API_KEY to one of the values in the API's API_KEYS.", file=sys.stderr)
        return 2
    with httpx.Client(base_url=BASE_URL, headers={"X-API-KEY": key}, timeout=10) as client:
        statuses = run(client)
    print(f"\nstatuses {statuses}, expected {EXPECTED}")
    return 0 if statuses == EXPECTED else 1


if __name__ == "__main__":
    sys.exit(main())
