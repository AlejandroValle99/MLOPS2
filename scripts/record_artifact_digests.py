"""Record `artifact_sha256` on every version registered before the digest existed.

The model server loads a version only after checking its files against that tag, and
refuses a version without one. Versions trained since the tag was introduced carry it from
registration; run this once to give the older ones theirs. It downloads each one, so it
needs MinIO as well as MLflow; model-init's container has both:

    docker compose --env-file .env.local run --rm model-init \
        python -m scripts.record_artifact_digests

Then restart the model server (it doesn't retry a version it refused before).

This trusts the files as they are today: it records whatever MinIO holds now. Run it when
the digest is introduced, never to get a refused version loaded: that refusal means the
files changed after the digest was recorded, which is the check doing its job.
"""

import getpass
import logging

from mlflow.tracking import MlflowClient

from stroke.config import MODEL_NAME
from stroke.registry import record_missing_digests


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    client = MlflowClient()  # MLFLOW_TRACKING_URI from the environment
    recorded = record_missing_digests(client, MODEL_NAME, actor=f"{getpass.getuser()} (backfill)")
    if recorded:
        print(f"{MODEL_NAME}: recorded artifact_sha256 on versions {', '.join(recorded)}")
    else:
        print(f"{MODEL_NAME}: every version already has its artifact_sha256")


if __name__ == "__main__":
    main()
