"""Roll the served model back: point `champion` at `previous_champion` again.

    MLFLOW_TRACKING_URI=http://localhost:5001 uv run --group ml python -m scripts.rollback

Only aliases move: nothing is retrained or deleted, and no model files are read, so the
registry is all this needs. The model server follows the alias and serves the restored
version, with that version's own threshold, within one reload interval. `previous_champion`
is used up by the rollback, so running this twice in a row fails the second time.
"""

from mlflow.tracking import MlflowClient

from stroke.config import MODEL_NAME
from stroke.registry import rollback


def main() -> None:
    client = MlflowClient()  # MLFLOW_TRACKING_URI from the environment
    version = rollback(client, MODEL_NAME)
    tags = client.get_model_version(MODEL_NAME, version).tags
    print(
        f"{MODEL_NAME}: champion is version {version} again "
        f"({tags.get('model_family', '?')}, threshold {tags.get('threshold', 'none')})"
    )


if __name__ == "__main__":
    main()
