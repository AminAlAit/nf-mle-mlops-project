"""MLflow Gateway: Centralized tracking adapter and registry service."""

from pathlib import Path
from typing import Any, Dict, List
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

class MLflowGateway:
    """Gateway adapter to decouple core training code from MLflow SDK calls."""

    def __init__(self, db_path: Path, artifacts_dir: Path, experiment_name: str = "mlops-taxi-project"):
        self.db_path         = db_path
        self.artifacts_dir   = artifacts_dir
        self.experiment_name = experiment_name
        self.tracking_uri    = f"sqlite:///{self.db_path}"

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client          = MlflowClient()
        self.experiment_id   = self._get_or_create_experiment()

    def _get_or_create_experiment(self) -> str:
        """Fetch existing experiment or initialize a new one with artifact path."""
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is not None:
            return experiment.experiment_id

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        return self.client.create_experiment(
            self.experiment_name,
            artifact_location=self.artifacts_dir.resolve().as_uri(),
        )

    def start_run(self, run_name: str = "linear-regression-taxi"):
        """Context manager hook for active MLflow runs."""
        return mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

    def log_run_data(
        self,
        tags: Dict[str, Any],
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: List[Path],
    ) -> None:
        """Record all parameters, metrics, and file artifacts to the active run."""
        mlflow.set_tags(tags)
        mlflow.log_params(params)
        for key, val in metrics.items():
            mlflow.log_metric(key, float(val))

        for artifact_path in artifacts:
            sub_folder = "reports" if "reports" in str(artifact_path) else "model"
            mlflow.log_artifact(str(artifact_path), artifact_path=sub_folder)

    def register_and_promote(
        self,
        pipeline: Any,
        model_name: str = "green-taxi-ride-duration",
        alias: str = "production",
    ) -> str:
        """Log model pipeline, register in the registry, and assign alias."""
        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline, name="mlflow-model", serialization_format="skops"
        )
        active_run = mlflow.active_run()
        run_id     = active_run.info.run_id

        try:
            self.client.create_registered_model(
                name=model_name,
                tags={"framework": "scikit-learn"},
                description="Model to predict ride duration in NYC",
            )
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise

        model_version = self.client.create_model_version(
            name=model_name,
            source=model_info.model_uri,
            run_id=run_id,
        )
        self.client.set_registered_model_alias(
            name=model_name,
            version=model_version.version,
            alias=alias,
        )
        return model_version.version
