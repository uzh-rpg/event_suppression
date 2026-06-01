import os
import shutil
import mlflow
import yaml

# Define the path to your existing run
old_run_path = "/home/rpg/Desktop/taming_event_flow/mlruns/mlruns/42"

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:/home/rpg/Desktop/taming_event_flow/mlruns")

# Start a new MLflow run
with mlflow.start_run() as run:
    new_run_id = run.info.run_id
    new_run_path = f"/home/rpg/Desktop/taming_event_flow/mlruns/mlruns/0/{new_run_id}"

    print(f"Restoring run {new_run_id} from {old_run_path}...")

    # Copy artifacts (models, logs, etc.)
    old_artifacts = os.path.join(old_run_path, "artifacts")
    new_artifacts = os.path.join(new_run_path, "artifacts")
    if os.path.exists(old_artifacts):
        shutil.copytree(old_artifacts, new_artifacts, dirs_exist_ok=True)

    # Copy parameters
    old_params = os.path.join(old_run_path, "params")
    new_params = os.path.join(new_run_path, "params")
    if os.path.exists(old_params):
        shutil.copytree(old_params, new_params, dirs_exist_ok=True)

        # Log parameters in MLflow
        for param_file in os.listdir(new_params):
            with open(os.path.join(new_params, param_file), "r") as f:
                mlflow.log_param(param_file, f.read().strip())

    # Copy metrics
    old_metrics = os.path.join(old_run_path, "metrics")
    new_metrics = os.path.join(new_run_path, "metrics")
    if os.path.exists(old_metrics):
        shutil.copytree(old_metrics, new_metrics, dirs_exist_ok=True)

        # Log metrics in MLflow
        for metric_file in os.listdir(new_metrics):
            with open(os.path.join(new_metrics, metric_file), "r") as f:
                values = [line.strip().split() for line in f.readlines()]
                for v in values:
                    mlflow.log_metric(metric_file, float(v[1]))

    # Copy tags
    old_tags = os.path.join(old_run_path, "tags")
    new_tags = os.path.join(new_run_path, "tags")
    if os.path.exists(old_tags):
        shutil.copytree(old_tags, new_tags, dirs_exist_ok=True)

        # Log tags in MLflow
        for tag_file in os.listdir(new_tags):
            with open(os.path.join(new_tags, tag_file), "r") as f:
                mlflow.set_tag(tag_file, f.read().strip())

    # Copy metadata
    old_meta = os.path.join(old_run_path, "meta.yaml")
    new_meta = os.path.join(new_run_path, "meta.yaml")
    if os.path.exists(old_meta):
        shutil.copy(old_meta, new_meta)

    print(f"Run restored successfully! You can now view it in MLflow UI.")

# Start MLflow UI (optional)
print("To view the run, start MLflow UI with: mlflow ui")
