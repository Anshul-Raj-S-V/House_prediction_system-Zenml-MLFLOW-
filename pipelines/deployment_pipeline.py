import os
from zenml import pipeline
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

# Import your custom steps
from pipelines.training_pipelines import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.model_loader_step import model_loader
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor

# Optional: specify requirements for reproducibility
requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")


@pipeline(enable_cache=False)
def continuous_deployment_pipeline():
    """
    Run a training job and (re)deploy an MLflow model deployment.
    """
    # Run your training pipeline
    trained_model = ml_pipeline()

    # Deploy the trained model with MLflow
    mlflow_model_deployer_step(
        model=trained_model,
        workers=1,  # Windows can't handle multiple MLflow workers easily
        deploy_decision=True
    )


@pipeline(enable_cache=False)
def inference_pipeline():
    """
    Run a batch inference job with data loaded dynamically.
    """
    # Load data for inference
    batch_data = dynamic_importer()

    # Load the deployed MLflow model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step"
    )

    # Run predictions using the deployed model
    predictor(service=model_deployment_service, input_data=batch_data)
