import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from zenml import ArtifactConfig, step
from zenml.client import Client
from zenml import Model

# Initialize ZenML model and experiment tracker
client = Client()
experiment_tracker = client.active_stack.experiment_tracker

model = Model(
    name="prices_predictor",
    version=None,
    license="Apache 2.0",
    description="Price prediction model for houses.",
)

@step(enable_cache=False, experiment_tracker=experiment_tracker.name, model=model)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[Pipeline, ArtifactConfig(name="sklearn_pipeline", is_model_artifact=True)]:
    """
    Builds, trains, and logs a Linear Regression model with MLflow.
    """

    # Validate input types
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")

    # Define preprocessing pipelines
    numerical_transformer = SimpleImputer(strategy="mean")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Define full ML pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LinearRegression())]
    )

    # Start MLflow tracking
    run = None
    try:
        run = mlflow.start_run(run_name="model_building", nested=True)
        mlflow.sklearn.autolog()

        logging.info("🚀 Starting Linear Regression model training...")
        pipeline.fit(X_train, y_train)
        logging.info("✅ Model training completed successfully.")

        # ✅ Log model explicitly for serving later
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="house_price_predictor",
        )
        logging.info("📦 Model successfully logged to MLflow as an artifact.")

        # Optional: Log column info
        onehot_encoder = (
            pipeline.named_steps["preprocessor"].transformers_[1][1].named_steps["onehot"]
        )
        onehot_encoder.fit(X_train[categorical_cols])
        expected_columns = numerical_cols.tolist() + list(
            onehot_encoder.get_feature_names_out(categorical_cols)
        )

        mlflow.log_dict({"expected_input_columns": expected_columns}, "input_schema.json")
        logging.info(f"📊 Logged expected columns: {expected_columns}")

    except Exception as e:
        logging.error(f"❌ Error during model training: {e}")
        raise e

    finally:
        if run:
            mlflow.end_run()
            logging.info("🧹 MLflow run closed properly.")

    return pipeline
