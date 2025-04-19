"""A Flower / sklearn client with MLflow integration."""

import warnings
import mlflow
import joblib
import os
from sklearn.metrics import log_loss

from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context
from only_this_was_working.task import (
    get_model,
    get_model_params,
    load_data,
    set_initial_params,
    set_model_params,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, X_test, y_train, y_test, client_id):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.client_id = client_id

    def fit(self, parameters, config):
        set_model_params(self.model, parameters)

        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)

        # Save client model
        self.save_client_model()

        return get_model_params(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)

        loss = log_loss(self.y_test, self.model.predict_proba(self.X_test))
        accuracy = self.model.score(self.X_test, self.y_test)

        # Log metrics to MLflow
        mlflow.log_metric(f"client_{self.client_id}_loss", loss)
        mlflow.log_metric(f"client_{self.client_id}_accuracy", accuracy)

        return loss, len(self.X_test), {"accuracy": accuracy}

    def save_client_model(self):
        """Save the client's model"""
        os.makedirs("fl_models", exist_ok=True)
        joblib.dump(self.model, f"fl_models/client_{self.client_id}_model.pkl")


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    client_id = context.node_config["client-id"]

    X_train, X_test, y_train, y_test = load_data(partition_id, num_partitions)

    # Create LogisticRegression Model
    penalty = context.run_config["penalty"]
    local_epochs = context.run_config["local-epochs"]
    model = get_model(penalty, local_epochs)

    # Setting initial parameters
    set_initial_params(model)

    return FlowerClient(model, X_train, X_test, y_train, y_test, client_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)