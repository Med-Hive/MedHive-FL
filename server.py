from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

from data.task import get_model, get_model_params, set_initial_params
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitRes, Parameters

import mlflow
import pyngrok


class CustomFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        print(f"\n--- Round {server_round} completed ---")
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        accuracies = [
            r[1].metrics["accuracy"] * r[1].num_examples for r in results if r[1].metrics
        ]
        examples = [
            r[1].num_examples for r in results if r[1].metrics
        ]
        if examples:
            avg_accuracy = sum(accuracies) / sum(examples)
            print(f"Round {server_round} - Avg Accuracy: {avg_accuracy:.4f}")
            aggregated_metrics["accuracy"] = avg_accuracy

            # Log to MLflow
            mlflow.log_metric("avg_accuracy", avg_accuracy, step=server_round)
            mlflow.log_metric("loss", aggregated_loss, step=server_round)

        return aggregated_loss, aggregated_metrics


def server_fn(context: Context):
    num_rounds = context.run_config.get("num-server-rounds", 5)
    penalty = context.run_config.get("penalty", "l2")
    local_epochs = context.run_config.get("local-epochs", 5)

    model = get_model(penalty, local_epochs)
    set_initial_params(model)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    # Start MLflow experiment
    mlflow.set_experiment("Federated_LogReg")
    mlflow.start_run()

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
