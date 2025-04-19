import os
import mlflow
from flwr.client import start_client
import argparse

def run_client(client_id, server_address):
    # Set up MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("federated_learning_experiment")
    
    # Start client
    start_client(
        server_address=server_address,
        client=FlowerClient(
            model=get_model(),
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            client_id=client_id
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FL client')
    parser.add_argument('--client-id', type=int, required=True, help='Client ID (1, 2, or 3)')
    parser.add_argument('--server-address', type=str, required=True, help='Server address (from ngrok)')
    
    args = parser.parse_args()
    
    run_client(args.client_id, args.server_address) 