import os
import mlflow
from pyngrok import ngrok
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
import joblib
from datetime import datetime

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("federated_learning_experiment")

def save_model(model, client_id=None):
    """Save model to fl_models directory"""
    os.makedirs("fl_models", exist_ok=True)
    if client_id:
        filename = f"fl_models/client_{client_id}_model.pkl"
    else:
        filename = "fl_models/final_model.pkl"
    joblib.dump(model, filename)

def main():
    # Initialize ngrok
    ngrok_tunnel = ngrok.connect(8080)
    print(f"Public URL: {ngrok_tunnel.public_url}")
    
    # Define strategy
    strategy = FedAvg(
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
    )
    
    # Configure server
    config = ServerConfig(num_rounds=3)
    
    # Start MLflow run
    with mlflow.start_run():
        # Start Flower server
        server_app = ServerApp(
            config=config,
            strategy=strategy,
        )
        
        # Start server
        server_app.start_server(
            server_address="0.0.0.0:8080",
            root_certificates=None,
        )
        
        # After training, save the final model
        final_model = server_app.strategy.aggregate_fit(
            server_app.strategy.parameters,
            server_app.strategy.parameters,
        )
        save_model(final_model)
        
        # Log metrics to MLflow
        mlflow.log_metric("num_clients", 3)
        mlflow.log_metric("rounds", 3)
        mlflow.log_artifact("fl_models/final_model.pkl")

if __name__ == "__main__":
    main() 