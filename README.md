# MedHive-FL: Federated Learning for Medical Diagnostics

A privacy-preserving federated learning system for medical diagnostics using Flower framework, with MLflow tracking and Docker containerization.

## 🌟 Features

- **Privacy-Preserving Learning**: Train machine learning models collaboratively without sharing raw patient data
- **Distributed Training**: Support for multiple clients training simultaneously
- **MLflow Integration**: Track experiments, metrics, and model performance
- **Docker Support**: Easy deployment and distribution of both server and client components
- **Ngrok Integration**: Secure tunneling for remote client connections
- **Real-time Monitoring**: Track training progress and model performance

## 🏗️ Architecture

The system consists of three main components:

1. **FL Server**: Coordinates the training process and aggregates model updates
2. **FL Clients**: Train models on local data and share only model parameters
3. **MLflow Server**: Tracks experiments and visualizes results

## 🚀 Getting Started

### Prerequisites

- Docker
- Python 3.9+
- Ngrok account (for server deployment)

### Server Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MedHive-FL.git
cd MedHive-FL
```

2. Set up environment variables:
```bash
# Create .env file with your Ngrok auth token
echo "NGROK_AUTHTOKEN=your_token_here" > .env
```

3. Start the server:
```bash
docker compose up -d
```

The server will start and expose:
- MLflow UI on port 8080
- FL Server with a public Ngrok URL (check logs for URL)

### Client Setup

1. Build the client Docker image:
```bash
docker build -t fl-client -f Dockerfile .
```

2. Run a client instance:
```bash
docker run -e SERVER_ADDRESS="<ngrok_url>" -e CLIENT_ID="<unique_id>" fl-client
```

## 📊 Model Details

- **Algorithm**: Logistic Regression
- **Training Strategy**: FedAvg (Federated Averaging)
- **Evaluation Metrics**: 
  - Accuracy
  - Loss
  - Training rounds completion

## 🔍 MLflow Tracking

Access the MLflow dashboard to monitor:
- Training progress
- Model metrics per round
- Aggregate model performance
- Client participation

## 📁 Project Structure

```
MedHive-FL/
├── client.py           # FL client implementation
├── server.py          # FL server implementation
├── docker-compose.yml # Server orchestration
├── Dockerfile        # Client container definition
├── requirements.txt  # Python dependencies
└── data/
    ├── data.csv     # Medical dataset
    └── task.py      # Data loading and model definitions
```

## 🛠️ Configuration

Server configuration options:
- `num-server-rounds`: Number of training rounds (default: 5)
- `penalty`: Regularization type (default: "l2")
- `local-epochs`: Client-side training epochs (default: 5)
- `min-available-clients`: Minimum clients for training (default: 2)

## 📈 Performance Tracking

The system tracks:
- Round-wise accuracy
- Loss metrics
- Client participation
- Training completion status

## 🔒 Privacy & Security

- Raw data never leaves client devices
- Only model parameters are shared
- Secure communication via Ngrok tunneling
- Client authentication support

## 📋 Requirements

All dependencies are listed in `requirements.txt`. Key dependencies:
- flower>=1.0.0
- scikit-learn>=1.0.2
- mlflow>=2.3.0
- pyngrok>=6.0.0
- pandas>=1.3.0

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests.

## 📄 License

This project is licensed under the terms of the LICENSE file included in the repository.

## 🙏 Acknowledgments

- Flower Framework team
- MLflow community
- Contributors and maintainers

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.