# MLflow Beginner Project

This project demonstrates how to use MLflow for tracking and managing machine learning experiments using the California Housing dataset.

## Project Structure
- `data_preprocessing.py`: Script for data loading and preprocessing.
- `train_model.py`: Script for training a linear regression model and logging it with MLflow.
- `app.py`: Flask application to deploy the trained model as an API.
- `Dockerfile`: Docker configuration to containerize the Flask app.
- `.github/workflows/main.yml`: GitHub Actions configuration for CI/CD.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mlflow-beginner-project.git
   cd mlflow-beginner-project
