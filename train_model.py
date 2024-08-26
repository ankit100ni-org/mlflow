from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from data_preprocessing import X_train, X_test, y_train, y_test

# Start an MLflow run
with mlflow.start_run():
    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Log parameters and metrics
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("MAE", mean_absolute_error(y_test, predictions))
    mlflow.log_metric("RMSE", mean_squared_error(y_test, predictions, squared=False))
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")
