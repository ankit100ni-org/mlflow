from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Load the model as a PyFunc model
model = mlflow.pyfunc.load_model("runs:/9417b54d56494930a9416e5026b7c375/model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(pd.DataFrame(data))
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
