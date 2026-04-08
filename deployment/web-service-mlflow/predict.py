import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from flask import Flask, request, jsonify

# 1. Setup Configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5050"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

# 2. Dynamically get the latest RUN_ID from Experiment '1'
try:
    runs = client.search_runs(experiment_ids=['1'])
    if not runs:
        raise Exception("No runs found in experiment 1")
    
    # Grab the latest run ID
    RUN_ID = runs[0].info.run_id
    print(f"🚀 Using latest Run ID: {RUN_ID}")
except Exception as e:
    print(f"❌ Error fetching Run ID: {e}")
    # Fallback to a hardcoded one if the dynamic search fails
    RUN_ID = "3e58818e7a6e48679a2b6e47818bd3ed"

# 3. Load Model and Artifacts using the dynamic RUN_ID
logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)

# Use 'artifact_path' to match your MLflow version
dv_path = mlflow.artifacts.download_artifacts(
    run_id=RUN_ID, 
    artifact_path="dict_vectorizer.bin"
)

with open(dv_path, 'rb') as f_in:
    dv = pickle.load(f_in)

def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    # The vectorizer turns the dict into the format the model needs
    # X = dv.transform(features)
    preds = model.predict(features)
    return float(preds[0])

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)