from flask import Flask, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('finalized_model.pkl', 'rb'))
# Load the targets dictionary from disk
with open('targets.pkl', 'rb') as f:
    targets = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return {'prediction': targets[int(prediction[0])]}