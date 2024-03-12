from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model from disk
model = pickle.load(open('finalized_model.pkl', 'rb'))

# Load the targets dictionary from disk
with open('targets.pkl', 'rb') as f:
    targets = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('./templates/home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(np.array(data['input']).reshape(1, -1))
    return {'prediction': targets[int(prediction[0])]}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)