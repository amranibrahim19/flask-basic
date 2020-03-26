import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tablib
import os
import pandas as pd
from flask_cors import CORS, cross_origin


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


app = Flask (__name__)
dataset = tablib.Dataset()
with open(os.path.join(os.path.dirname(__file__),'SalesJan2009.csv')) as f:
    dataset.csv = f.read()



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = int_features
    #prediction = model.predict(final_features)

    return render_template('index.html', prediction_text='Output {}'.format(final_features))

@app.route('/predict_api',methods=['GET'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    	app.run(port=9000, debug=True)