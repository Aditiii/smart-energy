from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS, cross_origin
from sklearn.cluster import SpectralBiclustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
import warnings
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from pmdarima import auto_arima
import torch
from transformers import AutoTokenizer, LlamaForCausalLM, PretrainedConfig, OpenLlamaModel, OpenLlamaConfig
import pickle
import openai
import streamlit as st

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/get-prediction', methods=['POST'])
@cross_origin()
def get_prediction():
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data['time'] = pd.DatetimeIndex(pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min'))
    data = data.set_index('time')

    data.columns = [i.replace(' [kW]', '') for i in data.columns]
    data['Furnace'] = data[['Furnace 1','Furnace 2']].sum(axis=1)
    data['Kitchen'] = data[['Kitchen 12','Kitchen 14','Kitchen 38']].sum(axis=1) #We could also use the mean 
    data.drop(['Furnace 1','Furnace 2','Kitchen 12','Kitchen 14','Kitchen 38','icon','summary'], axis=1, inplace=True)

    #Replacing invalid values in 'cloudCover' with backfill
    data['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
    data['cloudCover'] = data['cloudCover'].astype('float')

    #Reordering
    data = data[['use', 'gen', 'House overall', 'Dishwasher', 'Home office', 'Fridge', 'Wine cellar', 'Garage door', 'Barn',
                'Well', 'Microwave', 'Living room', 'Furnace', 'Kitchen', 'Solar', 'temperature', 'humidity', 'visibility', 
                'apparentTemperature', 'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity', 
                'dewPoint', 'precipProbability']]
    
    #correlation
    fig = plt.subplots(figsize=(8, 6))
    sns.heatmap(data[data.columns[0:15].tolist()].corr(), annot=True)
    plt.title('Correlation Matrix')
    
    try:
        return jsonify({'success': 'hello ji'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


def data_preprocessing():
    data = pd.read_csv("HomeC.csv",low_memory=False)
    data = data[:-1]
    data.head()

@app.route('/ask', methods=['POST'])
@cross_origin()
def ask_question():
    try:
        data = request.get_json()
        question = data['question']
        model_response = get_model_response(question)
        return jsonify({'response': model_response}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

def get_model_response(message):
    openai.api_key = 'sk-proj-e3QZjmsa1hrsvNqwJnIHT3BlbkFJlVCgUArGrhReSbRYVUDz'
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [{ "role": "user", "content": message }],
        stream=True
    )
    response_line = ""
    
    for chunk in chat:
        chunk_message = chunk['choices'][0]['delta']
        if chunk_message.get('content'):
            response_line += chunk_message['content']
    return response_line

@app.route('/get-response', methods=['POST'])
@cross_origin()
def get_response():    
    try:
        return jsonify({'success': 'hello ji'}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=8000, debug=True)