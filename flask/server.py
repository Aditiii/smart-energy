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
import json
from openai import OpenAI
import os
from pprint import pprint

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

def finetune_model():
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-proj-e3QZjmsa1hrsvNqwJnIHT3BlbkFJlVCgUArGrhReSbRYVUDz"))
    training_data = []
    training_data = prepare_example_conversation("train")
    validation_data = [] 
    validation_data = prepare_example_conversation("validate")
    training_file_name = "tmp_recipe_finetune_training.jsonl"
    json_data = json.dumps(training_data, indent=4)
    with open(training_file_name, 'w') as file:
        file.write(json_data)

    validation_file_name = "tmp_recipe_finetune_validation.jsonl"
    json_data = json.dumps(validation_data, indent=4)

    with open(validation_file_name, 'w') as file:
        file.write(json_data)

    with open(training_file_name, "rb") as training_fd:
        training_response = client.files.create(
            file=training_fd, purpose="fine-tune"
        )

    training_file_id = training_response.id

    with open(validation_file_name, "rb") as validation_fd:
        validation_response = client.files.create(
            file=validation_fd, purpose="fine-tune"
        )
    validation_file_id = validation_response.id

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)

    response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model="gpt-3.5-turbo",
    suffix="smart-energy",
    )

    job_id = response.id

    print("Job ID:", response.id)
    print("Status:", response.status)

    response = client.fine_tuning.jobs.retrieve("ftjob-7e0vys0urReIlaF3wbvAE7Kc")
    print("Trained Tokens:", response.trained_tokens)

    response = client.fine_tuning.jobs.list_events(job_id)

    events = response.data
    events.reverse()

    for event in events:
        print(event.message)

    response = client.fine_tuning.jobs.retrieve(job_id)
    fine_tuned_model_id = response.fine_tuned_model
    if fine_tuned_model_id is None: 
        raise RuntimeError("Fine-tuned model ID not found. Your job has likely not been completed yet.")

    print("Fine-tuned model ID:", fine_tuned_model_id)

def prepare_example_conversation(str):
    messages = []
    system_message = '''You are a helpful Smart Energy Assistant. 
            You are to answer all energy consumption related queries in a household.''' 
            # Answer the above question based on the following format: 
            # user prompt: Tell the current status of all the appliances - 
            # formatted output: {"to_say" : "Here is the current status of all appliances",  
            # "service" : "status()", "target" : "all"}

            # user prompt: "''' + messageInput + '''"'''

    if str == "train":
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": "Could you please switch off the fridge"})
        messages.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Sure, turing off the fridge",  
            "service" : "turn_off()", "target" : "fridge"}"'''})
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": "Could you please switch on the furnace"})
        messages.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Hey, sure, I'll turn on the furnace",  
            "service" : "turn_on()", "target" : "furnace"}"'''})
        return {"messages": messages}
    else:
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": "Could you put off the fridge"})
        messages.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Sure, turing off the fridge",  
            "service" : "turn_off()", "target" : "fridge"}"'''})
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": "Could you please shut down the furnace"})
        messages.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Hey, sure, I'll turn on the furnace",  
            "service" : "turn_off()", "target" : "furnace"}"'''})
    
        return {"messages": messages}


if __name__ == '__main__':
    finetune_model()
    # app.run(port=8000, debug=True)