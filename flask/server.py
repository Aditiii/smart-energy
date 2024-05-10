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
# from openai import OpenAI
import os
from pprint import pprint
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

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
    data['Kitchen'] = data[['Kitchen 12','Kitchen 14','Kitchen 38']].sum(axis=1)
    data.drop(['Furnace 1','Furnace 2','Kitchen 12','Kitchen 14','Kitchen 38','icon','summary'], axis=1, inplace=True)

    data['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)
    data['cloudCover'] = data['cloudCover'].astype('float')

    fig = plt.subplots(figsize=(8, 6))
    sns.heatmap(data[data.columns[0:15].tolist()].corr(), annot=True)
    plt.title('Correlation Matrix')
    data.drop(['use', 'gen'], axis=1, inplace=True)

    data['month'] = data.index.month
    data['day'] = data.index.day
    data['weekday'] = data.index.day_name()
    data['hour'] = data.index.hour
    data['minute'] = data.index.minute

    data_daily = data['Furnace'].resample('T').mean()
    size = int(len(data_daily)*0.9)
    train = data_daily[:size]
    test = data_daily[size:]

    model = ARIMA(train, order=(2,1,1))
    model_fit = model.fit()
    print('Akaike information criterion: ', model_fit.aic)
    plt.figure(figsize=(15,4))
    plt.plot(data_daily, c='blue',label='Data - Fridge')
    plt.plot(model_fit.predict(dynamic=False), c='red', label='model')
    plt.legend()
    plt.grid(), plt.margins(x=0)
    print(model_fit.summary())

    new_fit = model_fit.append(data_daily[size:size+1], refit=False)
    forecast = model_fit.forecast(len(test))
    confidence = model_fit.get_forecast(len(test)).conf_int(0.05)
    plt.figure(figsize=(15,4))
    plt.plot(train, c='blue',label='train data')
    plt.plot(model_fit.predict(dynamic=False), c='green', label='model')
    plt.plot(test, c='blue',label='test data')
    plt.plot(forecast, c='red', label='model')
    plt.fill_between(confidence.index,confidence['lower Furnace'],
                    confidence['upper Furnace'], color='k', alpha=.15)
    plt.legend()
    plt.grid(), plt.margins(x=0)
    print('MSE: %.3f' % (mean_squared_error(test, forecast)))
    print('RMSE: %.3f' % np.sqrt(mean_squared_error(test, forecast)))
    MAE = mean_absolute_error(test, forecast)
    print('MAE: %.3f' % MAE)

    print('R2 score: %.3f' % r2_score(test, forecast))

    n = 1
    X = data_daily.values
    size = int(len(X) * 0.9)
    train, test = X[0:size], X[size:len(X)]
    predictions = list()
    confidence = list()
    history = [x for x in train]
    # walk-forward validation
    for t in range(0,len(test),n):
        model = ARIMA(history, order=(2,0,1))
        model_fit = model.fit()
        output = model_fit.forecast(n).tolist()
        conf = model_fit.get_forecast(n).conf_int(0.05)
        predictions.extend(output)
        confidence.extend(conf)
        obs = test.tolist()[t:t+n]
        history = history[n:]
        history.extend(obs);  
    conf_int =  np.vstack(confidence)

    m = len(predictions) - len(test)
    index_extended = data_daily[size:].index.union(data_daily[size:].index.shift((m))[-(m):])
    predictions_series = pd.Series(predictions, index=index_extended)
    confidence = pd.DataFrame(conf_int, columns=['lower', 'upper'])
    plt.figure(figsize=(15,4))
    plt.plot(data_daily[:size], c='green',label='train data')
    plt.plot(data_daily[size:], c='blue',label='test data')
    plt.plot(predictions_series, c='red', label='predictions')
    plt.fill_between(predictions_series.index, confidence['lower'],
                    confidence['upper'], color='k', alpha=.15)
    plt.legend()
    plt.grid(), plt.margins(x=0)
    plt.title('Results for Dishwasher Data'), plt.xticks(rotation=45)

    print('MSE: %.5f' % (mean_squared_error(test, predictions[:len(test)])))
    print('RMSE: %.3f' % np.sqrt(mean_squared_error(test, predictions[:len(test)])))
    MAE = mean_absolute_error(test, predictions[:len(test)])
    MAPE = np.mean(np.abs(predictions[:len(test)] - test)/np.abs(test))
    MASE = np.mean(np.abs(test - predictions[:len(test)]))/(np.abs(np.diff(train)).sum()/(len(train)-1))
    print('MAE: %.3f' % MAE)
    print('R^2 score: %.3f' % r2_score(test, predictions[:len(test)]))

    forecasted_values_dict.clear()
    forecasted_values_dict = {"Dishwasher": [[index, value] for index, value in zip(predictions_series.index, predictions_series.values)]}
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(forecasted_values_dict)
    cleaned_data = {appliance: [value for _, value in values[:30]] for appliance, values in data.items()}

    print(json.dumps(cleaned_data, indent=4))
    json_data = json.dumps(cleaned_data)
    with open('forecasted_values_month.json', 'w') as json_file:
        json_file.write(json_data)

    try:
        return jsonify({'success': json_data}), 200
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
    write_jsonl(training_data, training_file_name)
    validation_file_name = "tmp_recipe_finetune_validation.jsonl"
    write_jsonl(validation_data, validation_file_name)
  
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

    response = client.fine_tuning.jobs.retrieve(job_id)
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
    messages1 = []
    messages2 = []
    system_message = '''You are a helpful Smart Energy Assistant. 
            You are to answer all energy consumption related queries in a household.''' 
            # Answer the above question based on the following format: 
            # user prompt: Tell the current status of all the appliances - 
            # formatted output: {"to_say" : "Here is the current status of all appliances",  
            # "service" : "status()", "target" : "all"}

            # user prompt: "''' + messageInput + '''"'''

    if str == "train":
        messages1.append({"role": "system", "content": system_message})
        messages1.append({"role": "user", "content": "Could you please switch off the fridge"})
        messages1.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Sure, turing off the fridge",  
            "service" : "turn_off()", "target" : "fridge"}"'''})
        messages2.append({"role": "system", "content": system_message})
        messages2.append({"role": "user", "content": "Could you please switch on the furnace"})
        messages2.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Hey, sure, I'll turn on the furnace",  
            "service" : "turn_on()", "target" : "furnace"}"'''})
        response_data = [{"messages": messages1}, {"messages": messages2}]
        return response_data
    else:
        messages1.append({"role": "system", "content": system_message})
        messages1.append({"role": "user", "content": "Could you put off the fridge"})
        messages1.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Sure, turing off the fridge",  
            "service" : "turn_off()", "target" : "fridge"}"'''})
        messages2.append({"role": "system", "content": system_message})
        messages2.append({"role": "user", "content": "Could you please shut down the furnace"})
        messages2.append({"role": "assistant", "content": '''"formatted output: {"to_say" : "Hey, sure, I'll turn on the furnace",  
            "service" : "turn_off()", "target" : "furnace"}"'''})
        response_data = [{"messages": messages1}, {"messages": messages2}]
        return response_data

@app.route('/anomalyDetection', methods=['POST'])
@cross_origin()
def anomaly_detection():
    try:
        data = request.get_json()
        current_values = data['currentValues']
        predicted_values = data['predictedValues']
        currentTime = data['currentTime']
        fridge_current_value = current_values['fridge']
        furnace_current_value = current_values['furnace']
        dishwasher_current_value = current_values['dishwasher']
        fridge_predicted_values = predicted_values['fridge_predicted']
        furnace_predicted_values = predicted_values['furnace_predicted']
        dishwasher_predicted_values = predicted_values['dishwasher_predicted']

        if len(fridge_predicted_values) > currentTime:
            fridge_predicted_value = fridge_predicted_values[currentTime]
            dishwasher_predicted_value = dishwasher_predicted_values[currentTime]
            furnace_predicted_value = furnace_predicted_values[currentTime]
            if fridge_predicted_value is not None:
                message = '''"Tell whether the energy consumption is anomalous or normal for fridge, predicted value is {fridge_predicted_value} kWh, actual value is {fridge_current_value} kWh with a reason in this format\n 
                '"formatted output: {"to_say": "The energy consumption of fridge is anomalous/normal with a reason why it is anomalous", "service": "anomaly()/normal()", "target": "fridge"}"'''
           
        else:
            print('No predicted value available for the current time.')

        # question = data['question']
        model_response = get_model_response(message)
        return jsonify({'response': model_response}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

def write_jsonl(data_list: list, filename: str) -> None:
    with open(filename, "w") as out:
        for ddict in data_list:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

if __name__ == '__main__':
    # finetune_model()
    app.run(port=8000, debug=True)