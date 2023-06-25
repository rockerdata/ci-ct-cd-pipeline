from flask import Flask, render_template, request
import numpy as np
import pickle
import os
import pickle as pkl
from config.config import CONFIG
from sklearn import datasets
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def new():
    return render_template('new.html')

def load_model():
    with open("./models/model.pkl",'rb')as f:
        loaded_model = pkl.load(f)
    return loaded_model

def preprocess():
    wbcd = wisconsin_breast_cancer_data = datasets.load_breast_cancer()
    feature_names = wbcd.feature_names
    labels = wbcd.target_names
    X_train, X_test, y_train, y_test = train_test_split(wbcd.data, wbcd.target, test_size = 0.2, random_state = 42)
    return X_test

@app.route('/predict', methods=['POST','GET'])
def predict(): 
    X_test = preprocess()
    # data=float(request.form['model_input'])
    
    # features = np.array([[data**3, data**2, data**1, data**0]])
    
    model= load_model()
    pred = model.predict(X_test)
    pred_proba=model.predict_proba(X_test)

    prediction_statement =  f"The output of the model is {pred[0]} with predict_proba {pred_proba[0]}"
    
    return render_template('new.html',statement=prediction_statement)


if __name__=='__main__':
    app.run()