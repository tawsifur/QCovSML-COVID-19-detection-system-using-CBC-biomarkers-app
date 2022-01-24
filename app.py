# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 00:54:16 2021

@author: LEGION
"""

  
# Importing essential libraries

from flask import Flask, render_template, request
import pickle
import numpy as np
import math
import pandas as pd
from joblib import dump
from joblib import load
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


#model1=pickle.load(open('GB11_model.pkl','rb'))
model2=pickle.load(open('RF22_model.pkl','rb'))
#model3=pickle.load(open('GB11_model.pkl','rb'))
#model3=pickle.load(open('XGB33_model.pkl','rb'))

model1=pickle.load(open('RF22_model.pkl','rb'))
model3=pickle.load(open('RF22_model.pkl','rb'))

st_model=pickle.load(open('Stack_model.pkl','rb'))


app = Flask(__name__) 

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Age=int(request.form['Age'])
        WBC=float(request.form['WBC'])
        LY=float(request.form['LY'])
        MO=float(request.form['MO'])



        xt1=pd.DataFrame([Age,WBC,LY,MO])
        xt1=np.array(xt1)
        xt1=xt1.reshape(1,-1)

        y_prob1=model1.predict_proba(np.array(xt1)) 
        y_prob2=model2.predict_proba(np.array(xt1))
        y_prob3=model3.predict_proba(np.array(xt1))
        x_pr=np.concatenate((y_prob1,y_prob2,y_prob3),1)
        y_pr=st_model.predict(np.array(x_pr))
        
        
            
        output= y_pr
        # dp=1/(1+math.exp(-output))
        # dp=round(dp,4)
        return render_template('result.html', prediction=output)

if __name__ == '__main__':
	app.run(debug=True)