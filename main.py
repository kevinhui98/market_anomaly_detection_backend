import os
import modal
from fastapi import Response, HTTPException, Query, Request
import requests
from datatime import dattime,timezone


app =modal.App("anomaly_detection")
image = modal.Image.debian_slim().pip_install("openai","pickle")

with image.imports():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    import os
    from openai import OpenAI


@app.cls(image = image,gpu = 'T4')
class Model:
    @modal.build()
    @modal.enter()
    def load_weights(self):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            # api_key = os.getenv("GROQ_API_KEY")
            api_key = os.environ["GROQ_API_KEY"]
        )
        self.dt_model = self.load_model("dt_model.pkl")
        self.rf_model = self.load_model("rf_model.pkl")
        self.lr_model = self.load_model("lr_model.pkl")
        self.dt_model_scaled = self.load_model("dt_model_scaled.pkl")
        self.rf_model_scaled = self.load_model("rf_model_scaled.pkl")
        self.svm_model_scaled = self.load_model("svm_model_scaled.pkl")
        self.lr_model_scaled = self.load_model("lr_model_scaled.pkl")

    def load_model(self, filename):
        with open(filename,"rb") as file:
            return pickle.load(file)  
    # def prepare_input(self, Y,VIX,GTITL2YR,GTITL10YR,GTITL30YR,EONIA,GTDEM30Y,GTDEM10Y,GTJPY10YR,GTDEM2Y,GTJPY30YR,GTJPY2YR,DXY,GTGBP20Y,GTGBP30Y,GTGBP2Y,USGG30YR,US0001M,GT10,XAU_BGNL,USGG3M,USGG2YR,MXBR,Cl1,CRY,BDIY,ECSURPUS,GBP,LUMSTRUU,LMBITR,MXRU,MXCN,JPY,LUACTRUU,LF94TRUU,EMUSTRUU,MXIN,LF98TRUU,MXUS,LG30TRUU,LP01TREU,MXEU,MXJP):
    #     input_df = pd.DataFrame([input_dict])
    #     return input_df, input_dict
    #     df = pd.read_csv("")
    
    def make_prediction(self, input_df):
        probs={
            "dt_model": self.dt_model.predict_proba(input_df)[0][1],
            "rf_model": self.rf_model.predict_proba(input_df)[0][1],
            "lr_model": self.lr_model.predict_proba(input_df)[0][1],
            "dt_model_scaled": self.dt_model_scaled.predict_proba(input_df)[0][1],
            "rf_model_scaled": self.rf_model_scaled.predict_proba(input_df)[0][1],
            "svm_model_scaled": self.svm_model_scaled.predict_proba(input_df)[0][1],
            "lr_model_scaled": self.lr_model_scaled.predict_proba(input_df)[0][1]
        }
        avg_prob = np.mean(list(probs.values()))
        return avg_prob
    
    def explain_prediction(self, request: Request,df,stock_name:str):
        # api_key =request.headers.get("X-API-KEY")
        # if api_key != self.api_key:
        #     raise HTTPException(
        #         status_code = 401,
        #         detail="Unauthorized"
        #     )

        # data = request.json()
        # prediction = self.client.predict(data)
        print("EXPLANATION PROMPT", prompt)
        raw_response = self.client.chat.completions.create(
            model = "llama-3.2-3b-preview",
            messages=[{
                "role": "user",
                "content": prompt
            }],
        )
        return raw_response.choices[0].message.content