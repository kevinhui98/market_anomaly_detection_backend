from fastapi import FastAPI, Query
import pickle
import pandas as pd
import numpy as np
import json
app = FastAPI()

# Load the all the models
def load_model(filename):
        with open(filename,"rb") as file:
            return pickle.load(file)  

dt_model = load_model("dt_model.pkl")
rf_model = load_model("rf_model.pkl")
lr_model = load_model("lr_model.pkl")
dt_model_scaled = load_model("dt_model_scaled.pkl")
rf_model_scaled = load_model("rf_model_scaled.pkl")
svm_model_scaled = load_model("svm_model_scaled.pkl")
lr_model_scaled = load_model("lr_model_scaled.pkl")

with open("data.json","rb") as data:
      df = pd.DataFrame(json.load(data))

# df = pd.read_csv("FinancialMarketData.xlsx - EWS.csv")
def preprocess_data(market_dict):
#       input_dict = {
#             'year' : [years for _, years in market_dict['Data']],
#             'Y' :[ Y for _, Y in market_dict['Y']]  ,
#             'VIX' :[ VIX for _, VIX in market_dict['VIX']]  ,
#             'GTITL2YR' :[ GTITL2YR for _, GTITL2YR in market_dict['GTITL2YR']]  ,
#             'GTITL10YR' :[ GTITL10YR for _, GTITL10YR in market_dict['GTITL10YR']]  ,
#             'GTITL30YR' :[ GTITL30YR for _, GTITL30YR in market_dict['GTITL30YR']]  ,
#             'EONIA' :[ EONIA for _, EONIA in market_dict['EONIA']]  ,
#             'GTDEM30Y' :[ GTDEM30Y for _, GTDEM30Y in market_dict['GTDEM30Y']]  ,
#             'GTDEM10Y' :[ GTDEM10Y for _, GTDEM10Y in market_dict['GTDEM10Y']]  ,
#             'GTJPY10YR' :[ GTJPY10YR for _, GTJPY10YR in market_dict['GTJPY10YR']]  ,
#             'GTDEM2Y' :[ GTDEM2Y for _, GTDEM2Y in market_dict['GTDEM2Y']]  ,
#             'GTJPY30YR' :[ GTJPY30YR for _, GTJPY30YR in market_dict['GTJPY30YR']]  ,
#             'GTJPY2YR' :[ GTJPY2YR for _, GTJPY2YR in market_dict['GTJPY2YR']]  ,
#             'DXY' :[ DXY for _, DXY in market_dict['DXY']]  ,
#             'GTGBP20Y' :[ GTGBP20Y for _, GTGBP20Y in market_dict['GTGBP20Y']]  ,
#             'GTGBP30Y' :[ GTGBP30Y for _, GTGBP30Y in market_dict['GTGBP30Y']]  ,
#             'GTGBP2Y' :[ GTGBP2Y for _, GTGBP2Y in market_dict['GTGBP2Y']]  ,
#             'USGG30YR' :[ USGG30YR for _, USGG30YR in market_dict['USGG30YR']]  ,
#             'US0001M' :[ US0001M for _, US0001M in market_dict['US0001M']]  ,
#             'GT10' :[ GT10 for _, GT10 in market_dict['GT10']]  ,
#             'XAU_BGNL' :[ XAU_BGNL for _, XAU_BGNL in market_dict['XAU_BGNL']]  ,
#             'USGG3M' :[ USGG3M for _, USGG3M in market_dict['USGG3M']]  ,
#             'USGG2YR' :[USGG2YR for _, USGG2YR in market_dict['USGG2YR']]  ,
#             'MXBR' :[MXBR for _, MXBR in market_dict['MXBR']]  ,
#             'Cl1' :[Cl1 for _, Cl1 in market_dict['Cl1']]  ,
#             'CRY' :[CRY for _, CRY in market_dict['CRY']]  ,
#             'BDIY' :[BDIY for _, BDIY in market_dict['BDIY']]  ,
#             'ECSURPUS' :[ECSURPUS for _, ECSURPUS in market_dict['ECSURPUS']]  ,
#             'GBP' :[GBP for _, GBP in market_dict['GBP']]  ,
#             'LUMSTRUU' :[ LUMSTRUU for _, LUMSTRUU in market_dict['LUMSTRUU']]  ,
#             'LMBITR' :[LMBITR for _, LMBITR in market_dict['LMBITR']]  ,
#             'MXRU' :[MXRU for _, MXRU in market_dict['MXRU']]  ,
#             'MXCN' :[MXCN for _, MXCN in market_dict['MXCN']]  ,
#             'JPY' :[JPY for _, JPY in market_dict['JPY']]  ,
#             'LUACTRUU' :[ LUACTRUU for _, LUACTRUU in market_dict['LUACTRUU']]  ,
#             'LF94TRUU' :[LF94TRUU for _, LF94TRUU in market_dict['LF94TRUU']]  ,
#             'EMUSTRUU' :[EMUSTRUU for _, EMUSTRUU in market_dict['EMUSTRUU']]  ,
#             'MXIN' :[MXIN for _, MXIN in market_dict['MXIN']]  ,
#             'LF98TRUU' :[LF98TRUU for _, LF98TRUU in market_dict['LF98TRUU']]  ,
#             'MXUS' :[MXUS for _, MXUS in market_dict['MXUS']]  ,
#             'LG30TRUU' :[ LG30TRUU for _, LG30TRUU in market_dict['LG30TRUU']]  ,
#             'LP01TREU' :[  LP01TREU for _, LP01TREU in market_dict['LP01TREU']]  ,
#             'MXEU' :[ MXEU for _, MXEU in market_dict['MXEU']]  ,
#             'MXJP' :[ MXJP for _, MXJP in market_dict['MXJP']]
#       }
      input_dict = {
            'Y' :market_dict['Y'] ,
            'year' : market_dict['Data'],
            'VIX' :market_dict['VIX']  ,
            'GTITL2YR' :market_dict['GTITL2YR']  ,
            'GTITL10YR' :market_dict['GTITL10YR']  ,
            'GTITL30YR' :market_dict['GTITL30YR']  ,
            'EONIA' :market_dict['EONIA']  ,
            'GTDEM30Y' :market_dict['GTDEM30Y']  ,
            'GTDEM10Y' :market_dict['GTDEM10Y']  ,
            'GTJPY10YR' :market_dict['GTJPY10YR']  ,
            'GTDEM2Y' :market_dict['GTDEM2Y']  ,
            'GTJPY30YR' :market_dict['GTJPY30YR']  ,
            'GTJPY2YR' :market_dict['GTJPY2YR']  ,
            'DXY' :market_dict['DXY']  ,
            'GTGBP20Y' :market_dict['GTGBP20Y']  ,
            'GTGBP30Y' :market_dict['GTGBP30Y']  ,
            'GTGBP2Y' :market_dict['GTGBP2Y']  ,
            'USGG30YR' :market_dict['USGG30YR']  ,
            'US0001M' :market_dict['US0001M']  ,
            'GT10' :market_dict['GT10']  ,
            'XAU BGNL' :market_dict['XAU BGNL']  ,
            'USGG3M' :market_dict['USGG3M']  ,
            'USGG2YR' :market_dict['USGG2YR']  ,
            'MXBR' :market_dict['MXBR']  ,
            'Cl1' :market_dict['Cl1']  ,
            'CRY' :market_dict['CRY']  ,
            'BDIY' :market_dict['BDIY']  ,
            'ECSURPUS' :market_dict['ECSURPUS']  ,
            'GBP' :market_dict['GBP']  ,
            'LUMSTRUU' :market_dict['LUMSTRUU']  ,
            'LMBITR' :market_dict['LMBITR']  ,
            'MXRU' :market_dict['MXRU']  ,
            'MXCN' :market_dict['MXCN']  ,
            'JPY' :market_dict['JPY']  ,
            'LUACTRUU' :market_dict['LUACTRUU']  ,
            'LF94TRUU' :market_dict['LF94TRUU']  ,
            'EMUSTRUU' :market_dict['EMUSTRUU']  ,
            'MXIN' :market_dict['MXIN']  ,
            'LF98TRUU' :market_dict['LF98TRUU']  ,
            'MXUS' :market_dict['MXUS']  ,
            'LG30TRUU' :market_dict['LG30TRUU']  ,
            'LP01TREU' :market_dict['LP01TREU']  ,
            'MXEU' :market_dict['MXEU']  ,
            'MXJP' :market_dict['MXJP']
      }
      data = pd.DataFrame([input_dict])
#       print("df")
#       print(df)
      return data

# Function to get predictions
def get_predictions(stock):
      preprocessed_data = preprocess_data(stock)
      prediction = lr_model_scaled.predict(preprocessed_data[stock])
      probability = lr_model_scaled.predict_proba(preprocessed_data[stock])
      # pred ={
      #       "dt": round(np.mean(dt_model.predict(df[stock]))),
      #       "lr": round(np.mean(lr_model.predict(df[stock]))),
      #       "rf": round(np.mean(rf_model.predict(df[stock]))),
      #       "dt_scaled": round(np.mean(dt_model_scaled.predict(df[stock]))),
      #       "lr_scaled": round(np.mean(lr_model_scaled.predict(df[stock]))),
      #       "rf_scaled": round(np.mean(rf_model_scaled.predict(df[stock]))),
      # }
      # probs ={
      #       "dt": dt_model.predict_proba(df[stock])[0][1],
            # "lr": lr_model.predict_proba(df[stock])[0][1],
            # "rf": rf_model.predict_proba(df[stock])[0][1],
            # "dt_scaled": dt_model_scaled.predict_proba(df[stock])[0][1],
            # "lr_scaled": lr_model_scaled.predict_proba(df[stock])[0][1],
            # "rf_scaled": rf_model_scaled.predict_proba(df[stock])[0][1],
      # }
      # avg_pred = round(np.mean(list(pred.values())))
      # avg_probs = round(np.mean(list(probs.values())))
      # return avg_pred,avg_probs
      # prediction = lr_model_scaled.predict(df[stock])
      # probability = lr_model_scaled.predict_proba(df[stock])
      # return prediction, probability
      # prediction = lr_model_scaled.predict(df["Y"])
      # probability = lr_model_scaled.predict_proba(df["Y"])
      # print(preprocess_data(df.iloc[-1]))
      return prediction, probability
      # return df

@app.get("/")
async def greeting():
      return "hello"

@app.post("/predict")
async def predict(stock:dict):
      prediction, probability = get_predictions(df.iloc[-1])
      return {
            "prediction": prediction.tolist(),
            "probability": probability.tolist(),
      }
      # print(preprocess_data(df.iloc[-1]))
      # return {
      #       "greet": "hello"
      # }
      # return df.iloc[-1]

if __name__ == "__main__":
      import uvicorn
      # uvicorn.run(app,port=8000)