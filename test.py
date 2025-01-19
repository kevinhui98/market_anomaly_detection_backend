from fastapi import FastAPI
import pickle
import pandas as pd

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

df = pd.read_csv("FinancialMarketData.xlsx - EWS.csv")
def preprocess_data(market_dict):
      input_dict = {
            'year' : [years for _, years in market_dict['Data']],
            'Y' :[ Y for _, Y in market_dict['Y']]  ,
            'VIX' :[ VIX for _, VIX in market_dict['VIX']]  ,
            'GTITL2YR' :[ GTITL2YR for _, GTITL2YR in market_dict['GTITL2YR']]  ,
            'GTITL10YR' :[ GTITL10YR for _, GTITL10YR in market_dict['GTITL10YR']]  ,
            'GTITL30YR' :[ GTITL30YR for _, GTITL30YR in market_dict['GTITL30YR']]  ,
            'EONIA' :[ EONIA for _, EONIA in market_dict['EONIA']]  ,
            'GTDEM30Y' :[ GTDEM30Y for _, GTDEM30Y in market_dict['GTDEM30Y']]  ,
            'GTDEM10Y' :[ GTDEM10Y for _, GTDEM10Y in market_dict['GTDEM10Y']]  ,
            'GTJPY10YR' :[ GTJPY10YR for _, GTJPY10YR in market_dict['GTJPY10YR']]  ,
            'GTDEM2Y' :[ GTDEM2Y for _, GTDEM2Y in market_dict['GTDEM2Y']]  ,
            'GTJPY30YR' :[ GTJPY30YR for _, GTJPY30YR in market_dict['GTJPY30YR']]  ,
            'GTJPY2YR' :[ GTJPY2YR for _, GTJPY2YR in market_dict['GTJPY2YR']]  ,
            'DXY' :[ DXY for _, DXY in market_dict['DXY']]  ,
            'GTGBP20Y' :[ GTGBP20Y for _, GTGBP20Y in market_dict['GTGBP20Y']]  ,
            'GTGBP30Y' :[ GTGBP30Y for _, GTGBP30Y in market_dict['GTGBP30Y']]  ,
            'GTGBP2Y' :[ GTGBP2Y for _, GTGBP2Y in market_dict['GTGBP2Y']]  ,
            'USGG30YR' :[ USGG30YR for _, USGG30YR in market_dict['USGG30YR']]  ,
            'US0001M' :[ US0001M for _, US0001M in market_dict['US0001M']]  ,
            'GT10' :[ GT10 for _, GT10 in market_dict['GT10']]  ,
            'XAU_BGNL' :[ XAU_BGNL for _, XAU_BGNL in market_dict['XAU_BGNL']]  ,
            'USGG3M' :[ USGG3M for _, USGG3M in market_dict['USGG3M']]  ,
            'USGG2YR' :[USGG2YR for _, USGG2YR in market_dict['USGG2YR']]  ,
            'MXBR' :[MXBR for _, MXBR in market_dict['MXBR']]  ,
            'Cl1' :[Cl1 for _, Cl1 in market_dict['Cl1']]  ,
            'CRY' :[CRY for _, CRY in market_dict['CRY']]  ,
            'BDIY' :[BDIY for _, BDIY in market_dict['BDIY']]  ,
            'ECSURPUS' :[ECSURPUS for _, ECSURPUS in market_dict['ECSURPUS']]  ,
            'GBP' :[GBP for _, GBP in market_dict['GBP']]  ,
            'LUMSTRUU' :[ LUMSTRUU for _, LUMSTRUU in market_dict['LUMSTRUU']]  ,
            'LMBITR' :[LMBITR for _, LMBITR in market_dict['LMBITR']]  ,
            'MXRU' :[MXRU for _, MXRU in market_dict['MXRU']]  ,
            'MXCN' :[MXCN for _, MXCN in market_dict['MXCN']]  ,
            'JPY' :[JPY for _, JPY in market_dict['JPY']]  ,
            'LUACTRUU' :[ LUACTRUU for _, LUACTRUU in market_dict['LUACTRUU']]  ,
            'LF94TRUU' :[LF94TRUU for _, LF94TRUU in market_dict['LF94TRUU']]  ,
            'EMUSTRUU' :[EMUSTRUU for _, EMUSTRUU in market_dict['EMUSTRUU']]  ,
            'MXIN' :[MXIN for _, MXIN in market_dict['MXIN']]  ,
            'LF98TRUU' :[LF98TRUU for _, LF98TRUU in market_dict['LF98TRUU']]  ,
            'MXUS' :[MXUS for _, MXUS in market_dict['MXUS']]  ,
            'LG30TRUU' :[ LG30TRUU for _, LG30TRUU in market_dict['LG30TRUU']]  ,
            'LP01TREU' :[  LP01TREU for _, LP01TREU in market_dict['LP01TREU']]  ,
            'MXEU' :[ MXEU for _, MXEU in market_dict['MXEU']]  ,
            'MXJP' :[ MXJP for _, MXJP in market_dict['MXJP']]
      }
      df = pd.DataFrame([input_dict])
      print("df")
      print(df)
      return df

# Function to get predictions
def get_predictions(stock):
      # preprocess_data = preprocess_data(dict)
      # prediction = lr_model_scaled.predict(preprocess_data[stock])
      # probability = lr_model_scaled.predict_proba(preprocess_data[stock])
      prediction = lr_model_scaled.predict(df[stock])
      probability = lr_model_scaled.predict_proba(df[stock])
      return prediction, probability
@app.post("/predict")
async def predict(stock):
      prediction, probability = get_predictions(stock)
      return {
            "prediction": prediction.tolist(),
            "probability": probability.tolist(),
      }

if __name__ == "__main__":
      import uvicorn
      uvicorn.run(app,host="0.0.0.0",port=10000)