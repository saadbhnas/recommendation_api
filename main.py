# -*- coding: utf-8 -*-

from fastapi import FastAPI
#from recommendation_model.predict import make_prediction
from pydantic import BaseModel
from fastapi.params import Body

class Title(BaseModel):
    movie_title:str

app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}





import joblib
from recommendation_model.config.core import  trained_model_dir
from recommendation_model.config.core import  dataset_folder,config
import pandas as pd


save_path = trained_model_dir / config.app_config.similiarity_score
similiarity_score = joblib.load(filename=save_path)
df = pd.read_csv(dataset_folder/'movies_metadata.csv' , low_memory=False)

@app.post("/title")
async def title(payload:dict=Body(...)):
    
    title_to_index = pd.Series(df.index , index=df.title).to_dict()
    
    movie_title = payload['movie_title']
    
    idx = title_to_index[movie_title]
    
    similar_movies = list(enumerate(similiarity_score[idx]))
    
    sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1] , reverse=True)
    
    sim_scosorted_similar_moviesres = sorted_similar_movies[1:6]
    
    movie_indices = [i[0] for i in sim_scosorted_similar_moviesres]
    
    similar_titles = df['title'].iloc[movie_indices]
    
    
    return {"data" : similar_titles}


"""
@app.get("/predict")
def predict():
    
    predictions = make_prediction()
    
    return {"similar_movies" : predictions}
"""

#hh