# -*- coding: utf-8 -*-
'''
from fastapi import FastAPI, HTTPException
#from recommendation_model.predict import make_prediction
from pydantic import BaseModel
from fastapi.params import Body
import joblib
from recommendation_model.config.core import  trained_model_dir
from recommendation_model.config.core import  config
import pandas as pd
import requests

save_path = trained_model_dir / config.app_config.similiarity_score
similiarity_score = joblib.load(filename=save_path)
#df = pd.read_csv(dataset_folder/'movies_metadata.csv' , low_memory=False)

class Title(BaseModel):
    movie_title:str
    
#df = pd.read_csv(r'recommendation_model/dataset/movies_metadata.csv' , low_memory=False)


app = FastAPI()



@app.get("/")
def dataframee():
    url = 'https://raw.githubusercontent.com/saadbhnas/api-deployment/master/recommendation_model/dataset/movies_metadata.csv'
    output_file = "recommendation_model/dataset/movies_metadata.csv"  # Path inside Railway
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("CSV file downloaded successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading CSV file: {e}")
    

# Replace with your file's raw URL or GitHub API download URL




@app.post("/title")
async def title(payload: dict = Body(...)):
    df = pd.read_csv("recommendation_model/dataset/movies_metadata.csv", low_memory=False)
    try:
        #df = pd.read_csv(dataset_folder/'movies_metadata.csv' , low_memory=False)
        # Ensure required columns exist
        required_columns = ['original_title', 'title']
        for col in required_columns:
            if col not in df.columns:
                #raise HTTPException(status_code=500, detail=f"Column '{col}' is missing in the DataFrame!")
                raise HTTPException(status_code=500, detail=f"Missing column '{col}'. Current columns: {df.columns}")

        # Create a title-to-index mapping
        title_to_index = pd.Series(df.index, index=df['title']).to_dict()

        # Extract the movie title from the request payload
        movie_title = payload.get('movie_title')
        if not movie_title:
            raise HTTPException(status_code=400, detail="The 'movie_title' key is missing in the request payload.")

        # Check if the movie title exists in the DataFrame
        if movie_title not in title_to_index:
            raise HTTPException(status_code=404, detail=f"Movie title '{movie_title}' not found in the dataset.")

        # Get the index of the movie
        idx = title_to_index[movie_title]

        # Calculate similarity scores for the movie
        similar_movies = list(enumerate(similiarity_score[idx]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

        # Select top 5 similar movies (excluding the movie itself)
        top_similar_movies = sorted_similar_movies[1:6]
        movie_indices = [i[0] for i in top_similar_movies]

        # Get the titles of similar movies
        similar_titles = df['title'].iloc[movie_indices].tolist()

        return {"data": similar_titles}

    except Exception as e:
        # Return a 500 error for unexpected issues
        raise HTTPException(status_code=500, detail=str(e))
'''
from fastapi import FastAPI, HTTPException, Body
from pathlib import Path
import pandas as pd
import requests
import logging

# Initialize FastAPI app
app = FastAPI()

# Logger for debugging
logger = logging.getLogger("uvicorn.error")

# Define paths and file download URL
dataset_path = Path("recommendation_model/dataset/movies_metadata.csv")
download_url = "https://raw.githubusercontent.com/saadbhnas/api-deployment/master/recommendation_model/dataset/movies_metadata.csv"

# Function to download the file if it doesn't exist
def download_file_if_missing():
    if not dataset_path.exists():
        try:
            logger.info(f"Dataset not found locally. Downloading from {download_url}...")
            dataset_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            response = requests.get(download_url, stream=True)
            response.raise_for_status()  # Raise an error for HTTP issues

            with open(dataset_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info("Dataset downloaded successfully.")
        except Exception as e:
            logger.error(f"Failed to download the dataset: {e}")
            raise HTTPException(status_code=500, detail="Failed to download the dataset.")

# Ensure the file is available at startup
download_file_if_missing()

# Load the dataset
try:
    df = pd.read_csv(dataset_path, low_memory=False)
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    raise HTTPException(status_code=500, detail="Failed to load the dataset.")

# Precompute the title-to-index mapping
required_columns = ['original_title', 'title']
for col in required_columns:
    if col not in df.columns:
        raise Exception(f"Missing column '{col}'. Current columns: {df.columns}")

title_to_index = pd.Series(df.index, index=df['title']).to_dict()

# Mock similarity scores (replace this with your actual similarity scores)
similarity_score = [[0.9] * len(df)]  # Example, replace with real data

@app.post("/title")
async def title(payload: dict = Body(...)):
    try:
        movie_title = payload.get('movie_title')
        if not movie_title:
            raise HTTPException(status_code=400, detail="The 'movie_title' key is missing in the request payload.")

        if movie_title not in title_to_index:
            raise HTTPException(status_code=404, detail=f"Movie title '{movie_title}' not found in the dataset.")

        idx = title_to_index[movie_title]
        similar_movies = list(enumerate(similarity_score[idx]))
        sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
        top_similar_movies = sorted_similar_movies[1:6]
        movie_indices = [i[0] for i in top_similar_movies]
        similar_titles = df['title'].iloc[movie_indices].tolist()

        return {"data": similar_titles}

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
