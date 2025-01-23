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



@app.get("/download")
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
async def title_t(payload: dict = Body(...)):
    try:
        df = pd.read_csv(r'recommendation_model/dataset/movies_metadata.csv' , low_memory=False)
        # Ensure required columns exist
        print(df)
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
