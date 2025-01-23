from fastapi import FastAPI
import requests

app = FastAPI()

@app.on_event("startup")
async def download_csv():
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
