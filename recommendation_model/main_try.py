import os
import requests

def download_file():
    # Define the URL of the file and the local path to save it
    file_url = "https://raw.githubusercontent.com/saadbhnas/api-deployment/master/recommendation_model/dataset/movies_metadata.csv"
    file_path = "recommendation_model/dataset/movies_metadata.csv"

    # Check if the file already exists
    if not os.path.exists(file_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        print("Downloading the file...")
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"File downloaded and saved to {file_path}")
        else:
            raise Exception(f"Failed to download file: {response.status_code} {response.text}")
    else:
        print(f"File already exists at {file_path}")

# Call this function at runtime (e.g., during app startup or endpoint call)
download_file()


