import pandas as pd
from recommendation_model import __version__ as _version
from recommendation_model.config.core import dataset_folder , config
from recommendation_model.preprocess.data_manager import load_pipeline
from sklearn.metrics.pairwise import cosine_similarity

import joblib
from recommendation_model.config.core import  trained_model_dir

save_path = trained_model_dir / config.app_config.similiarity_score

df = pd.read_csv(dataset_folder/'movies_metadata.csv' , low_memory=False)
'''
pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
recommedation_pipe = load_pipeline(file_name=pipeline_file_name)

recommedation_pipe_ = recommedation_pipe.fit_transform(df)

similarity_score = cosine_similarity(recommedation_pipe_,recommedation_pipe_)
'''
similiarity_score = joblib.load(filename=save_path)

def make_prediction(similiarity_score=similiarity_score):
    
    title_to_index = pd.Series(df.index , index=df.title).to_dict()
    
    movie_title = input('Enter movie title')
    
    idx = title_to_index[movie_title]
    
    similar_movies = list(enumerate(similiarity_score[idx]))
    
    sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1] , reverse=True)
    
    sim_scosorted_similar_moviesres = sorted_similar_movies[1:6]
    
    movie_indices = [i[0] for i in sim_scosorted_similar_moviesres]
    
    similar_titles = df['title'].iloc[movie_indices]
    

    return similar_titles


predictions = make_prediction()

print(predictions)

df[df['title'] == 'Last Man Standing']['release_date']
