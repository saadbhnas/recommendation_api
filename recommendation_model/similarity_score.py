# -*- coding: utf-8 -*-
import pandas as pd
from recommendation_model import __version__ as _version
from recommendation_model.config.core import dataset_folder , config
from recommendation_model.preprocess.data_manager import load_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from recommendation_model.config.core import  trained_model_dir



df = pd.read_csv(dataset_folder/'movies_metadata.csv' , low_memory=False)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
recommedation_pipe = load_pipeline(file_name=pipeline_file_name)

recommedation_pipe_ = recommedation_pipe.fit_transform(df)

subset_size = 100
subset = recommedation_pipe_[:subset_size]

similarity_score = cosine_similarity(subset, recommedation_pipe_)

save_path = trained_model_dir / config.app_config.similiarity_score
joblib.dump(similarity_score, save_path)