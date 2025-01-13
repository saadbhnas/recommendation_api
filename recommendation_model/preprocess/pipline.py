# -*- coding: utf-8 -*-
"""
now for pipeline the steps will be as follow:-
1-handle json format 
2-extracting useful information from columns 
3-drop useless columns 
4-replacing string inside float type columns 
5-changing type of some columns 
6-train model
"""
#ddsdsdsds


from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from feature_engine.selection import DropFeatures
from recommendation_model.preprocess.precproessing_json import Handle_json
from recommendation_model.preprocess.changing_type import Changing_type
from recommendation_model.preprocess.content import CreatingContent
from recommendation_model.preprocess.helper_functions import Extract_value
from recommendation_model.preprocess.replacing import Replacing
from recommendation_model.config.core import config , dataset_folder





recommendation_pipe = Pipeline([
    (
     "handling json",
      Handle_json(variable=config.model_config.movies_metadata_json_variables)     
     ),
    (
     "extracting useful information from columns",
     Extract_value(
         variable=config.model_config.vars_extract_name,
         key='name')
     ),
    (
     "drop useless columns",
     DropFeatures(features_to_drop=config.model_config.variables_drop)
     ),
    (
     "replacing string inside float type overview",
     Replacing(
         variables=[config.model_config.variables_to_replace[0]],
         to_replace=config.model_config.overview_replace,
         value='missing')
     ),
    (
     "replacing string inside float type budget",
     Replacing(
         variables=[config.model_config.variables_to_replace[2]],
         to_replace=config.model_config.strings_to_replace_budget,
         value=np.nan)
     ),
    (
     "replacing string inside float type popularity",
     Replacing(
         variables=[config.model_config.variables_to_replace[1]],
         to_replace=config.model_config.popularity_replace,
         value=np.nan)
     ),
    (
     "changing type of some columns",
     Changing_type(
         variables=config.model_config.changing_vars_type,
         target_type=float)
     ),
    (
     "training model",
     CreatingContent(variables=config.model_config.creating_content_column)
     )
    
    ])

df = pd.read_csv(dataset_folder/'movies_metadata.csv')

movies_metadata_cleaned = recommendation_pipe.fit_transform(df)