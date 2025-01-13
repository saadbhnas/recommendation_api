from sklearn.base import TransformerMixin , BaseEstimator
#import re
import json
import numpy as np
import pandas as pd
from recommendation_model.config.core import dataset_folder , config
pd.options.display.max_columns = None



df = pd.read_csv(dataset_folder/'movies_metadata.csv'
                 , date_parser=True)

class Handle_json(TransformerMixin , BaseEstimator):
    def __init__(self,variable):
        if not isinstance(variable,list):
            raise ValueError('variable should be a list')
            
        self.variable = variable
        
    def fit(self,x,y=None):
        return self
        
    def Handle_json(self,x):
        #here we first make the the json foramt ready for parsying ito python 
        #variable x = dataframe 
        #x = self.handle_none_in_json(x)
        
        
        try :
            
            x = x.replace("'",'"')
            
            x = x.replace('None' , '"missing"')
        
            x = json.loads(x)
            
        except :
            
            x = np.nan
            
        return x
    
    
        
        
    
    
    def transform(self,x):
        
        x = x.copy()
        
        for var in self.variable:
            
            #x[var] = x[var].apply(lambda x : self.handle_none_in_json(x))
            
            x[var] = x[var].apply(lambda x : self.Handle_json(x))
            
        return x
    

hj = Handle_json(config.model_config.movies_metadata_json_variables)
new_df = hj.fit_transform(df)
