# -*- coding: utf-8 -*-
"""
the idea is function that extract certain key & value pairs from the json format 
which was preprocessed by the class handle json in prprocessing_json file 
the preprocessed cell will be in the python format list which have nested 
dict inside 

now thier are some columns which is direct dict and some columns have the 
following format [{},{},{}]

so i need function or class that take the column and then filter using 
isinstance maybe to get the list and ignore nons and we can make it 
like chose if we are dealing with dict or list and thier is nested dict
inside 

"""
from sklearn.base import TransformerMixin , BaseEstimator
from recommendation_model.preprocess.precproessing_json import new_df
import numpy as np
from recommendation_model.config.core import config

class Extract_value(TransformerMixin , BaseEstimator):
    def __init__(self,variable,key,job_role= None ,key_n:str = None ):
        if not isinstance(variable,list):
            
            raise ValueError('variable should be a list') 
        self.variable = variable
        self.key = key
        self.job_role = job_role
        self.key_n = key_n
        
    def fit(self,x,y=None):
        return self
    
    def process_list(self,x):
        # this help process the cells that have list 
            
        R = []
        for i in x :
            R.append(i[self.key])
        return '|'.join(R)
      
                        
    def process_dict(self,x):
        #this help process the cells that have dictionary
        return x[self.key]
    
    def process_nons(self,x):
        #this help process the cells that have non_types
        return np.nan
        
    def handle_cell_job_role(self,x):
        if isinstance(x, list):
            for i in x:
                if i['department'] == self.job_role:
                    return i[self.key]
        else:
            return np.nan
                        
        
    
    def handle_cell(self,x):
        
           if  isinstance(x,list):
               return self.process_list(x)
            
           elif  isinstance(x,dict):
               return self.process_dict(x)
        
           else :
              return self.process_nons(x)
        
   
    def transform(self,x):
        
        x = x.copy()
        
        if self.job_role :
            
             for var in self.variable:
            
                 x[self.job_role] = x[var].apply(lambda x : self.handle_cell_job_role(x))
            
             return x
            
        else:
            
        
           for var in self.variable:
            
               x[var] = x[var].apply(lambda x : self.handle_cell(x))
                
           return x
'''                
class Extract_value(TransformerMixin , BaseEstimator):
    def __init__(self,variable,key,job_role= None ,key_n:str = None ):
        if not isinstance(variable,list):
            
            raise ValueError('variable should be a list') 
            
        self.variable = variable
        
        def fit(self,x,y=None):
            return self
'''     
        


"""
now another functionality need to be added to get name of certain job role 
within crew members 

to do this first need to add if check to see if thier is job role filter 

if none then we get names directly 

if provided then we see if the key for example would be department and the job
would be something like producer , writer and so on 

"""

EV = Extract_value(config.model_config.vars_extract_name,key='name')
df_extracted_name = EV.fit_transform(new_df)


