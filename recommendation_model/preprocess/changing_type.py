# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin , BaseEstimator


class Changing_type(TransformerMixin , BaseEstimator):
    def __init__(self,variables,target_type):
        if not isinstance(variables, list):
            raise ValueError('variables should be in list')
            
        self.variables = variables    
        self.target_type = target_type
        
        
    def fit(self,x,y=None):
        return self
    
    def transform(self,x):
        x = x.copy()
        
        for var in self.variables:
            
            x[var] = x[var].astype(self.target_type)
            
        return x
    
    
