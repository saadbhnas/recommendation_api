# -*- coding: utf-8 -*-
from sklearn.base import TransformerMixin , BaseEstimator




class Replacing(TransformerMixin , BaseEstimator):
    def __init__(self,variables,to_replace,value):
        if not isinstance(variables, list):
            raise ValueError('variables should be in list')
            
        self.variables = variables    
        self.to_replace = to_replace
        self.value = value
        
    def fit(self,x,y=None):
        return self
    
    def transform(self,x):
        
        
        for var in self.variables:
            
            x[var] = x[var].replace(self.to_replace,self.value)
            
        return x

    
    

