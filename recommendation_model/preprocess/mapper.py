from sklearn.base import TransformerMixin , BaseEstimator
from recommendation_model.config.core import config

class Mapper(TransformerMixin , BaseEstimator):
    def __init__(self,variables,mappings):
        if not isinstance(variables, list):
            raise ValueError('variables should be in list')
            
        self.variables = variables
        self.mappings = mappings
        
    def fit(self,x,y=None):
        return self
    
    def transform(self,x):
        x = x.copy()
        
        for var in self.variables:
            
            x[var] = x[var].map(self.mappings)
            
        return x
    
    
