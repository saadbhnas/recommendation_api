# -*- coding: utf-8 -*-

from sklearn.base import TransformerMixin , BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer





class CreatingContent(TransformerMixin,BaseEstimator):
    def __init__(self,variables):
        
        self.variables = variables
        
    def fit(self,x,y=None):
        return self
    
    def transform(self,x):
        
        x=x.copy()
        
        
        
        x['content'] = x[self.variables[0]].astype(str) + '' + x[self.variables[1]].astype(str) + '' + \
        x[self.variables[2]].astype(str)  + '' + x[self.variables[3]].astype(str) + '' + \
        x[self.variables[4]].astype(str) + '' + x[self.variables[5]].astype(str) + '' + \
        x[self.variables[6]].astype(str) + '' + x[self.variables[7]].astype(str) + '' + \
        x[self.variables[8]].astype(str) + '' + x[self.variables[9]].astype(str) + '' + \
        x[self.variables[10]].astype(str) + '' + x[self.variables[11]].astype(str) + '' + \
        x[self.variables[12]].astype(str) + '' + x[self.variables[13]].astype(str) + '' + \
        x[self.variables[14]].astype(str) + '' + x[self.variables[15]].astype(str)
        
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(x['content'])
        
        tfidf_transformer = TfidfTransformer()
        tfidf = tfidf_transformer.fit_transform(bow)
        
        
        return tfidf



