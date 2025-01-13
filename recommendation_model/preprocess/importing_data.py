# -*- coding: utf-8 -*-

from pymongo import MongoClient
import pandas as pd

cluster = MongoClient('mongodb+srv://bhnas:harry228@cluster0.b26eduu.mongodb.net/development')
db = cluster['development']
collection = db['users']
data = collection.find()


list_01 = []

for doc in data:
    list_01.append(doc)
    
df = pd.DataFrame(list_01)


