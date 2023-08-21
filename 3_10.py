from fastapi import FastAPI, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import func
from sqlalchemy import create_engine
#from database import SessionLocal
from schema import PostGet
#from table_post import Post
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from random import randint
import random
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from category_encoders import PolynomialEncoder

import os
import pickle
import pandas as pd
from catboost import CatBoostClassifier

from sqlalchemy import Column, Integer, String, text, func, TIMESTAMP, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

file_name = "catboost_model"

class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.dropout = nn.Sequential(
            nn.Dropout(p=0.2)
        )
        n = 32
        self.block1 = nn.Sequential(
            nn.Linear(input_size, n), 
            nn.BatchNorm1d(n),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Linear(n, n * 2), 
            nn.BatchNorm1d(n * 2),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Linear(n * 2, n * 2), 
            nn.BatchNorm1d(n * 2),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Linear(n * 2, n * 4), 
            nn.BatchNorm1d(n * 4),
            nn.ReLU()
        )
        self.block5 = nn.Sequential(
            nn.Linear(n * 4, n * 4), 
            nn.BatchNorm1d(n * 4),
            nn.ReLU()
        )
        self.block6 = nn.Sequential(
            nn.Linear(n * 4, n * 8), 
            nn.BatchNorm1d(n * 8),
            nn.ReLU()
        )
        self.block7 = nn.Sequential(
            nn.Linear(n * 8, n * 8), 
            nn.BatchNorm1d(n * 8),
            nn.ReLU()
        )
        self.block8 = nn.Sequential(
            nn.Linear(n * 8, n * 16), 
            nn.BatchNorm1d(n * 16),
            nn.ReLU()
        )
        self.block9 = nn.Sequential(
            nn.Linear(n * 16, n * 16), 
            nn.BatchNorm1d(n * 16),
            nn.ReLU()
        )
        self.block10 = nn.Sequential(
            nn.Linear(n * 16, 1),
            nn.Softmax(dim=0)
            #nn.Sigmoid()
            #nn.Tanh()
        )

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.dropout(output)
        output = self.block3(output) + output
        output = self.dropout(output)
        output = self.block4(output)
        output = self.dropout(output)
        output = self.block5(output) + output
        output = self.dropout(output)
        output = self.block6(output)
        output = self.dropout(output)
        output = self.block7(output) + output
        output = self.dropout(output)
        output = self.block8(output)
        output = self.dropout(output)
        output = self.block9(output) + output
        output = self.dropout(output)
        output = self.block10(output)
        

        return output



def load_models(file_name):
    from_file = CatBoostClassifier()
    # LOAD MODEL HERE PLS :)
    loaded_model = from_file.load_model(get_model_path(file_name))
    return loaded_model

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_net(n):
    model = Net(n)
    model.load_state_dict(torch.load(get_model_path('net'), map_location=torch.device('cpu'))) #load_models(file_name)
    return model

app = FastAPI()

engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    

def get_user_data():
    user_data = batch_load_sql(f"SELECT * FROM public.user_data")

    for col in ['city', 'country', 'age']:
        codes = user_data[col].astype('category').cat.codes
        scaler = StandardScaler()
        user_data[col] = scaler.fit_transform(codes.to_frame())
    
    one_hot_columns = ['gender', 'os', 'exp_group', 'source']
 
    user_data[['exp_group', 'gender']] = user_data[['exp_group', 'gender']].astype('int') 
    dummies = pd.get_dummies(user_data[one_hot_columns].astype(object), drop_first=True)
    user_data[dummies.columns] = dummies
    user_data.drop(one_hot_columns, axis=1, inplace=True)

    kmeans = KMeans(n_clusters=5, n_init=100, random_state=0).fit(user_data.drop('user_id', axis=1))
    user_data['cluster'] = kmeans.predict(user_data.drop('user_id', axis=1))
    
    one_hot_columns = ['cluster']
    dummies = pd.get_dummies(user_data[one_hot_columns].astype(object), drop_first=True)
    user_data[dummies.columns] = dummies
    user_data.drop(one_hot_columns, axis=1, inplace=True)
    
        
    return user_data

def get_post_text():
    post_text = batch_load_sql(f"SELECT * FROM public.post_text_df")
    post_text = post_text.iloc[post_text.drop('post_id', axis=1).drop_duplicates().index.tolist()].dropna()
    
    
    
    embeddings = pd.read_sql('SELECT * FROM "denis21.97@mail.ru_lesson_22" WHERE emb_id IS NOT NULL', con=engine).dropna(axis=1)   
   
   
    
    embeddings.index = post_text.index
    post_text = pd.concat([post_text, embeddings.drop('emb_id', axis=1)], axis=1)
    
    
    
    one_hot_columns = ['topic']
    dummies = pd.get_dummies(post_text[one_hot_columns].astype(object), drop_first=True)
    post_text[dummies.columns] = dummies

    
    return post_text

user_data = get_user_data() 
post_text = get_post_text()   
   
model = load_models(file_name)
#net = load_net(user_data.shape[1] + post_text.shape[1] - 4)


@app.get("/post/recommendations/", response_model=List[PostGet])
def get_recommendations(id: int, time: datetime, limit: int = 5):
    
    device = 'cpu'
    user = user_data[user_data['user_id'] == id].dropna(axis=1)
    if user.shape[0] == 0:
        user = user_data[user_data['user_id'] == random.choice(user_data['user_id'].tolist())].dropna(axis=1)
 
    user = user.loc[user.index.repeat(post_text.shape[0])]
    user.index = post_text.index
    
  
    
    #net
    #input_df = pd.concat([post_text.drop(['text', 'topic', 'post_id'], axis=1), user.drop('user_id', axis=1)], axis=1)
    #input_df.index = [i for i in range(input_df.shape[0])]
    #input_df.columns = [i for i in range(input_df.shape[1])]
    #probs = pd.DataFrame(net(torch.tensor(input_df.values).float()).detach().numpy())
    #probs.index = post_text['post_id'].index
    #probs_posts = pd.concat([probs, post_text['post_id'].to_frame()], axis=1)
    #probs_posts.rename(columns = {0 : 'probs'}, inplace=True)
    #net
    
    
    #catboost
    input_df = pd.concat([post_text.drop(['text', 'post_id', 'topic'], axis=1), user.drop('user_id', axis=1)], axis=1)

    pred_prob = model.predict_proba(input_df)
    probs_posts = pd.DataFrame(model.predict_proba(input_df), post_text['post_id'], columns=['to_drop', 'probs']).drop('to_drop', axis=1) #catboost
    #catboost
   
    
    sorted_probs_posts = probs_posts.sort_values('probs', ascending=False)
 
    
    selected_posts = sorted_probs_posts.iloc[:limit].merge(post_text[['post_id', 'text', 'topic']], how='inner', on='post_id')

    result = [PostGet(**{"id": row['post_id'], "text": row['text'], "topic": row['topic']}) for index, row in selected_posts.iterrows()]

    if result is None:
        raise HTTPException(200)
    return result