import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import requests


def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding



df=joblib.load('embeddings.joblib')


incoming_query=input("Ask a Question : ")
question_embedding=create_embedding([incoming_query])[0]
#print(question_embedding)

# Find Question Embeddings with other embeddings 
# print(np.vstack(df['embedding'].values))  #np.vstack makes the embeddings 2d
# print(np.vstack(df['embedding']).shape)
similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
print(similarities)
top_results=2
max_indx=similarities.argsort()[::-1][0:top_results] #this will make the sorted indices of cosine similarities and make it revese and only top 3
print(max_indx)
new_df=df.loc[max_indx]
print(new_df[["title","number","text"]])