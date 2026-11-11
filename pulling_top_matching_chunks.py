import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding


jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])
       
    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk) 
        if(i==5): #read 5 chunks for now
            break
    break
# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
print(df)
incoming_query=input("Ask a Question : ")
question_embedding=create_embedding([incoming_query])[0]
#print(question_embedding)

# Find Question Embeddings with other embeddings 
# print(np.vstack(df['embedding'].values))  #np.vstack makes the embeddings 2d
# print(np.vstack(df['embedding']).shape)
similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
print(similarities)
top_results=3
max_indx=similarities.argsort()[::-1][0:top_results] #this will make the sorted indices of cosine similarities and make it revese and only top 3
print(max_indx)
new_df=df.loc[max_indx]
print(new_df[["title","number","text"]])
