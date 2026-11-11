import requests
import os
import json
import pandas as pd

def create_embedding(text_list):
    #https://github.com/ollama/ollama/blob/main/docs/api.md#generatre-embeddings
    r=requests.post("http://localhost:11434/api/embed",json={
        "model" : "bge-m3",
        "input" : text_list
    })
    embedding= r.json()["embeddings"]
    return embedding

jsons=os.listdir("jsons")     #Listed all the jsons
chunk_id=0

my_dict=[]
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content=json.load(f)
    print(f"Creating embeddings for {json_file}")
    embeddings=create_embedding([c["text"] for c in content["chunks"]])
    for i,chunk in enumerate(content["chunks"]):
        chunk["chunk_id"]=chunk_id
        chunk["embedding"]=embeddings[i]
        chunk_id+=1
        my_dict.append(chunk)
        if(i==5):
            break

    
#print(my_dict)

df=pd.DataFrame.from_records(my_dict)
print(df)


# text_list=create_embedding(["I am Adil Ahmed","I am a good boy"])
# print(text_list)