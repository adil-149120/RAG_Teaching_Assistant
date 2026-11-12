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

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream" : False
    })

    response = r.json()
    return response 




df=joblib.load('embeddings.joblib')


incoming_query=input("Ask a Question : ")
question_embedding=create_embedding([incoming_query])[0]
#print(question_embedding)

# Find Question Embeddings with other embeddings 
# print(np.vstack(df['embedding'].values))  #np.vstack makes the embeddings 2d
# print(np.vstack(df['embedding']).shape)
similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
#       print(similarities)
top_results=5
max_indx=similarities.argsort()[::-1][0:top_results] #this will make the sorted indices of cosine similarities and make it revese and only top 3
#       print(max_indx)
new_df=df.loc[max_indx]
#       print(new_df[["title","number","text"]])



prompt=f'''I am teaching web development in my Sigma web development course. Here are the video subtitle chunks containing video title, video number, start time in seconds, end time in seconds,
the text at that time:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
--------------------------------------------------------------------------
"{incoming_query}"
User asked this question related to video chunks, you have to answer in a human way(don't mention the above format it's just for you) 
where and how much content is taught in which video (in which video and in what timestamp) and guide the user to go to that particular 
video. If user asks unrelated question, tell him that you can only answer questions related to the course 
(don't repeat the question asked by the user and dont unnecessarily mention anything if not asked also not the last line just give the answer in about 50 words not more than 70 words)
'''

with open("prompt.txt","w") as f:
    f.write(prompt)

response=inference(prompt)["response"]
print("\nResponse : ",response)

with open("response.txt","w") as f:
    f.write(response)

# for index,item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

    