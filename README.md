
# How to use this RAG AI Teaching Assistant on your own data(videos):
## Step - 1: Collect Your Videos
Move all your video files to the video folder

## step - 2: Convert to mp3
Convert all the video file to mp3 by running 'video_to_mp3.py'

## step - 3: Convert mp3 to json
Convert all the mp3 files to json by running 'mp3_to_json.py'

## step - 4: Convert json files to vectors
Use the file 'preprocess_json.py' to convert the json files to a dataframe with embeddings & save it as a joblib pickle

## step - 5: Prompt generation and feeding to LLM
Read the joblib file and load it to the memroy. Then create a relevent prompt as per the user query and feed it to the LLM
