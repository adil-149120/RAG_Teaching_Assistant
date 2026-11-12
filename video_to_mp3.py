#This converts the videofiles to mp3.
#We will use the resultant mp3 files as the input for the whishper(to avoid loads on whisper)
import os
import subprocess
vdo_files=os.listdir("Rag-sample_videos")
for file in vdo_files:
    print(file)
    tutorial_no=file.split("#")[1].split(" ")[0]
    file_name=file.split(" ｜ ")[0]
    course_name=file.split(" ｜ ")[1].split(" - ")[0]
    print(tutorial_no,file_name,course_name,sep='\t')
    print("\n")
    #subprocess.run(["ffmpeg", "-i", f"Rag-sample_videos/{file}", f"Audios/{tutorial_no}_{file_name}.mp3"])