from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
#Backend Database
import sqlite3

#Recommendation systems
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re
import json


def cleanResume(resumeText):
    resumeText = re.sub('\r\n', ' ', resumeText) # remove blank spaces
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-/:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) # non ascii values
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    resumeText = resumeText.lower() # convert to lower case
    return resumeText

def data_loader():
    data = pd.read_csv('dataset/filtere_dice_jobs_dataset.csv', low_memory=False)
    data['skills'] = data['skills'].apply(lambda x:cleanResume(x))
    data['jobtitle'] = data['jobtitle'].apply(lambda x:cleanResume(x))
    return data


def skills(df,skill_data):
      indexx = []
      id = []
      job_title = []
      score = []
      score_list = []
      dataset = pd.DataFrame(df.skills)
      dataset.loc[len(dataset)] = skill_data
      
      
      tfidf = TfidfVectorizer()

      tfidf_matrix = tfidf.fit_transform(dataset.skills)
      indice = pd.Series(dataset.index, index=dataset['skills'])
      cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

      similarity_scores = pd.DataFrame(cosine_sim[len(dataset)-1], columns=["score"]) 
      movie_indices = similarity_scores.sort_values("score", ascending=False)[0:10].index 

      result1 = dataset['skills'].iloc[movie_indices]
      result2 = similarity_scores.iloc[movie_indices]

      
      id.append(result1.index)
      for indx in range(1,len(id[0])):
        indexx.append(id[0][indx])

      score.append(result2.values.tolist())
      for sc in range(1,len(score[0])):
        score_list.append(score[0][sc])


      for i in range(0,len(indexx)):
        data_cell_2 = df.at[indexx[i], "jobtitle"]                   
        job_title.append(data_cell_2)

      #outcome = pd.DataFrame({'Id':indexx, 'JD':job_title, 'Scores':score_list})
      outcome = pd.DataFrame({'Id':indexx, 'JD':job_title})
      return outcome



def skill_list(df,num):
      print("=====>")
      data_skill = df.at[num, "skills"]
      words = data_skill.split(' ')
      skill_full = " ".join(set(words))

      return skill_full


def db_connect():
    con = sqlite3.connect("demo.db")
    cur = con.cursor()
    return con, cur
    

def db_insert(user_id,data):
    con, cur = db_connect()
    cur.executemany("INSERT INTO user VALUES(?, ?)", data)
    con.commit()

def db_retrieve(user_id):
    con, cur = db_connect()
    res = cur.execute("SELECT skills from user")
    return res.fetchall()



app = FastAPI()


@app.post("/Login/")
async def create_item(skill:str=''):
    user_id = 1 
    df = data_loader()
    data = [(user_id,skill)]
    db_insert(user_id,data)
    db_list = db_retrieve(user_id)
    return skills(df,skill).to_json(orient ='split')


@app.post("/recommendation/")
async def create_item(number:int=''):
    user_id = 1 
    df = data_loader()
    skill = skill_list(df,number)
    print(skill)
    data = [(user_id,skill)]
    db_insert(user_id,data)
    db_list = db_retrieve(user_id)

    return skills(df,skill).to_json(orient ='split')
