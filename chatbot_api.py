# src/chatbot_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import datetime

app = FastAPI()
logs = []

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query):
    if "formation" in query.question.lower():
        answer = "Les formations disponibles sont Data Basics et Web Starter."
    else:
        answer = "Je suis un chatbot citoyen, posez-moi vos questions !"
    
    logs.append({
        "question": query.question,
        "answer": answer,
        "time": datetime.datetime.now()
    })
    return {"answer": answer}

@app.get("/stats")
def stats():
    # Regrouper par minute
    df = {}
    for log in logs:
        minute = log["time"].strftime("%H:%M")
        df[minute] = df.get(minute, 0) + 1
    return df