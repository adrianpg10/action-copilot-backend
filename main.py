from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import json
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os 
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class AnalyzeRequest(BaseModel):
    text: str
    url: str = ""

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Eres un asistente que analiza páginas web y devuelve acciones concretas.
Responde SOLO con JSON válido, sin texto extra, con esta estructura exacta:
{
  "summary": "resumen en 1 frase",
  "actions": ["acción 1", "acción 2", "acción 3"],
  "next_step": "la acción más importante a hacer ahora"
}"""
            },
            {
                "role": "user",
                "content": f"Analiza este contenido de una web:\n\n{req.text[:3000]}"
            }
        ],
        temperature=0.3
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return data