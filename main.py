from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class AnalyzeRequest(BaseModel):
    text: str
    url: str = ""

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    prompt = f"""Eres un asistente que analiza páginas web y devuelve acciones concretas.
Responde SOLO con JSON válido, sin texto extra, sin markdown, sin backticks, con esta estructura exacta:
{{
  "summary": "resumen en 1 frase",
  "actions": ["acción 1", "acción 2", "acción 3"],
  "next_step": "la acción más importante a hacer ahora"
}}

Analiza este contenido de una web:

{req.text[:3000]}"""

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    raw = response.text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return data