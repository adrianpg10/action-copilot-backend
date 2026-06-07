from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class AnalyzeRequest(BaseModel):
    text: str
    url: str = ""

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": """Eres un asistente especializado en subvenciones y ayudas para autónomos y pymes en España y Europa.
Analiza el contenido de la página web y responde SOLO con JSON válido, sin texto extra, sin markdown, sin backticks:
{
  "summary": "resumen en 1 frase de qué trata esta página",
  "actions": ["subvención o ayuda detectada con importe si aparece", "requisito clave 1", "requisito clave 2"],
  "next_step": "el paso más concreto e inmediato para solicitar esta ayuda"
}
Si la página no contiene información sobre subvenciones o ayudas, devuelve:
{
  "summary": "Esta página no contiene información sobre subvenciones o ayudas",
  "actions": ["Prueba en sede.agenciatributaria.gob.es", "Consulta el BOE en boe.es", "Visita ipex.es para ayudas a exportación"],
  "next_step": "Navega a una página con información de subvenciones para analizar"
}"""
            },
            {
                "role": "user",
                "content": f"Analiza esta página:\n\n{req.text[:1500]}"
            }
        ]
    )

    raw = response.choices[0].message.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return data