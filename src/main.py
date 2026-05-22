from fastapi import FastAPI
from src.routers.ia_router import ia_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Servicio de Extracción AI con Moondream",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "El motor local de Moondream mediante Ollama está activo."}

app.include_router(ia_router, prefix="/ia", tags=["Inteligencia Artificial (VLM)"])
