from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.ia_router import ia_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Servicio de Extracción AI con Moondream",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
def read_root():
    return {"message": "El motor local de Moondream mediante Ollama está activo."}

app.include_router(ia_router, prefix="/ia", tags=["Inteligencia Artificial (VLM)"])