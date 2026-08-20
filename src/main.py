from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers.ia_router import ia_router
from src.routers.prediccion_router import prediccion_router
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Servicio de Extracción AI con Gemini",
    version="4.0.0"
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
    return {"message": "API de extracción de contratos con Gemini activa."}

app.include_router(ia_router, prefix="/ia", tags=["Inteligencia Artificial - Extracción de Contratos"])
app.include_router(prediccion_router, prefix="/predictions", tags=["Predicciones Temporales"])

#