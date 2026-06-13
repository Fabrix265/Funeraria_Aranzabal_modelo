from fastapi import APIRouter, UploadFile, File, HTTPException
from src.services.ia_service import IAService
from src.schemas.ia import TranscripcionContratoOut

ia_router = APIRouter()


@ia_router.post("/process-contract", response_model=TranscripcionContratoOut, status_code=200)
async def procesar_contrato_con_ia(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Formato invalido ({file.content_type}).")

    imagen_bytes = await file.read()
    resultado = await IAService.procesar_imagen_contrato(imagen_bytes)
    return resultado
