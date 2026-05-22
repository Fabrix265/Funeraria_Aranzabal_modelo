from fastapi import APIRouter, UploadFile, File, HTTPException, status
from src.schemas.ia import TranscripcionContratoOut
from src.services.ia_service import IAService

ia_router = APIRouter()

@ia_router.post("/procesar-contrato", response_model=TranscripcionContratoOut, status_code=status.HTTP_200_OK)
async def procesar_contrato_con_ia(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato inválido ({file.content_type}). Sube una imagen PNG, JPG o JPEG."
        )
    try:
        imagen_bytes = await file.read()
        return await IAService.procesar_imagen_contrato(imagen_bytes)
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inesperado al procesar el archivo: {str(e)}"
        )