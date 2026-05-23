from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks
from src.services.ia_service import IAService
import uuid

ia_router = APIRouter()

tareas: dict = {}

@ia_router.post("/procesar-contrato", status_code=200)
async def procesar_contrato_con_ia(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Formato invalido ({file.content_type}).")

    tarea_id = str(uuid.uuid4())
    tareas[tarea_id] = {"estado": "procesando", "resultado": None, "error": None}
    imagen_bytes = await file.read()

    async def procesar():
        try:
            resultado = await IAService.procesar_imagen_contrato(imagen_bytes)
            tareas[tarea_id]["estado"] = "listo"
            tareas[tarea_id]["resultado"] = resultado
        except Exception as e:
            tareas[tarea_id]["estado"] = "error"
            tareas[tarea_id]["error"] = str(e)

    background_tasks.add_task(procesar)
    return {"tarea_id": tarea_id}

@ia_router.get("/tarea/{tarea_id}")
def consultar_tarea(tarea_id: str):
    if tarea_id not in tareas:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    return tareas[tarea_id]
