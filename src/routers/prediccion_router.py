from fastapi import APIRouter
from src.schemas.prediccion import (
    PrediccionRequest,
    PrediccionResponse,
    ModeloInfoResponse,
    ComparativaResponse,
    HistoryResponse,
    DistribucionResponse,
    DistribucionCompletaResponse,
    DistribucionRequest,
)
from src.services.prediccion_service import PrediccionService

prediccion_router = APIRouter()


@prediccion_router.get("/models", response_model=ModeloInfoResponse)
def listar_modelos():
    return PrediccionService.obtener_info()


@prediccion_router.post("/predict", response_model=PrediccionResponse)
def predecir(request: PrediccionRequest):
    return PrediccionService.predecir(request.modelo.value, request.target.value, request.pasos)


@prediccion_router.get("/compare", response_model=ComparativaResponse)
def comparar_modelos():
    return PrediccionService.obtener_comparativa()


@prediccion_router.get("/history/{target}", response_model=HistoryResponse)
def obtener_historial(target: str):
    return PrediccionService.obtener_historial(target)


@prediccion_router.get("/distribution/coffins", response_model=DistribucionResponse)
def listar_distribucion_coffins():
    return {
        "total_servicios": 1,
        "distribucion": PrediccionService.obtener_distribucion_ataudes(),
    }


@prediccion_router.get("/distribution/chapels", response_model=DistribucionResponse)
def listar_distribucion_chapels():
    return {
        "total_servicios": 1,
        "distribucion": PrediccionService.obtener_distribucion_capillas(),
    }


@prediccion_router.post("/distribution/predict", response_model=DistribucionCompletaResponse)
def predecir_distribucion(request: DistribucionRequest):
    return PrediccionService.predecir_distribucion(
        request.modelo.value, request.target.value,
        request.mes_inicio, request.mes_fin
    )
