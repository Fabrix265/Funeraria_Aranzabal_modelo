from fastapi import APIRouter
from src.schemas.prediccion import DemandaRequest, DemandaResponse
from src.services.demanda_service import DemandaService

demanda_router = APIRouter()


@demanda_router.post("/demanda", response_model=DemandaResponse)
def predecir_demanda(request: DemandaRequest):
    return DemandaService.predecir(
        stock_actual=request.stock_actual,
        meses=request.meses
    )
