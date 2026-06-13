from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum


class ModeloEnum(str, Enum):
    sarima = "sarima"
    prophet = "prophet"
    xgboost = "xgboost"
    lgbm = "lgbm"
    lstm = "lstm"
    ets = "ets"


class TargetEnum(str, Enum):
    servicios_totales = "servicios_totales"
    monto_total = "monto_total"


class PrediccionRequest(BaseModel):
    modelo: ModeloEnum = Field(..., description="Modelo a usar para la predicción")
    target: TargetEnum = Field(..., description="Variable objetivo a predecir")
    pasos: int = Field(6, ge=1, le=24, description="Número de pasos a predecir (1-24 meses)")


class PrediccionResponse(BaseModel):
    modelo: str = Field(..., description="Nombre del modelo utilizado")
    target: str = Field(..., description="Variable objetivo predicha")
    pasos: int = Field(..., description="Número de pasos predecidos")
    periodo_inicio: str = Field(..., description="Período de inicio de la predicción (YYYY-MM)")
    predicciones: List[Dict[str, Any]] = Field(..., description="Lista de predicciones con mes y valor")


class ModeloInfoResponse(BaseModel):
    targets: List[str] = Field(..., description="Variables objetivo disponibles")
    modelos: List[str] = Field(..., description="Modelos disponibles")
    train_periodo: str = Field(..., description="Período de entrenamiento")
    test_periodo: str = Field(..., description="Período de prueba")
    train_months: int = Field(..., description="Meses de entrenamiento")
    test_months: int = Field(..., description="Meses de prueba")


class ComparativaResponse(BaseModel):
    train_periodo: str = Field(..., description="Período de entrenamiento")
    test_periodo: str = Field(..., description="Período de prueba")
    metricas: List[Dict[str, Any]] = Field(default=[], description="Métricas de comparación por modelo")


class HistoryResponse(BaseModel):
    target: str = Field(..., description="Variable objetivo")
    datos: List[Dict[str, Any]] = Field(..., description="Lista de datos históricos con mes y valor")


class DistribucionItem(BaseModel):
    nombre: str = Field(..., description="Nombre del modelo/tipo")
    proporcion: float = Field(..., description="Proporción promedio anual (0-1)")
    proporcion_estacional: Optional[Dict[str, float]] = Field(None, description="Proporción por mes (1-12)")
    cantidad_estimada: Optional[float] = Field(None, description="Cantidad estimada basada en el total de servicios")


class DistribucionResponse(BaseModel):
    total_servicios: int = Field(..., description="Total de servicios sobre el que se aplica la distribución")
    distribucion: List[DistribucionItem] = Field(..., description="Lista de items con proporción y cantidad estimada")


class DistribucionCompletaResponse(BaseModel):
    modelo: str = Field(..., description="Modelo utilizado")
    target: str = Field(..., description="Variable objetivo")
    pasos: int = Field(..., description="Número de pasos predecidos")
    periodo_inicio: str = Field(..., description="Período de inicio de la predicción (YYYY-MM)")
    predicciones: List[Dict[str, Any]] = Field(..., description="Predicciones con distribución de ataúdes y capillas")


class DistribucionRequest(BaseModel):
    modelo: ModeloEnum = Field(..., description="Modelo a usar para la predicción")
    target: TargetEnum = Field(..., description="Variable objetivo a predecir")
    mes_inicio: str = Field(..., description="Mes de inicio (YYYY-MM)")
    mes_fin: str = Field(..., description="Mes de fin (YYYY-MM)")
