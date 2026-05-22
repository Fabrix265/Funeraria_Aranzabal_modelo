from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
from decimal import Decimal
from enum import Enum

class TipoPagoEnum(str, Enum):
    directo = "directo"
    seguro = "seguro"
    mixto = "mixto"

class TranscripcionContratoOut(BaseModel):
    fecha: Optional[date] = Field(None, description="Fecha extraída del contrato (YYYY-MM-DD)")
    contratante_nombre: Optional[str] = Field(None, description="Nombre completo del contratante / cliente")
    contratante_dni: Optional[str] = Field(None, description="DNI del contratante (8 dígitos)")
    contratante_telefono: Optional[str] = Field(None, description="Teléfono del contratante")
    fallecido_nombre: Optional[str] = Field(None, description="Nombre completo del fallecido")
    direccion_velacion: Optional[str] = Field(None, description="Dirección exacta extraída del campo Dirección")
    tipo_pago: Optional[TipoPagoEnum] = Field(None, description="Tipo de pago: directo, seguro o mixto")
    ataud_modelo: Optional[str] = Field(None, description="Modelo de ataúd")
    ataud_color: Optional[str] = Field(None, description="Color del ataúd si se especifica")
    capilla_modelo: Optional[str] = Field(None, description="Modelo de capilla ardiente (solo el nombre limpio)")
    ids_vehiculos_detectados: List[str] = Field(default=[], description="Lista de vehículos usados detectados")
    cantidad_cargadores: Optional[int] = Field(None, description="Cantidad de cargadores (número entero)")
    costo: Optional[Decimal] = Field(None, description="Monto del precio TOTAL")