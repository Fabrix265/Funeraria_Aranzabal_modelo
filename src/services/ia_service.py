import io
import json
import logging
import base64
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any
import httpx

from PIL import Image, ImageOps, ImageFilter
Image.MAX_IMAGE_PIXELS = None

from fastapi import HTTPException, status

logger = logging.getLogger("fastapi")

PROMPT_CONTRATO = """Eres un asistente especializado en leer contratos funerarios escaneados.
El contrato tiene DOS zonas claramente distintas que NO debes confundir:

ZONA 1 - MEMBRETE (IGNORAR para extracción de datos):
- Es el encabezado superior con el nombre y logo de la funeraria.
- Contiene la dirección y teléfonos DE LA EMPRESA (ej: "Av. Condorcanqui...", "044-679338").
- NUNCA uses estos datos para rellenar campos del contrato.

ZONA 2 - CUERPO DEL CONTRATO (ÚNICA fuente válida):
- Empieza donde dice "CONTRATO" y tiene campos con líneas para rellenar.
- Campos escritos a mano o mecanografiados por el cliente:
* "Trujillo, __ de __ del 20__" → fecha
* "Señor(a):" → contratante_nombre
* "Teléfono:" → contratante_telefono (junto al nombre del contratante)
* "Doc. Identidad:" → contratante_dni
* "Dirección:" → direccion_velacion (dirección del cliente/velatorio, NO de la empresa)
* "Oxiso:" o "Occiso:" → fallecido_nombre
* "Velatorio:" → puede complementar direccion_velacion si dice "SU CASA" u otro lugar
* "Forma de Pago:" → tipo_pago
* Tabla "COSTO DEL SERVICIO": filas Ataúd, Capilla ardiente, Carroza, Carroza para flores, Cargadores → extraer modelo/descripción y cantidades
* "TOTAL" al pie → costo

REGLAS ESTRICTAS:
- Devuelve ÚNICAMENTE el JSON sin ningún texto adicional, sin markdown, sin ```.
- Lee cada campo con atención, no inventes datos.
- Para "tipo_pago" usa SOLO una de estas palabras exactas: directo, seguro, mixto.
Si dice "al contado", "efectivo", "dinero", "cheque", "particular" usa "directo".
Si dice "seguro", "aseguradora" usa "seguro".
Si es combinación usa "mixto". Si no está claro usa null.
- Para "ids_vehiculos_detectados" usa SOLO estos valores exactos: porta_ataud, porta_flores, mixto, auto, microbus.
Detecta si hay descripción escrita en las filas "Carroza" (→ porta_ataud) y "Carroza para flores" (→ porta_flores).
Si no hay nada escrito en esas filas, usa [].
- Para "cantidad_cargadores" lee la fila "Cargadores" de la tabla. Solo acepta 4, 6 o null.
- Para "fecha" combina día, mes y año escritos en el campo "Trujillo, __ de __ del 20__". Formato YYYY-MM-DD.
- Para "contratante_dni" lee el campo "Doc. Identidad:" del cuerpo del contrato. Extrae exactamente 8 dígitos. Si no son 8 dígitos usa null.
- Para "contratante_telefono" lee el campo "Teléfono:" que está en la misma línea o muy cerca del campo "Señor(a):".
IGNORA los teléfonos del membrete superior de la empresa (044-679338, 943441226, 980494319, 044-564963).
- Para "direccion_velacion" lee ÚNICAMENTE el campo etiquetado como "Dirección:" en el cuerpo del contrato.
El campo "Velatorio:" es DIFERENTE y debe ser IGNORADO completamente para este dato.
- Para "ataud_modelo" lee la descripción escrita en la fila "Ataúd" de la tabla COSTO DEL SERVICIO.
- Para "ataud_color" extrae el color si está mencionado junto al modelo del ataúd.
- Para "capilla_modelo" lee la descripción escrita en la fila "Capilla ardiente".
- Para "costo" lee el valor numérico del campo "TOTAL" al pie del contrato.
- Si un campo no aparece o no puedes leerlo con certeza usa null.

Estructura JSON exacta:
{
"fecha": "YYYY-MM-DD o null",
"contratante_nombre": "nombre completo en mayusculas o null",
"contratante_dni": "exactamente 8 digitos o null",
"contratante_telefono": "solo digitos sin guiones ni espacios o null",
"fallecido_nombre": "nombre completo en mayusculas o null",
"direccion_velacion": "valor del campo Direccion del contrato o null",
"tipo_pago": "directo o seguro o mixto o null",
"ataud_modelo": "descripcion escrita en fila Ataud de la tabla o null",
"ataud_color": "color del ataud o null",
"capilla_modelo": "descripcion escrita en fila Capilla ardiente o null",
"ids_vehiculos_detectados": [],
"cantidad_cargadores": null,
"costo": 0.0
}"""


def preprocesar_imagen(imagen_bytes: bytes, max_dim: int = 1600) -> bytes:
    imagen = Image.open(io.BytesIO(imagen_bytes))
    imagen = imagen.convert("L").convert("RGB")
    ancho, alto = imagen.size

    bbox = imagen.convert("L").getbbox()
    if bbox:
        imagen = imagen.crop(bbox)
        ancho, alto = imagen.size

    if ancho > max_dim or alto > max_dim:
        escala = max_dim / max(ancho, alto)
        imagen = imagen.resize((int(ancho * escala), int(alto * escala)), Image.Resampling.LANCZOS)

    imagen = ImageOps.autocontrast(imagen, cutoff=1)
    imagen = imagen.filter(ImageFilter.SHARPEN)

    buffer = io.BytesIO()
    imagen.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def limpiar_y_parsear_json(contenido: str) -> Dict[str, Any]:
    contenido = contenido.strip()
    if "```" in contenido:
        lineas = [l for l in contenido.split("\n") if not l.strip().startswith("```")]
        contenido = "\n".join(lineas).strip()
    inicio = contenido.find("{")
    fin = contenido.rfind("}")
    if inicio != -1 and fin != -1:
        contenido = contenido[inicio:fin+1]
    return json.loads(contenido)


def normalizar_campos(datos: Dict[str, Any]) -> Dict[str, Any]:
    # --- FECHA: convertir "25 de Julio del 2025" → "2025-07-25" ---
    fecha_raw = str(datos.get("fecha") or "").strip()
    if fecha_raw and not fecha_raw.startswith("20"):
        meses = {
            "enero": "01", "febrero": "02", "marzo": "03", "abril": "04",
            "mayo": "05", "junio": "06", "julio": "07", "agosto": "08",
            "septiembre": "09", "octubre": "10", "noviembre": "11", "diciembre": "12"
        }
        import re
        match = re.search(r"(\d{1,2})\s+de\s+(\w+)\s+del?\s+(\d{4})", fecha_raw, re.IGNORECASE)
        if match:
            dia, mes_str, anio = match.groups()
            mes = meses.get(mes_str.lower())
            if mes:
                datos["fecha"] = f"{anio}-{mes}-{dia.zfill(2)}"
            else:
                datos["fecha"] = None
        else:
            datos["fecha"] = None

    # --- DIRECCION: si el modelo devuelve un dict, extraer string útil ---
    direccion = datos.get("direccion_velacion")
    if isinstance(direccion, dict):
        # Intenta extraer algún valor string del dict
        for v in direccion.values():
            if isinstance(v, str) and len(v) > 5:
                datos["direccion_velacion"] = v
                break
        else:
            datos["direccion_velacion"] = None
    elif not isinstance(direccion, str):
        datos["direccion_velacion"] = None

    # --- TIPO PAGO ---
    tipo_raw = str(datos.get("tipo_pago") or "").lower().strip()
    if any(p in tipo_raw for p in ["seguro", "aseguradora", "poliza"]):
        datos["tipo_pago"] = "seguro"
    elif any(p in tipo_raw for p in ["mixto", "combinado", "parcial"]):
        datos["tipo_pago"] = "mixto"
    elif any(p in tipo_raw for p in ["directo", "efectivo", "dinero", "contado", "cheque", "particular"]):
        datos["tipo_pago"] = "directo"
    elif tipo_raw in ["directo", "seguro", "mixto"]:
        datos["tipo_pago"] = tipo_raw
    else:
        datos["tipo_pago"] = None

    # --- CARGADORES ---
    if datos.get("cantidad_cargadores") not in [4, 6, None]:
        datos["cantidad_cargadores"] = None

    # --- VEHÍCULOS ---
    validos = {"porta_ataud", "porta_flores", "mixto", "auto", "microbus"}
    raw = datos.get("ids_vehiculos_detectados", [])
    datos["ids_vehiculos_detectados"] = [v for v in raw if v in validos] if isinstance(raw, list) else []

    # --- DNI ---
    dni = str(datos.get("contratante_dni") or "").strip()
    if not (dni.isdigit() and len(dni) == 8):
        datos["contratante_dni"] = None

    # --- TELÉFONO ---
    telefono_raw = str(datos.get("contratante_telefono") or "").strip()
    telefono_limpio = "".join(filter(str.isdigit, telefono_raw))
    datos["contratante_telefono"] = telefono_limpio if telefono_limpio else None

    # --- COLOR ATAÚD ---
    if str(datos.get("ataud_color") or "").lower() in ["null", "none", ""]:
        datos["ataud_color"] = None

    return datos


class ExtractorIAInterface(ABC):
    @abstractmethod
    async def extraer_datos_contrato(self, imagen_bytes: bytes) -> Dict[str, Any]:
        pass


# ─────────────────────────────────────────────
# MODELO ACTIVO: minicpm-v:8b-2.6-q4_K_M
# 36%/64% CPU/GPU — ~3 minutos por contrato
# Mejor OCR de manuscritos en hardware actual
# ─────────────────────────────────────────────
class OllamaStrategy(ExtractorIAInterface):
    def __init__(self):
        self.OLLAMA_URL = "http://localhost:11434/api/chat"
        self.MODEL_NAME = "minicpm-v:8b-2.6-q4_K_M"
        logger.info(f"Estrategia Ollama inicializada con modelo: {self.MODEL_NAME}")

    async def extraer_datos_contrato(self, imagen_bytes: bytes) -> Dict[str, Any]:
        contenido = ""
        try:
            imagen_optimizada_bytes = preprocesar_imagen(imagen_bytes, max_dim=3000)
            imagen_b64 = base64.b64encode(imagen_optimizada_bytes).decode('utf-8')

            payload = {
                "model": self.MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_CONTRATO,
                        "images": [imagen_b64]
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0
                }
            }

            async with httpx.AsyncClient(timeout=1200.0) as client:  # 5 minutos
                response = await client.post(self.OLLAMA_URL, json=payload)
                response.raise_for_status()

                contenido = response.json()["message"]["content"]
                logger.info(f"Respuesta de {self.MODEL_NAME}:\n{contenido}")

                datos = limpiar_y_parsear_json(contenido)
                return normalizar_campos(datos)

        except json.JSONDecodeError:
            logger.error(f"JSON inválido recibido:\n{contenido}")
            raise HTTPException(status_code=422, detail="La IA no pudo estructurar la respuesta.")
        except Exception as e:
            logger.error(f"Error en procesamiento Ollama: {repr(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error en inferencia local: {repr(e)}")



class MockStrategy(ExtractorIAInterface):
    async def extraer_datos_contrato(self, imagen_bytes: bytes) -> Dict[str, Any]:
        import asyncio
        await asyncio.sleep(1)
        return {
            "fecha": "2025-07-25",
            "contratante_nombre": "VANESA ALEJANDRA MEDINA FLORES",
            "contratante_dni": "18112990",
            "contratante_telefono": "970248382",
            "fallecido_nombre": "ZORAIDA CAMPOS NORIEGA",
            "direccion_velacion": "Psj. Los Jilgueros 177 Urb. Los Pinos",
            "tipo_pago": "directo",
            "ataud_modelo": "Ataúd Americano Blanco con Adornos",
            "ataud_color": "Blanco",
            "capilla_modelo": "Capilla Ardiente Iluminada",
            "ids_vehiculos_detectados": ["porta_ataud"],
            "cantidad_cargadores": 4,
            "costo": 1750.00
        }


# ─────────────────────────────────────────────
# Cambia USE_MOCK = True para desarrollo rápido
# sin esperar inferencia del modelo
# ─────────────────────────────────────────────
USE_MOCK = False

_ollama_instancia = None
_mock_instancia = None


def obtener_extractor_ia() -> ExtractorIAInterface:
    global _ollama_instancia, _mock_instancia
    if USE_MOCK:
        if _mock_instancia is None:
            _mock_instancia = MockStrategy()
        return _mock_instancia
    else:
        if _ollama_instancia is None:
            _ollama_instancia = OllamaStrategy()
        return _ollama_instancia


class IAService:
    @staticmethod
    async def procesar_imagen_contrato(imagen_bytes: bytes) -> Dict[str, Any]:
        extractor = obtener_extractor_ia()
        return await extractor.extraer_datos_contrato(imagen_bytes)