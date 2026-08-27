import os
import json
import logging
import numpy as np
import joblib
from typing import Dict, Any, List, Optional
from fastapi import HTTPException

logger = logging.getLogger("fastapi")


class DemandaService:
    _modelo = None
    _metadata: Dict[str, Any] = {}
    _cargado: bool = False
    _PATH_MODELOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelos")

    @classmethod
    def _cargar_modelo(cls):
        if cls._cargado:
            return

        modelo_path = os.path.join(cls._PATH_MODELOS, "modelo_demanda_rf.pkl")
        metadata_path = os.path.join(cls._PATH_MODELOS, "demanda_metadata.json")

        if not os.path.exists(modelo_path):
            raise HTTPException(status_code=500, detail="Modelo de demanda no encontrado. Ejecute el notebook de entrenamiento.")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=500, detail="Metadata de demanda no encontrada.")

        cls._modelo = joblib.load(modelo_path)
        with open(metadata_path, encoding="utf-8") as f:
            cls._metadata = json.load(f)

        cls._cargado = True
        logger.info("Modelo de demanda RF cargado exitosamente.")

    @staticmethod
    def _construir_features_demanda(
        cat_activa: str,
        cat_features: List[str],
        mes: int,
        anio: int,
        t: int,
        lag_1: float = 0,
        lag_2: float = 0,
        lag_3: float = 0,
        rolling_mean_3: float = 0,
    ) -> np.ndarray:
        features = {}
        for cat in cat_features:
            features[f"cat_{cat}"] = 1 if cat == cat_activa else 0
        features["mes"] = mes
        features["anio"] = anio
        features["t"] = t
        features["lag_1"] = lag_1
        features["lag_2"] = lag_2
        features["lag_3"] = lag_3
        features["rolling_mean_3"] = rolling_mean_3

        all_feature_names = DemandaService._metadata.get("features", [])
        row = []
        for feat in all_feature_names:
            row.append(features.get(feat, 0))
        return np.array([row])

    @staticmethod
    def predecir(stock_actual: Optional[Dict[str, int]] = None, meses: int = 6) -> dict:
        DemandaService._cargar_modelo()

        modelo = DemandaService._modelo
        metadata = DemandaService._metadata
        cat_features = metadata.get("cat_features", [])
        precio_promedio = metadata.get("precio_promedio_categoria", {})
        proporciones = metadata.get("proporciones_modelo_especifico", {})

        ultimo_periodo = metadata.get("ultimo_periodo_entrenado", "2026-02")
        year, month = map(int, ultimo_periodo.split("-"))

        num_categorias = len(cat_features)
        num_periodos_entrenamiento = metadata.get("num_periodos_entrenamiento", 40)
        t_base = num_periodos_entrenamiento

        demanda_por_categoria = {}
        predicciones_por_categoria = {}

        for cat in cat_features:
            preds = []
            for i in range(meses):
                m = month + i + 1
                y = year
                while m > 12:
                    m -= 12
                    y += 1

                lag_1 = preds[-1] if len(preds) >= 1 else 0
                lag_2 = preds[-2] if len(preds) >= 2 else 0
                lag_3 = preds[-3] if len(preds) >= 3 else 0

                if len(preds) >= 3:
                    rolling_mean_3 = float(np.mean(preds[-3:]))
                elif len(preds) > 0:
                    rolling_mean_3 = float(np.mean(preds))
                else:
                    rolling_mean_3 = 0

                features_row = DemandaService._construir_features_demanda(
                    cat, cat_features,
                    mes=m, anio=y, t=t_base + i,
                    lag_1=lag_1, lag_2=lag_2, lag_3=lag_3,
                    rolling_mean_3=rolling_mean_3,
                )
                pred = max(0, float(modelo.predict(features_row)[0]))
                preds.append(pred)

            predicciones_por_categoria[cat] = preds
            demanda_por_categoria[cat] = round(np.mean(preds), 1)

        desglose_por_modelo = {}
        for cat, cantidad in demanda_por_categoria.items():
            dist = proporciones.get(cat, {})
            desglose_por_modelo[cat] = [
                {"modelo": modelo_name, "cantidad": round(cantidad * pct, 1)}
                for modelo_name, pct in dist.items()
            ]

        monto_esperado_total = 0
        demanda_categoria_list = []
        for cat, cantidad in demanda_por_categoria.items():
            precio = precio_promedio.get(cat, 0)
            monto = round(cantidad * precio, 2)
            monto_esperado_total += monto
            demanda_categoria_list.append({
                "categoria": cat,
                "cantidad_predicha": cantidad,
                "precio_promedio": round(precio, 2),
                "monto_esperado": monto
            })

        alertas = []
        if stock_actual:
            for cat, demanda in demanda_por_categoria.items():
                stock = stock_actual.get(cat, 0)
                punto_reorden = demanda * 1.2
                if stock < punto_reorden:
                    alertas.append({
                        "categoria": cat,
                        "stock_actual": stock,
                        "demanda_predicha": demanda,
                        "unidades_a_comprar": round(punto_reorden - stock, 1)
                    })

        start_month = month + meses
        start_year = year
        while start_month > 12:
            start_month -= 12
            start_year += 1

        return {
            "periodo_inicio": ultimo_periodo,
            "meses": meses,
            "demanda_por_categoria": demanda_categoria_list,
            "desglose_por_modelo": desglose_por_modelo,
            "monto_esperado_total": round(monto_esperado_total, 2),
            "alertas_reorden": alertas
        }
