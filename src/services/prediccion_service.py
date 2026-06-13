import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, List
from fastapi import HTTPException

logger = logging.getLogger("fastapi")


class PrediccionService:
    _modelos: Dict[str, Any] = {}
    _scalers: Dict[str, Any] = {}
    _cargado: bool = False
    _metadata: Dict[str, Any] = {}
    _historical_values: Dict[str, List[float]] = {}
    _PATH_MODELOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modelos")
    _LAGS = [1, 2, 3, 6]
    _WINDOW_SIZE = 2

    @classmethod
    def _cargar_modelos(cls):
        if cls._cargado:
            return

        metadata_path = os.path.join(cls._PATH_MODELOS, "model_metadata.json")
        with open(metadata_path) as f:
            cls._metadata = json.load(f)

        cls._historical_values = cls._metadata.get("historical_values", {})

        for key in cls._metadata.get("saved", {}):
            pkl_path = os.path.join(cls._PATH_MODELOS, f"{key}.pkl")
            keras_path = os.path.join(cls._PATH_MODELOS, f"{key}.keras")

            if os.path.exists(keras_path):
                from tensorflow.keras.models import load_model
                cls._modelos[key] = load_model(keras_path)
                logger.info(f"Modelo Keras cargado: {key}")
            elif os.path.exists(pkl_path):
                cls._modelos[key] = joblib.load(pkl_path)
                logger.info(f"Modelo PKL cargado: {key}")

        for target in cls._metadata.get("targets", []):
            scaler_path = os.path.join(cls._PATH_MODELOS, f"scaler_{target}.pkl")
            if os.path.exists(scaler_path):
                cls._scalers[target] = joblib.load(scaler_path)
                logger.info(f"Scaler cargado: {target}")

        cls._cargado = True
        logger.info("Todos los modelos cargados exitosamente.")

    @staticmethod
    def obtener_info() -> dict:
        PrediccionService._cargar_modelos()
        m = PrediccionService._metadata
        return {
            "targets": m.get("targets", []),
            "modelos": ["sarima", "prophet", "xgboost", "lgbm", "lstm", "ets"],
            "train_periodo": m.get("train_period", ""),
            "test_periodo": m.get("test_period", ""),
            "train_months": m.get("train_months", 0),
            "test_months": m.get("test_months", 0),
        }

    @staticmethod
    def _predecir_sarima_ets(modelo: str, target: str, pasos: int) -> list:
        key = f"{modelo}_{target}"
        results = PrediccionService._modelos[key]
        pred = results.forecast(steps=pasos)
        return pred.values.tolist() if hasattr(pred, "values") else list(pred)

    @staticmethod
    def _predecir_prophet(target: str, pasos: int) -> list:
        key = f"prophet_{target}"
        model = PrediccionService._modelos[key]
        future = model.make_future_dataframe(periods=pasos, freq="MS")
        forecast = model.predict(future)
        return forecast["yhat"].values[-pasos:].tolist()

    @staticmethod
    def _predecir_ml_recursivo(modelo: str, target: str, pasos: int) -> list:
        key = f"{modelo}_{target}"
        model = PrediccionService._modelos[key]
        history = list(PrediccionService._historical_values.get(target, []))

        n_features = model.n_features_in_
        if n_features == 5:
            lags_config = [1, 2, 3, 6]
        else:
            lags_config = list(range(1, n_features))

        max_lag = max(lags_config)
        if len(history) < max_lag:
            raise HTTPException(
                status_code=400,
                detail=f"Historial insuficiente para {modelo}. Se necesitan al menos {max_lag} valores."
            )

        predictions = []
        current_history = list(history)

        for step in range(pasos):
            lags = [current_history[-lag] for lag in lags_config]
            mes_num = (step + 1) % 12 + 1
            features = np.array([lags + [mes_num]]).reshape(1, -1)
            pred = model.predict(features)[0]
            predictions.append(float(pred))
            current_history.append(float(pred))

        return predictions

    @staticmethod
    def _predecir_lstm(target: str, pasos: int) -> list:
        key = f"lstm_{target}"
        model = PrediccionService._modelos[key]
        scaler = PrediccionService._scalers.get(target)

        if scaler is None:
            raise HTTPException(
                status_code=400,
                detail=f"Scaler no encontrado para {target}."
            )

        history = list(PrediccionService._historical_values.get(target, []))

        if len(history) < PrediccionService._WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Historial insuficiente para LSTM. Se necesitan al menos {PrediccionService._WINDOW_SIZE} valores."
            )

        current_window = history[-PrediccionService._WINDOW_SIZE:]
        predictions = []

        for _ in range(pasos):
            window_array = np.array(current_window).reshape(1, PrediccionService._WINDOW_SIZE, 1)
            window_scaled = scaler.transform(window_array.reshape(-1, 1)).reshape(1, PrediccionService._WINDOW_SIZE, 1)
            pred_scaled = model.predict(window_scaled, verbose=0)
            pred_value = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions.append(float(pred_value))
            current_window = current_window[1:] + [float(pred_value)]

        return predictions

    @staticmethod
    def predecir(modelo: str, target: str, pasos: int) -> dict:
        PrediccionService._cargar_modelos()
        key = f"{modelo}_{target}"

        if key not in PrediccionService._modelos:
            raise HTTPException(status_code=404, detail=f"Modelo {key} no encontrado.")

        if modelo in ["sarima", "ets"]:
            pred_values = PrediccionService._predecir_sarima_ets(modelo, target, pasos)
        elif modelo == "prophet":
            pred_values = PrediccionService._predecir_prophet(target, pasos)
        elif modelo in ["xgboost", "lgbm"]:
            pred_values = PrediccionService._predecir_ml_recursivo(modelo, target, pasos)
        elif modelo == "lstm":
            pred_values = PrediccionService._predecir_lstm(target, pasos)
        else:
            raise HTTPException(status_code=400, detail=f"Modelo {modelo} no soportado.")

        test_period = PrediccionService._metadata.get("test_period", "")
        if test_period and "a" in test_period:
            last_test = test_period.split("a")[-1].strip()
            year, month = last_test.split("-")
            start_month = int(month) + 1
            start_year = int(year)
            if start_month > 12:
                start_month = 1
                start_year += 1
            fecha_inicio = f"{start_year}-{start_month:02d}"
        else:
            fecha_inicio = "2026-03"

        year, month = map(int, fecha_inicio.split("-"))
        fechas = pd.date_range(start=pd.Timestamp(year=year, month=month, day=1), periods=pasos, freq="MS")

        predicciones = [
            {"mes": f.strftime("%Y-%m"), "valor": round(float(v), 2)}
            for f, v in zip(fechas, pred_values)
        ]

        return {
            "modelo": modelo,
            "target": target,
            "pasos": pasos,
            "periodo_inicio": fecha_inicio,
            "predicciones": predicciones,
        }

    @staticmethod
    def obtener_comparativa() -> dict:
        PrediccionService._cargar_modelos()
        m = PrediccionService._metadata
        return {
            "train_periodo": m.get("train_period", ""),
            "test_periodo": m.get("test_period", ""),
            "metricas": m.get("comparison_metrics", []),
        }

    @staticmethod
    def obtener_historial(target: str) -> dict:
        PrediccionService._cargar_modelos()
        historical_data = PrediccionService._metadata.get("historical_data", {})
        if target not in historical_data:
            raise HTTPException(status_code=404, detail=f"No hay datos históricos para {target}")
        return {
            "target": target,
            "datos": historical_data[target],
        }

    @staticmethod
    def obtener_distribucion_ataudes() -> list:
        PrediccionService._cargar_modelos()
        dist = PrediccionService._metadata.get("distribucion_ataudes", {})
        
        total_historical = 0
        historical = PrediccionService._metadata.get("historical_data", {}).get("servicios_totales", [])
        for item in historical:
            total_historical += item.get("valor", 0)
        
        raw_items = []
        for nombre, meses in dist.items():
            if isinstance(meses, dict):
                peso = sum(meses.values())
                raw_items.append({"nombre": nombre, "peso": peso, "meses": meses})
            else:
                raw_items.append({"nombre": nombre, "peso": meses, "meses": None})
        
        total_pesos = sum(item["peso"] for item in raw_items) or 1
        
        result = []
        for item in raw_items:
            proporcion = item["peso"] / total_pesos
            meses_dict = item["meses"]
            meses_normalizados = None
            if meses_dict:
                meses_normalizados = {k: round(v / item["peso"], 4) if item["peso"] > 0 else 0 for k, v in meses_dict.items()}
            result.append({
                "nombre": item["nombre"],
                "proporcion": round(proporcion, 4),
                "proporcion_estacional": meses_normalizados,
                "cantidad_estimada": round(total_historical * proporcion, 1)
            })
        return result

    @staticmethod
    def obtener_distribucion_capillas() -> list:
        PrediccionService._cargar_modelos()
        dist = PrediccionService._metadata.get("distribucion_capillas", {})
        
        total_historical = 0
        historical = PrediccionService._metadata.get("historical_data", {}).get("servicios_totales", [])
        for item in historical:
            total_historical += item.get("valor", 0)
        
        raw_items = []
        for nombre, meses in dist.items():
            if isinstance(meses, dict):
                peso = sum(meses.values())
                raw_items.append({"nombre": nombre, "peso": peso, "meses": meses})
            else:
                raw_items.append({"nombre": nombre, "peso": meses, "meses": None})
        
        total_pesos = sum(item["peso"] for item in raw_items) or 1
        
        result = []
        for item in raw_items:
            proporcion = item["peso"] / total_pesos
            meses_dict = item["meses"]
            meses_normalizados = None
            if meses_dict:
                meses_normalizados = {k: round(v / item["peso"], 4) if item["peso"] > 0 else 0 for k, v in meses_dict.items()}
            result.append({
                "nombre": item["nombre"],
                "proporcion": round(proporcion, 4),
                "proporcion_estacional": meses_normalizados,
                "cantidad_estimada": round(total_historical * proporcion, 1)
            })
        return result

    @staticmethod
    def predecir_distribucion(modelo: str, target: str, mes_inicio: str, mes_fin: str) -> dict:
        PrediccionService._cargar_modelos()

        test_period = PrediccionService._metadata.get("test_period", "")
        if test_period and "a" in test_period:
            last_test = test_period.split("a")[-1].strip()
            year, month = last_test.split("-")
            start_month = int(month) + 1
            start_year = int(year)
            if start_month > 12:
                start_month = 1
                start_year += 1
            periodo_inicio = f"{start_year}-{start_month:02d}"
        else:
            periodo_inicio = "2026-03"

        y1, m1 = map(int, periodo_inicio.split("-"))
        y2, m2 = map(int, mes_fin.split("-"))
        pasos = (y2 - y1) * 12 + (m2 - m1) + 1
        pasos = max(pasos, 1)

        resultado = PrediccionService.predecir(modelo, target, pasos)
        dist_ataudes = PrediccionService._metadata.get("distribucion_ataudes", {})
        dist_capillas = PrediccionService._metadata.get("distribucion_capillas", {})

        predicciones_filtradas = []
        for pred in resultado["predicciones"]:
            mes = pred["mes"]
            if mes < mes_inicio or mes > mes_fin:
                continue

            total = pred["valor"]
            mes_num = int(mes.split("-")[1])

            ataudes_est = []
            for nombre, meses in dist_ataudes.items():
                if isinstance(meses, dict):
                    prop = meses.get(str(mes_num), 0)
                else:
                    prop = meses
                ataudes_est.append({"nombre": nombre, "proporcion": round(prop, 4), "cantidad_estimada": round(total * prop, 1)})

            capillas_est = []
            for nombre, meses in dist_capillas.items():
                if isinstance(meses, dict):
                    prop = meses.get(str(mes_num), 0)
                else:
                    prop = meses
                capillas_est.append({"nombre": nombre, "proporcion": round(prop, 4), "cantidad_estimada": round(total * prop, 1)})

            predicciones_filtradas.append({
                "mes": mes,
                "total_servicios": round(total, 1),
                "ataudes": ataudes_est,
                "capillas": capillas_est,
            })

        return {
            "modelo": modelo,
            "target": target,
            "pasos": pasos,
            "periodo_inicio": periodo_inicio,
            "predicciones": predicciones_filtradas,
        }
