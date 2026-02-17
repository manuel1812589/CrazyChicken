import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

MODELS_DIR = "app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_ANOMALY_VENTAS = os.path.join(MODELS_DIR, "anomaly_ventas.pkl")
MODEL_ANOMALY_TICKET = os.path.join(MODELS_DIR, "anomaly_ticket.pkl")
MODEL_ANOMALY_CANTIDAD = os.path.join(MODELS_DIR, "anomaly_cantidad.pkl")
MODEL_FORECAST_VENTAS = os.path.join(MODELS_DIR, "forecast_ventas.pkl")


def _build_anomaly_pipeline(contamination: float = 0.1) -> Pipeline:
    """
    Construye un pipeline: StandardScaler → IsolationForest.
    contamination = fracción esperada de anomalías (0.05 a 0.20).
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "iso",
                IsolationForest(
                    n_estimators=200,
                    contamination=contamination,
                    random_state=42,
                    warm_start=False,
                ),
            ),
        ]
    )


def _save_model(model, path: str) -> None:
    joblib.dump(model, path)


def _load_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _flag_anomalies(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Devuelve array bool: True = anomalía detectada.
    IsolationForest retorna -1 (anomalía) o +1 (normal).
    """
    preds = pipeline.predict(X)
    return preds == -1


def _anomaly_score(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """
    Score de anomalía normalizado [0, 1].
    Valores más altos = más anómalo.
    """
    raw = pipeline.named_steps["iso"].score_samples(
        pipeline.named_steps["scaler"].transform(X)
    )
    shifted = -raw
    min_v, max_v = shifted.min(), shifted.max()
    if max_v == min_v:
        return np.zeros(len(shifted))
    return (shifted - min_v) / (max_v - min_v)


def entrenar_detector_ventas(df: pd.DataFrame) -> dict:
    """
    Entrena IsolationForest con ventas_mes + ticket_promedio + cantidad_ventas.
    Persiste el modelo en disco.

    Parámetros
    ----------
    df : DataFrame con columnas ['ventas_mes', 'ticket_promedio', 'cantidad_ventas']

    Retorna
    -------
    dict con métricas del entrenamiento.
    """
    features = ["ventas_mes", "ticket_promedio", "cantidad_ventas"]
    df_clean = df[features].dropna()

    if len(df_clean) < 6:
        return {"error": "Se necesitan al menos 6 meses de datos para entrenar."}

    X = df_clean.values
    pipeline = _build_anomaly_pipeline(contamination=0.1)
    pipeline.fit(X)
    _save_model(pipeline, MODEL_ANOMALY_VENTAS)

    anomalias = _flag_anomalies(pipeline, X)
    return {
        "muestras_entrenadas": len(df_clean),
        "anomalias_detectadas_en_train": int(anomalias.sum()),
        "modelo_guardado": MODEL_ANOMALY_VENTAS,
        "fecha_entrenamiento": datetime.now().isoformat(),
    }


def detectar_anomalias_ventas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el detector de ventas al DataFrame.
    Si no existe modelo entrenado, entrena con los mismos datos.

    Retorna el DataFrame original con columnas adicionales:
        - es_anomalia   (bool)
        - score_anomalia (float 0-1)
        - razon_anomalia (str)
    """
    features = ["ventas_mes", "ticket_promedio", "cantidad_ventas"]
    df_out = df.copy()

    pipeline = _load_model(MODEL_ANOMALY_VENTAS)
    if pipeline is None:
        entrenar_detector_ventas(df)
        pipeline = _load_model(MODEL_ANOMALY_VENTAS)

    df_feat = df_out[features].fillna(df_out[features].median())
    X = df_feat.values

    df_out["es_anomalia"] = _flag_anomalies(pipeline, X)
    df_out["score_anomalia"] = _anomaly_score(pipeline, X)
    df_out["razon_anomalia"] = df_out.apply(
        lambda row: _explicar_anomalia_ventas(row, df_out), axis=1
    )
    return df_out


def _explicar_anomalia_ventas(row: pd.Series, df: pd.DataFrame) -> str:
    """Genera una explicación legible de por qué el mes es anómalo."""
    if not row["es_anomalia"]:
        return ""

    razones = []
    media_ventas = df["ventas_mes"].mean()
    std_ventas = df["ventas_mes"].std()
    media_ticket = df["ticket_promedio"].mean()
    std_ticket = df["ticket_promedio"].std()
    media_cant = df["cantidad_ventas"].mean()
    std_cant = df["cantidad_ventas"].std()

    if abs(row["ventas_mes"] - media_ventas) > 1.5 * std_ventas:
        direccion = "alta" if row["ventas_mes"] > media_ventas else "baja"
        razones.append(f"venta mensual inusualmente {direccion}")

    if abs(row["ticket_promedio"] - media_ticket) > 1.5 * std_ticket:
        direccion = "alto" if row["ticket_promedio"] > media_ticket else "bajo"
        razones.append(f"ticket promedio inusualmente {direccion}")

    if abs(row["cantidad_ventas"] - media_cant) > 1.5 * std_cant:
        direccion = "alta" if row["cantidad_ventas"] > media_cant else "baja"
        razones.append(f"cantidad de ventas inusualmente {direccion}")

    return "; ".join(razones) if razones else "combinación inusual de indicadores"


def detectar_anomalias_ticket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detector específico para ticket_promedio + ventas_totales + cantidad_ventas.
    """
    features = ["ticket_promedio", "ventas_totales", "cantidad_ventas"]
    df_out = df.copy()

    pipeline = _load_model(MODEL_ANOMALY_TICKET)
    if pipeline is None:
        df_feat = df_out[features].dropna()
        if len(df_feat) >= 6:
            p = _build_anomaly_pipeline(0.1)
            p.fit(df_feat.values)
            _save_model(p, MODEL_ANOMALY_TICKET)
            pipeline = p
        else:
            df_out["es_anomalia"] = False
            df_out["score_anomalia"] = 0.0
            df_out["razon_anomalia"] = ""
            return df_out

    df_feat = df_out[features].fillna(df_out[features].median())
    X = df_feat.values

    df_out["es_anomalia"] = _flag_anomalies(pipeline, X)
    df_out["score_anomalia"] = _anomaly_score(pipeline, X)
    df_out["razon_anomalia"] = df_out.apply(
        lambda row: (
            _generar_razon_simple(row, df_out, "ticket_promedio", "ticket")
            if row["es_anomalia"]
            else ""
        ),
        axis=1,
    )
    return df_out


def detectar_anomalias_cantidad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detector específico para cantidad_ventas + ventas_acumulada.
    """
    features_base = ["cantidad_ventas"]
    if "cantidad_acumulada" in df.columns:
        features_base.append("cantidad_acumulada")

    df_out = df.copy()

    pipeline = _load_model(MODEL_ANOMALY_CANTIDAD)
    if pipeline is None:
        df_feat = df_out[features_base].dropna()
        if len(df_feat) >= 6:
            p = _build_anomaly_pipeline(0.1)
            p.fit(df_feat.values)
            _save_model(p, MODEL_ANOMALY_CANTIDAD)
            pipeline = p
        else:
            df_out["es_anomalia"] = False
            df_out["score_anomalia"] = 0.0
            df_out["razon_anomalia"] = ""
            return df_out

    df_feat = df_out[features_base].fillna(df_out[features_base].median())
    X = df_feat.values

    df_out["es_anomalia"] = _flag_anomalies(pipeline, X)
    df_out["score_anomalia"] = _anomaly_score(pipeline, X)
    df_out["razon_anomalia"] = df_out.apply(
        lambda row: (
            _generar_razon_simple(row, df_out, "cantidad_ventas", "cantidad de ventas")
            if row["es_anomalia"]
            else ""
        ),
        axis=1,
    )
    return df_out


def _generar_razon_simple(
    row: pd.Series, df: pd.DataFrame, col: str, label: str
) -> str:
    media = df[col].mean()
    std = df[col].std()
    if std == 0:
        return "valor inusual"
    z = (row[col] - media) / std
    if z > 1.5:
        return f"{label} excepcionalmente alta (+{z:.1f}σ)"
    elif z < -1.5:
        return f"{label} excepcionalmente baja ({z:.1f}σ)"
    return "combinación inusual de indicadores"


def entrenar_forecast_ventas(df: pd.DataFrame) -> dict:
    """
    Entrena regresión lineal sobre ventas_mes para proyectar tendencia.
    Usa índice temporal (1, 2, 3, ...) como feature.
    """
    df_clean = df[["ventas_mes"]].dropna().reset_index(drop=True)

    if len(df_clean) < 4:
        return {"error": "Se necesitan al menos 4 meses para proyectar tendencia."}

    X = np.arange(1, len(df_clean) + 1).reshape(-1, 1)
    y = df_clean["ventas_mes"].values

    model = LinearRegression()
    model.fit(X, y)
    _save_model(model, MODEL_FORECAST_VENTAS)

    r2 = model.score(X, y)
    pendiente = model.coef_[0]

    return {
        "r2": round(r2, 4),
        "tendencia_mensual": round(pendiente, 2),
        "direccion": "creciente" if pendiente > 0 else "decreciente",
        "modelo_guardado": MODEL_FORECAST_VENTAS,
    }


def predecir_ventas_proximos_meses(df: pd.DataFrame, n_meses: int = 3) -> list[dict]:
    """
    Proyecta ventas para los próximos n_meses.

    Retorna lista de dicts:
        [{"mes": 13, "ventas_proyectadas": 85000, "intervalo_inferior": ..., "intervalo_superior": ...}]
    """
    model = _load_model(MODEL_FORECAST_VENTAS)
    if model is None:
        entrenar_forecast_ventas(df)
        model = _load_model(MODEL_FORECAST_VENTAS)

    n_actual = len(df.dropna(subset=["ventas_mes"]))

    df_clean = df[["ventas_mes"]].dropna().reset_index(drop=True)
    X_hist = np.arange(1, len(df_clean) + 1).reshape(-1, 1)
    residuals = df_clean["ventas_mes"].values - model.predict(X_hist)
    std_error = np.std(residuals)

    resultados = []
    for i in range(1, n_meses + 1):
        idx = np.array([[n_actual + i]])
        pred = model.predict(idx)[0]
        resultados.append(
            {
                "mes_futuro": n_actual + i,
                "ventas_proyectadas": round(pred, 2),
                "intervalo_inferior": round(pred - 1.28 * std_error, 2),
                "intervalo_superior": round(pred + 1.28 * std_error, 2),
            }
        )

    return resultados


def generar_recomendaciones_ml(
    df_ventas: pd.DataFrame,
    meta_mensual: float,
    n_forecast: int = 3,
) -> dict:
    """
    Pipeline completo de ML que genera recomendaciones automáticas.

    1. Detecta anomalías en ventas
    2. Proyecta próximos meses
    3. Analiza tendencia y cumplimiento de meta
    4. Retorna recomendaciones estructuradas listas para pasar a generar_respuesta_chat()

    Parámetros
    ----------
    df_ventas   : DataFrame con columnas ventas_mes, ticket_promedio, cantidad_ventas, nombre_mes
    meta_mensual: meta de ventas en pesos
    n_forecast  : número de meses a proyectar

    Retorna
    -------
    dict con:
        - anomalias      : lista de meses anómalos con razón
        - forecast       : proyección de próximos meses
        - recomendaciones: lista de strings con acciones concretas
        - contexto_ia    : str listo para pasar a generar_respuesta_chat()
    """
    resultado = {
        "anomalias": [],
        "forecast": [],
        "recomendaciones": [],
        "contexto_ia": "",
    }

    df_con_anomalias = detectar_anomalias_ventas(df_ventas)
    meses_anomalos = df_con_anomalias[df_con_anomalias["es_anomalia"]]

    for _, row in meses_anomalos.iterrows():
        resultado["anomalias"].append(
            {
                "mes": row.get("nombre_mes", "Desconocido"),
                "ventas": row.get("ventas_mes", 0),
                "score": round(row["score_anomalia"], 3),
                "razon": row["razon_anomalia"],
            }
        )

    info_modelo = entrenar_forecast_ventas(df_ventas)
    if "error" not in info_modelo:
        resultado["forecast"] = predecir_ventas_proximos_meses(df_ventas, n_forecast)

    ventas_recientes = df_ventas["ventas_mes"].iloc[-3:].mean()
    ventas_anteriores = (
        df_ventas["ventas_mes"].iloc[:-3].mean()
        if len(df_ventas) > 3
        else ventas_recientes
    )
    tendencia_reciente = (
        (ventas_recientes - ventas_anteriores) / ventas_anteriores * 100
        if ventas_anteriores != 0
        else 0
    )

    cumplimiento_meta = (df_ventas["ventas_mes"] >= meta_mensual).mean() * 100
    ticket_promedio = (
        df_ventas["ticket_promedio"].mean() if "ticket_promedio" in df_ventas else 0
    )
    cantidad_promedio = (
        df_ventas["cantidad_ventas"].mean() if "cantidad_ventas" in df_ventas else 0
    )

    recomendaciones = _reglas_recomendacion(
        anomalias=resultado["anomalias"],
        tendencia=info_modelo.get("direccion", "estable"),
        tendencia_pct=tendencia_reciente,
        cumplimiento_meta=cumplimiento_meta,
        forecast=resultado["forecast"],
        meta_mensual=meta_mensual,
        ticket_promedio=ticket_promedio,
        cantidad_promedio=cantidad_promedio,
    )
    resultado["recomendaciones"] = recomendaciones

    resultado["contexto_ia"] = _construir_contexto_ia(
        anomalias=resultado["anomalias"],
        forecast=resultado["forecast"],
        recomendaciones=recomendaciones,
        tendencia_pct=tendencia_reciente,
        cumplimiento_meta=cumplimiento_meta,
        meta_mensual=meta_mensual,
    )

    return resultado


def _reglas_recomendacion(
    anomalias: list,
    tendencia: str,
    tendencia_pct: float,
    cumplimiento_meta: float,
    forecast: list,
    meta_mensual: float,
    ticket_promedio: float,
    cantidad_promedio: float,
) -> list[str]:
    """
    Motor de reglas que traduce hallazgos de ML en recomendaciones accionables.
    """
    rec = []

    # — Anomalías —
    if anomalias:
        meses_str = ", ".join(a["mes"] for a in anomalias[:3])
        rec.append(
            f"Se detectaron meses con comportamiento atípico: {meses_str}. "
            "Revisar eventos externos (feriados, competencia, clima) que puedan explicar las desviaciones."
        )
        anomalias_bajas = [a for a in anomalias if "baja" in a.get("razon", "")]
        if anomalias_bajas:
            rec.append(
                "Los meses anómalos con ventas bajas sugieren caídas de demanda puntuales. "
                "Evaluar implementar promociones de rescate en semanas previas a meses históricamente débiles."
            )

    # — Tendencia —
    if tendencia == "decreciente" and tendencia_pct < -5:
        rec.append(
            f"La tendencia reciente es decreciente ({tendencia_pct:.1f}%). "
            "Prioridad alta: revisar mix de productos, precios y fuerza de ventas de forma inmediata."
        )
    elif tendencia == "creciente" and tendencia_pct > 5:
        rec.append(
            f"El negocio muestra crecimiento reciente ({tendencia_pct:.1f}%). "
            "Aprovechar el momentum: incrementar stock, reforzar la experiencia del cliente y medir NPS."
        )
    elif abs(tendencia_pct) <= 5:
        rec.append(
            "Las ventas están estabilizadas. "
            "Para romper la meseta considerar acciones de upselling, nuevos combos o expansión de horario."
        )

    if cumplimiento_meta < 50:
        rec.append(
            f"Solo el {cumplimiento_meta:.0f}% de los meses alcanzó la meta. "
            "Reevaluar si la meta es realista o implementar incentivos por tramo (70%, 85%, 100%)."
        )
    elif cumplimiento_meta < 75:
        rec.append(
            f"Cumplimiento de meta en {cumplimiento_meta:.0f}%. "
            "Foco en los meses de mayor tráfico para asegurar superávit que compense los meses bajos."
        )
    else:
        rec.append(
            f"Buen cumplimiento de meta ({cumplimiento_meta:.0f}%). "
            "Considerar elevar la meta progresivamente para sostener el reto al equipo."
        )

    if forecast:
        prox = forecast[0]
        diferencia = prox["ventas_proyectadas"] - meta_mensual
        if diferencia < 0:
            rec.append(
                f"El modelo proyecta ${prox['ventas_proyectadas']:,.0f} para el próximo mes, "
                f"${abs(diferencia):,.0f} por debajo de la meta. "
                "Activar plan de contingencia: promociones del mes, refuerzo en redes sociales."
            )
        else:
            rec.append(
                f"Proyección del próximo mes: ${prox['ventas_proyectadas']:,.0f} "
                f"(${diferencia:,.0f} sobre la meta). Mantener operaciones y monitorear ticket promedio."
            )

    if ticket_promedio > 0 and ticket_promedio < meta_mensual * 0.005:
        rec.append(
            "Ticket promedio bajo en relación a las metas. "
            "Implementar estrategia de combos familiares, bebidas incluidas o postres de alto margen."
        )

    return rec


def _construir_contexto_ia(
    anomalias: list,
    forecast: list,
    recomendaciones: list,
    tendencia_pct: float,
    cumplimiento_meta: float,
    meta_mensual: float,
) -> str:
    """
    Construye el string de contexto enriquecido con ML
    para pasarlo a generar_respuesta_chat() de ai_service.py.
    """
    lineas = ["=== ANÁLISIS DE MACHINE LEARNING ===\n"]

    if anomalias:
        lineas.append("ANOMALÍAS DETECTADAS:")
        for a in anomalias:
            lineas.append(
                f"  • {a['mes']}: ventas ${a['ventas']:,.0f} | "
                f"score anomalía {a['score']:.2f} | razón: {a['razon']}"
            )
    else:
        lineas.append(
            "ANOMALÍAS: Ningún mes con comportamiento estadísticamente atípico."
        )

    lineas.append("")

    if forecast:
        lineas.append("PROYECCIÓN DE VENTAS:")
        for f in forecast:
            lineas.append(
                f"  • Mes +{f['mes_futuro']}: ${f['ventas_proyectadas']:,.0f} "
                f"(rango: ${f['intervalo_inferior']:,.0f} – ${f['intervalo_superior']:,.0f})"
            )
        lineas.append("")

    lineas.append(
        f"TENDENCIA RECIENTE: {tendencia_pct:+.1f}% (últimos 3 vs anteriores)"
    )
    lineas.append(
        f"CUMPLIMIENTO DE META: {cumplimiento_meta:.0f}% de los meses sobre ${meta_mensual:,.0f}"
    )
    lineas.append("")

    lineas.append("RECOMENDACIONES AUTOMÁTICAS DEL SISTEMA ML:")
    for i, r in enumerate(recomendaciones, 1):
        lineas.append(f"  {i}. {r}")

    return "\n".join(lineas)


def resumen_ml_dashboard(df_ventas: pd.DataFrame, meta_mensual: float) -> dict:
    """
    Versión ligera para mostrar métricas de ML en el dashboard
    sin ejecutar el pipeline completo.

    Retorna
    -------
    dict con:
        - n_anomalias     : int
        - mes_mas_anomalo : str
        - tendencia       : str  ("creciente" | "decreciente" | "estable")
        - proyeccion_mes1 : float
        - alerta          : str  (mensaje de alerta principal)
    """
    resumen = {
        "n_anomalias": 0,
        "mes_mas_anomalo": "—",
        "tendencia": "estable",
        "proyeccion_mes1": None,
        "alerta": "",
    }

    if df_ventas.empty or len(df_ventas) < 4:
        resumen["alerta"] = "Datos insuficientes para análisis ML."
        return resumen

    try:
        df_anom = detectar_anomalias_ventas(df_ventas)
        anom = df_anom[df_anom["es_anomalia"]]
        resumen["n_anomalias"] = len(anom)
        if not anom.empty:
            idx_max = anom["score_anomalia"].idxmax()
            resumen["mes_mas_anomalo"] = anom.loc[idx_max, "nombre_mes"]
    except Exception:
        pass

    try:
        info = entrenar_forecast_ventas(df_ventas)
        resumen["tendencia"] = info.get("direccion", "estable")
        forecast = predecir_ventas_proximos_meses(df_ventas, 1)
        if forecast:
            resumen["proyeccion_mes1"] = forecast[0]["ventas_proyectadas"]
    except Exception:
        pass

    if resumen["n_anomalias"] > 0:
        resumen["alerta"] = (
            f"⚠️ {resumen['n_anomalias']} mes(es) con comportamiento atípico detectado(s). "
            f"El más relevante: {resumen['mes_mas_anomalo']}."
        )
    elif resumen["tendencia"] == "decreciente":
        resumen["alerta"] = (
            "Tendencia decreciente detectada. Revisar estrategia de ventas."
        )
    else:
        resumen["alerta"] = "Sin alertas críticas. Monitoreo normal."

    return resumen
