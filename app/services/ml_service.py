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
    features = [f for f in features if f in df.columns]
    if not features:
        return {"error": "No hay columnas válidas para entrenar."}
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
    df_out = df.copy()
    features = ["ventas_mes", "ticket_promedio", "cantidad_ventas"]
    features = [f for f in features if f in df_out.columns]
    if not features:
        df_out["es_anomalia"] = False
        df_out["score_anomalia"] = 0.0
        df_out["razon_anomalia"] = ""
        return df_out

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
    if not row["es_anomalia"]:
        return ""

    razones = []

    if "ventas_mes" in df.columns:
        media_ventas = df["ventas_mes"].mean()
        std_ventas = df["ventas_mes"].std()
        if std_ventas > 0 and abs(row["ventas_mes"] - media_ventas) > 1.5 * std_ventas:
            direccion = "alta" if row["ventas_mes"] > media_ventas else "baja"
            razones.append(f"venta mensual inusualmente {direccion}")

    if "ticket_promedio" in df.columns:
        media_ticket = df["ticket_promedio"].mean()
        std_ticket = df["ticket_promedio"].std()
        if (
            std_ticket > 0
            and abs(row["ticket_promedio"] - media_ticket) > 1.5 * std_ticket
        ):
            direccion = "alto" if row["ticket_promedio"] > media_ticket else "bajo"
            razones.append(f"ticket promedio inusualmente {direccion}")

    if "cantidad_ventas" in df.columns:
        media_cant = df["cantidad_ventas"].mean()
        std_cant = df["cantidad_ventas"].std()
        if std_cant > 0 and abs(row["cantidad_ventas"] - media_cant) > 1.5 * std_cant:
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
        df_ventas["ticket_promedio"].mean()
        if "ticket_promedio" in df_ventas.columns
        else 0
    )
    cantidad_promedio = (
        df_ventas["cantidad_ventas"].mean()
        if "cantidad_ventas" in df_ventas.columns
        else 0
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


def generar_contexto_ml_ticket(df: pd.DataFrame, meta_ticket: float) -> str:
    """
    Pipeline ML para el chat de Ticket Promedio.
    Columnas esperadas: ticket_promedio, ventas_totales, cantidad_ventas, nombre_mes
    """
    lineas = ["=== ANÁLISIS ML – TICKET PROMEDIO ===\n"]

    cols_anom = [
        c
        for c in ["ticket_promedio", "ventas_totales", "cantidad_ventas"]
        if c in df.columns
    ]
    if cols_anom and len(df) >= 6:
        pipeline = _load_model(MODEL_ANOMALY_TICKET)
        if pipeline is None:
            p = _build_anomaly_pipeline(0.1)
            p.fit(df[cols_anom].fillna(df[cols_anom].median()).values)
            _save_model(p, MODEL_ANOMALY_TICKET)
            pipeline = p
        X = df[cols_anom].fillna(df[cols_anom].median()).values
        flags = _flag_anomalies(pipeline, X)
        scores = _anomaly_score(pipeline, X)
        df_tmp = df.copy()
        df_tmp["_flag"] = flags
        df_tmp["_score"] = scores
        anomalos = df_tmp[df_tmp["_flag"]]
        if not anomalos.empty:
            lineas.append("ANOMALÍAS DETECTADAS (ticket):")
            for _, r in anomalos.iterrows():
                lineas.append(
                    f"  • {r.get('nombre_mes','?')}: ticket ${r.get('ticket_promedio',0):,.2f} | score {r['_score']:.2f}"
                )
        else:
            lineas.append("ANOMALÍAS: Ningún mes con ticket atípico.")
    else:
        lineas.append("ANOMALÍAS: Datos insuficientes para detección.")

    lineas.append("")

    if "ticket_promedio" in df.columns and len(df) >= 4:
        df_c = df[["ticket_promedio"]].dropna().reset_index(drop=True)
        X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        y_h = df_c["ticket_promedio"].values
        from sklearn.linear_model import LinearRegression

        m = LinearRegression().fit(X_h, y_h)
        std_e = np.std(y_h - m.predict(X_h))
        pendiente = m.coef_[0]
        lineas.append(
            f"PROYECCIÓN TICKET ({'creciente' if pendiente > 0 else 'decreciente'}, {pendiente:+.2f}/mes):"
        )
        for i in range(1, 4):
            pred = m.predict([[len(df_c) + i]])[0]
            lineas.append(
                f"  • Mes +{i}: ${pred:,.2f}  (rango: ${pred - 1.28*std_e:,.2f} – ${pred + 1.28*std_e:,.2f})"
            )
        lineas.append("")

    if "ticket_promedio" in df.columns:
        rec3 = df["ticket_promedio"].iloc[-3:].mean()
        ant = df["ticket_promedio"].iloc[:-3].mean() if len(df) > 3 else rec3
        tend_pct = ((rec3 - ant) / ant * 100) if ant != 0 else 0
        cumpl = (df["ticket_promedio"] >= meta_ticket).mean() * 100
        lineas.append(f"TENDENCIA RECIENTE: {tend_pct:+.1f}% (últimos 3 vs anteriores)")
        lineas.append(f"CUMPLIMIENTO META ${meta_ticket}: {cumpl:.0f}% de los meses")

    return "\n".join(lineas)


def generar_contexto_ml_cantidad(df: pd.DataFrame, meta_cantidad: float) -> str:
    """
    Pipeline ML para el chat de Cantidad de Ventas.
    Columnas esperadas: cantidad_ventas, cantidad_acumulada, nombre_mes
    """
    lineas = ["=== ANÁLISIS ML – CANTIDAD DE VENTAS ===\n"]

    # — Anomalías —
    cols_anom = [
        c for c in ["cantidad_ventas", "cantidad_acumulada"] if c in df.columns
    ]
    if cols_anom and len(df) >= 6:
        pipeline = _load_model(MODEL_ANOMALY_CANTIDAD)
        if pipeline is None:
            p = _build_anomaly_pipeline(0.1)
            p.fit(df[cols_anom].fillna(df[cols_anom].median()).values)
            _save_model(p, MODEL_ANOMALY_CANTIDAD)
            pipeline = p
        X = df[cols_anom].fillna(df[cols_anom].median()).values
        flags = _flag_anomalies(pipeline, X)
        scores = _anomaly_score(pipeline, X)
        df_tmp = df.copy()
        df_tmp["_flag"] = flags
        df_tmp["_score"] = scores
        anomalos = df_tmp[df_tmp["_flag"]]
        if not anomalos.empty:
            lineas.append("ANOMALÍAS DETECTADAS (cantidad):")
            for _, r in anomalos.iterrows():
                lineas.append(
                    f"  • {r.get('nombre_mes','?')}: {r.get('cantidad_ventas',0):,.0f} ventas | score {r['_score']:.2f}"
                )
        else:
            lineas.append("ANOMALÍAS: Ningún mes con volumen atípico.")
    else:
        lineas.append("ANOMALÍAS: Datos insuficientes.")

    lineas.append("")

    if "cantidad_ventas" in df.columns and len(df) >= 4:
        df_c = df[["cantidad_ventas"]].dropna().reset_index(drop=True)
        X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        y_h = df_c["cantidad_ventas"].values
        from sklearn.linear_model import LinearRegression

        m = LinearRegression().fit(X_h, y_h)
        std_e = np.std(y_h - m.predict(X_h))
        pendiente = m.coef_[0]
        lineas.append(
            f"PROYECCIÓN CANTIDAD ({'creciente' if pendiente > 0 else 'decreciente'}, {pendiente:+.0f}/mes):"
        )
        for i in range(1, 4):
            pred = m.predict([[len(df_c) + i]])[0]
            lineas.append(
                f"  • Mes +{i}: {pred:,.0f} ventas  (rango: {pred - 1.28*std_e:,.0f} – {pred + 1.28*std_e:,.0f})"
            )
        lineas.append("")

    if "cantidad_ventas" in df.columns:
        rec3 = df["cantidad_ventas"].iloc[-3:].mean()
        ant = df["cantidad_ventas"].iloc[:-3].mean() if len(df) > 3 else rec3
        tend_pct = ((rec3 - ant) / ant * 100) if ant != 0 else 0
        cumpl = (df["cantidad_ventas"] >= meta_cantidad).mean() * 100
        lineas.append(f"TENDENCIA RECIENTE: {tend_pct:+.1f}%")
        lineas.append(
            f"CUMPLIMIENTO META {meta_cantidad:,.0f} ventas: {cumpl:.0f}% de los meses"
        )

    return "\n".join(lineas)


def generar_contexto_ml_crecimiento(df: pd.DataFrame, meta_crecimiento: float) -> str:
    """
    Pipeline ML para el chat de Crecimiento de Ventas.
    Columnas esperadas: crecimiento_ventas, ventas_mes, nombre_mes, cumple_meta
    """
    lineas = ["=== ANÁLISIS ML – CRECIMIENTO DE VENTAS ===\n"]

    if "crecimiento_ventas" in df.columns and len(df) >= 6:
        X = df[["crecimiento_ventas"]].fillna(0).values
        pipeline = (
            _load_model(_MODEL_PATHS.get("crecimiento", ""))
            if hasattr(df, "_dummy")
            else None
        )
        import os

        crecimiento_path = os.path.join(MODELS_DIR, "anomaly_crecimiento.pkl")
        pipeline = _load_model(crecimiento_path)
        if pipeline is None:
            pipeline = _build_anomaly_pipeline(0.1)
            pipeline.fit(X)
            _save_model(pipeline, crecimiento_path)
        flags = _flag_anomalies(pipeline, X)
        scores = _anomaly_score(pipeline, X)
        df_tmp = df.copy()
        df_tmp["_flag"] = flags
        df_tmp["_score"] = scores
        anomalos = df_tmp[df_tmp["_flag"]]
        if not anomalos.empty:
            lineas.append("ANOMALÍAS DETECTADAS (crecimiento):")
            for _, r in anomalos.iterrows():
                crec_pct = r.get("crecimiento_ventas", 0) * 100
                lineas.append(
                    f"  • {r.get('nombre_mes','?')}: {crec_pct:+.1f}% | score {r['_score']:.2f}"
                )
        else:
            lineas.append(
                "ANOMALÍAS: Ningún mes con crecimiento estadísticamente atípico."
            )
    else:
        lineas.append("ANOMALÍAS: Datos insuficientes.")

    lineas.append("")

    if "ventas_mes" in df.columns and len(df) >= 4:
        df_c = df[["ventas_mes"]].dropna().reset_index(drop=True)
        X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        y_h = df_c["ventas_mes"].values
        from sklearn.linear_model import LinearRegression

        m = LinearRegression().fit(X_h, y_h)
        std_e = np.std(y_h - m.predict(X_h))
        pendiente = m.coef_[0]
        lineas.append(
            f"PROYECCIÓN VENTAS ({'creciente' if pendiente > 0 else 'decreciente'}, {pendiente:+.0f}/mes):"
        )
        for i in range(1, 4):
            pred = m.predict([[len(df_c) + i]])[0]
            lineas.append(
                f"  • Mes +{i}: ${pred:,.0f}  (rango: ${pred - 1.28*std_e:,.0f} – ${pred + 1.28*std_e:,.0f})"
            )
        lineas.append("")

    if "crecimiento_ventas" in df.columns:
        prom = df["crecimiento_ventas"].mean() * 100
        cumpl = (df["crecimiento_ventas"] >= meta_crecimiento).mean() * 100
        tend = (
            "mejorando"
            if df["crecimiento_ventas"].iloc[-3:].mean()
            > df["crecimiento_ventas"].iloc[:3].mean()
            else "deteriorándose"
        )
        lineas.append(f"CRECIMIENTO PROMEDIO: {prom:+.2f}%")
        lineas.append(
            f"CUMPLIMIENTO META {meta_crecimiento*100:.0f}%: {cumpl:.0f}% de los meses"
        )
        lineas.append(f"TENDENCIA GENERAL: {tend}")

    return "\n".join(lineas)


def generar_contexto_ml_vendedores(df: pd.DataFrame, meta_vendedor: float) -> str:
    """
    Pipeline ML para el chat de Ventas por Vendedor.
    Columnas esperadas: vendedor, ventas_totales, clientes_atendidos, productos_vendidos
    """
    lineas = ["=== ANÁLISIS ML – VENTAS POR VENDEDOR ===\n"]

    cols_group = [
        c
        for c in ["ventas_totales", "clientes_atendidos", "productos_vendidos"]
        if c in df.columns
    ]
    if "vendedor" in df.columns and cols_group and len(df["vendedor"].unique()) >= 4:
        df_vend = df.groupby("vendedor", as_index=False)[cols_group].sum()
        import os

        vendedores_path = os.path.join(MODELS_DIR, "anomaly_vendedores.pkl")
        X = df_vend[cols_group].fillna(0).values
        pipeline = _load_model(vendedores_path)
        if pipeline is None:
            pipeline = _build_anomaly_pipeline(0.15)
            pipeline.fit(X)
            _save_model(pipeline, vendedores_path)
        flags = _flag_anomalies(pipeline, X)
        scores = _anomaly_score(pipeline, X)
        df_vend["_flag"] = flags
        df_vend["_score"] = scores
        anomalos = df_vend[df_vend["_flag"]]
        if not anomalos.empty:
            lineas.append("VENDEDORES CON DESEMPEÑO ATÍPICO (IsolationForest):")
            media_v = df_vend["ventas_totales"].mean()
            for _, r in anomalos.iterrows():
                dir_str = (
                    "por encima" if r["ventas_totales"] > media_v else "por debajo"
                )
                lineas.append(
                    f"  • {r['vendedor']}: ${r['ventas_totales']:,.0f} — {dir_str} del promedio | score {r['_score']:.2f}"
                )
        else:
            lineas.append(
                "ANOMALÍAS: Ningún vendedor con desempeño estadísticamente atípico."
            )
    else:
        lineas.append(
            "ANOMALÍAS: Se necesitan al menos 4 vendedores para detección ML."
        )

    lineas.append("")

    if "ventas_totales" in df.columns and "vendedor" in df.columns:
        n_vend = df["vendedor"].nunique()
        total = df["ventas_totales"].sum()
        prom_vend = total / n_vend if n_vend > 0 else 0
        cumpl = (
            df.groupby("vendedor")["ventas_totales"].sum() >= meta_vendedor
        ).mean() * 100
        mejor = df.groupby("vendedor")["ventas_totales"].sum().idxmax()
        lineas.append(f"PROMEDIO POR VENDEDOR: ${prom_vend:,.0f}")
        lineas.append(
            f"CUMPLIMIENTO META ${meta_vendedor:,.0f}: {cumpl:.0f}% de vendedores"
        )
        lineas.append(f"VENDEDOR LÍDER: {mejor}")
        if "ventas_promedio_por_vendedor" in df.columns:
            df_m = df.groupby("nombre_mes", as_index=False)[
                "ventas_promedio_por_vendedor"
            ].first()
            if len(df_m) >= 4:
                df_c = (
                    df_m[["ventas_promedio_por_vendedor"]]
                    .dropna()
                    .reset_index(drop=True)
                )
                X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
                y_h = df_c["ventas_promedio_por_vendedor"].values
                from sklearn.linear_model import LinearRegression

                m = LinearRegression().fit(X_h, y_h)
                pendiente = m.coef_[0]
                std_e = np.std(y_h - m.predict(X_h))
                pred1 = m.predict([[len(df_c) + 1]])[0]
                lineas.append("")
                lineas.append(f"PROYECCIÓN PROMEDIO POR VENDEDOR:")
                lineas.append(
                    f"  • Mes +1: ${pred1:,.0f}  (rango: ${pred1 - 1.28*std_e:,.0f} – ${pred1 + 1.28*std_e:,.0f})"
                )
                lineas.append(
                    f"  Tendencia: {'creciente' if pendiente > 0 else 'decreciente'} ({pendiente:+.0f}/mes)"
                )

    return "\n".join(lineas)


def generar_contexto_ml_productos(df: pd.DataFrame, meta_productos: float) -> str:
    """
    Pipeline ML para el chat de Productos Vendidos por Tipo.
    Columnas esperadas: tipo_plato, cantidad_vendida, ventas_totales, cantidad_ventas
    """
    lineas = ["=== ANÁLISIS ML – PRODUCTOS VENDIDOS ===\n"]

    cols_group = [c for c in ["cantidad_vendida", "ventas_totales"] if c in df.columns]
    if "tipo_plato" in df.columns and cols_group:
        df_tipo = df.groupby("tipo_plato", as_index=False)[cols_group].sum()
        if len(df_tipo) >= 4:
            import os

            productos_path = os.path.join(MODELS_DIR, "anomaly_productos.pkl")
            X = df_tipo[cols_group].fillna(0).values
            pipeline = _load_model(productos_path)
            if pipeline is None:
                pipeline = _build_anomaly_pipeline(0.15)
                pipeline.fit(X)
                _save_model(pipeline, productos_path)
            flags = _flag_anomalies(pipeline, X)
            scores = _anomaly_score(pipeline, X)
            df_tipo["_flag"] = flags
            df_tipo["_score"] = scores
            anomalos = df_tipo[df_tipo["_flag"]]
            if not anomalos.empty:
                lineas.append("TIPOS DE PLATO CON PARTICIPACIÓN ATÍPICA:")
                media_t = df_tipo["cantidad_vendida"].mean()
                for _, r in anomalos.iterrows():
                    dir_str = (
                        "excepcionalmente alto"
                        if r["cantidad_vendida"] > media_t
                        else "excepcionalmente bajo"
                    )
                    lineas.append(
                        f"  • {r['tipo_plato']}: {r['cantidad_vendida']:,.0f} uds — {dir_str} | score {r['_score']:.2f}"
                    )
            else:
                lineas.append(
                    "ANOMALÍAS: Todos los tipos tienen participación estadísticamente normal."
                )
        else:
            lineas.append("ANOMALÍAS: Se necesitan al menos 4 tipos de plato.")
    else:
        lineas.append("ANOMALÍAS: Datos insuficientes.")

    lineas.append("")

    if (
        "cantidad_vendida" in df.columns
        and "mes" in df.columns
        and len(df["mes"].unique()) >= 4
    ):
        df_m = (
            df.groupby("mes", as_index=False)["cantidad_vendida"]
            .sum()
            .sort_values("mes")
        )
        df_c = df_m[["cantidad_vendida"]].dropna().reset_index(drop=True)
        X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        y_h = df_c["cantidad_vendida"].values
        from sklearn.linear_model import LinearRegression

        m = LinearRegression().fit(X_h, y_h)
        std_e = np.std(y_h - m.predict(X_h))
        pendiente = m.coef_[0]
        lineas.append(
            f"PROYECCIÓN PRODUCTOS TOTALES ({'creciente' if pendiente > 0 else 'decreciente'}):"
        )
        for i in range(1, 4):
            pred = m.predict([[len(df_c) + i]])[0]
            lineas.append(
                f"  • Mes +{i}: {pred:,.0f} uds  (rango: {pred - 1.28*std_e:,.0f} – {pred + 1.28*std_e:,.0f})"
            )
        lineas.append("")

    if "tipo_plato" in df.columns and "cantidad_vendida" in df.columns:
        df_dist = df.groupby("tipo_plato")["cantidad_vendida"].sum()
        total = df_dist.sum()
        meses = df["mes"].nunique() if "mes" in df.columns else 1
        cumpl = (
            (df.groupby("mes")["cantidad_vendida"].sum() >= meta_productos).mean() * 100
            if "mes" in df.columns
            else 0
        )
        lineas.append(f"TOTAL PRODUCTOS: {total:,.0f}")
        lineas.append(
            f"CUMPLIMIENTO META {meta_productos:,.0f}: {cumpl:.0f}% de los meses"
        )
        lineas.append("DISTRIBUCIÓN POR TIPO:")
        for tipo, cant in df_dist.sort_values(ascending=False).items():
            lineas.append(f"  • {tipo}: {cant:,.0f} ({cant/total*100:.1f}%)")

    return "\n".join(lineas)


def generar_contexto_ml_participacion(
    df: pd.DataFrame, meta_participacion: float
) -> str:
    """
    Pipeline ML para el chat de Participación de Productos.
    Columnas esperadas: producto, participacion_porcentual, unidades_vendidas,
                        ventas_totales, categoria, cumple_meta
    """
    lineas = ["=== ANÁLISIS ML – PARTICIPACIÓN DE PRODUCTOS ===\n"]

    cols_anom = [
        c for c in ["participacion_porcentual", "unidades_vendidas"] if c in df.columns
    ]
    if cols_anom and len(df) >= 6:
        import os

        productos_path = os.path.join(MODELS_DIR, "anomaly_productos.pkl")
        X = df[cols_anom].fillna(0).values
        pipeline = _load_model(productos_path)
        if pipeline is None:
            pipeline = _build_anomaly_pipeline(0.1)
            pipeline.fit(X)
            _save_model(pipeline, productos_path)
        flags = _flag_anomalies(pipeline, X)
        scores = _anomaly_score(pipeline, X)
        df_tmp = df.copy()
        df_tmp["_flag"] = flags
        df_tmp["_score"] = scores
        anomalos = df_tmp[df_tmp["_flag"]]
        if not anomalos.empty:
            lineas.append("PRODUCTOS CON PARTICIPACIÓN ATÍPICA (IsolationForest):")
            media_p = df["participacion_porcentual"].mean()
            for _, r in anomalos.iterrows():
                dir_str = "alta" if r["participacion_porcentual"] > media_p else "baja"
                lineas.append(
                    f"  • {r.get('producto','?')}: {r['participacion_porcentual']:.2f}% — participación {dir_str} | score {r['_score']:.2f}"
                )
        else:
            lineas.append(
                "ANOMALÍAS: Ningún producto con participación estadísticamente atípica."
            )
    else:
        lineas.append("ANOMALÍAS: Datos insuficientes para detección.")

    lineas.append("")

    if "participacion_porcentual" in df.columns:
        total_prods = len(df)
        sobre_meta = (df["participacion_porcentual"] >= meta_participacion * 100).sum()
        top3 = df.nlargest(3, "participacion_porcentual")[
            "participacion_porcentual"
        ].sum()
        top5 = df.nlargest(5, "participacion_porcentual")[
            "participacion_porcentual"
        ].sum()
        diversif = "Alta" if top3 < 50 else "Media" if top3 < 70 else "Baja"
        lider = (
            df.loc[df["participacion_porcentual"].idxmax(), "producto"]
            if "producto" in df.columns
            else "N/A"
        )
        debil = (
            df.loc[df["participacion_porcentual"].idxmin(), "producto"]
            if "producto" in df.columns
            else "N/A"
        )

        lineas.append(
            f"PRODUCTOS SOBRE META ({meta_participacion*100:.0f}%): {sobre_meta} de {total_prods}"
        )
        lineas.append(f"CONCENTRACIÓN TOP 3: {top3:.1f}%  |  TOP 5: {top5:.1f}%")
        lineas.append(f"NIVEL DE DIVERSIFICACIÓN: {diversif}")
        lineas.append(
            f"PRODUCTO LÍDER: {lider} ({df['participacion_porcentual'].max():.2f}%)"
        )
        lineas.append(
            f"PRODUCTO MÁS DÉBIL: {debil} ({df['participacion_porcentual'].min():.2f}%)"
        )

    if "categoria" in df.columns and "participacion_porcentual" in df.columns:
        lineas.append("")
        lineas.append("PARTICIPACIÓN POR CATEGORÍA:")
        df_cat = (
            df.groupby("categoria")["participacion_porcentual"]
            .sum()
            .sort_values(ascending=False)
        )
        for cat, part in df_cat.items():
            lineas.append(f"  • {cat}: {part:.1f}%")

    return "\n".join(lineas)
