from __future__ import annotations

import os
import logging
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not _GROQ_API_KEY:
    logging.warning(
        "⚠️  GROQ_API_KEY no está configurada. "
        "Las funciones de IA devolverán un mensaje de error hasta que se configure."
    )

logger = logging.getLogger(__name__)

MODELS_DIR = "app/models"
os.makedirs(MODELS_DIR, exist_ok=True)

_MODEL_PATHS = {
    "ventas": os.path.join(MODELS_DIR, "anomaly_ventas.pkl"),
    "ticket": os.path.join(MODELS_DIR, "anomaly_ticket.pkl"),
    "cantidad": os.path.join(MODELS_DIR, "anomaly_cantidad.pkl"),
    "crecimiento": os.path.join(MODELS_DIR, "anomaly_crecimiento.pkl"),
    "productos": os.path.join(MODELS_DIR, "anomaly_productos.pkl"),
    "vendedores": os.path.join(MODELS_DIR, "anomaly_vendedores.pkl"),
    "forecast": os.path.join(MODELS_DIR, "forecast_ventas.pkl"),
}


def _pipeline_anomalia(contamination: float = 0.1) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "iso",
                IsolationForest(
                    n_estimators=200,
                    contamination=contamination,
                    random_state=42,
                ),
            ),
        ]
    )


def _cargar_o_entrenar(key: str, X: np.ndarray, contamination: float = 0.1) -> Pipeline:
    path = _MODEL_PATHS[key]
    if os.path.exists(path):
        return joblib.load(path)
    pipeline = _pipeline_anomalia(contamination)
    pipeline.fit(X)
    joblib.dump(pipeline, path)
    return pipeline


def _detectar(pipeline: Pipeline, X: np.ndarray):
    flags = pipeline.predict(X) == -1
    raw = -pipeline.named_steps["iso"].score_samples(
        pipeline.named_steps["scaler"].transform(X)
    )
    mn, mx = raw.min(), raw.max()
    scores = (raw - mn) / (mx - mn) if mx != mn else np.zeros(len(raw))
    return flags, scores


def _zscore_razon(valor: float, media: float, std: float, label: str) -> str | None:
    if std == 0:
        return None
    z = (valor - media) / std
    if z > 1.5:
        return f"{label} excepcionalmente alta (+{z:.1f}σ)"
    if z < -1.5:
        return f"{label} excepcionalmente baja ({z:.1f}σ)"
    return None


def _bloque_anomalias_texto(df_anom: pd.DataFrame, col_mes: str = "nombre_mes") -> str:
    anomalos = df_anom[df_anom["_es_anomalia"]]
    if anomalos.empty:
        return ""
    lineas = ["\n⚠️ ANOMALÍAS DETECTADAS POR ML (IsolationForest):"]
    for _, r in anomalos.iterrows():
        lineas.append(
            f"  • {r.get(col_mes, 'Desconocido')} — score {r['_score_anomalia']:.2f} "
            f"| {r['_razon_anomalia']}"
        )
    return "\n".join(lineas)


def _bloque_forecast_texto(df: pd.DataFrame, col_ventas: str, n: int = 3) -> str:
    df_c = df[[col_ventas]].dropna().reset_index(drop=True)
    if len(df_c) < 4:
        return ""
    X_hist = np.arange(1, len(df_c) + 1).reshape(-1, 1)
    y_hist = df_c[col_ventas].values
    model = LinearRegression().fit(X_hist, y_hist)
    std_e = np.std(y_hist - model.predict(X_hist))
    path = _MODEL_PATHS.get("forecast", "")
    if path:
        joblib.dump(model, path)
    n_base = len(df_c)
    pendiente = model.coef_[0]
    direccion = "creciente" if pendiente > 0 else "decreciente"
    lineas = [f"\nPROYECCIÓN ML ({direccion}, pendiente {pendiente:+.0f}/mes):"]
    for i in range(1, n + 1):
        pred = model.predict([[n_base + i]])[0]
        lineas.append(
            f"  • Mes +{i}: ${pred:,.0f}  "
            f"(rango: ${pred - 1.28*std_e:,.0f} – ${pred + 1.28*std_e:,.0f})"
        )
    return "\n".join(lineas)


def _recomendaciones_ml(
    n_anomalias: int,
    tendencia_pct: float,
    cumplimiento_meta_pct: float,
    forecast_mes1: float | None,
    meta: float,
    extra: str = "",
) -> str:
    rec = []
    if n_anomalias > 0:
        rec.append(
            f"{n_anomalias} mes(es) con comportamiento atípico. "
            "Investigar causas externas y activar plan de contingencia en meses históricamente débiles."
        )
    if tendencia_pct < -5:
        rec.append(
            f"Tendencia reciente negativa ({tendencia_pct:.1f}%). "
            "Revisar pricing, mix de productos y fuerza de ventas de forma inmediata."
        )
    elif tendencia_pct > 5:
        rec.append(
            f"Momentum positivo ({tendencia_pct:+.1f}%). "
            "Ampliar capacidad operativa, reforzar stock y medir NPS para sostener el crecimiento."
        )
    else:
        rec.append(
            "Ventas estabilizadas. Implementar upselling, nuevos combos o ampliar horario para romper la meseta."
        )
    if cumplimiento_meta_pct < 50:
        rec.append(
            f"Solo {cumplimiento_meta_pct:.0f}% de meses sobre meta. "
            "Reevaluar si la meta es alcanzable o introducir incentivos escalonados (70/85/100%)."
        )
    elif cumplimiento_meta_pct < 75:
        rec.append(
            f"Cumplimiento del {cumplimiento_meta_pct:.0f}%. "
            "Concentrar esfuerzos en meses de alto tráfico para generar superávit que compense los bajos."
        )
    else:
        rec.append(
            f"Buen cumplimiento ({cumplimiento_meta_pct:.0f}%). "
            "Elevar progresivamente la meta para mantener el reto al equipo."
        )
    if forecast_mes1 is not None:
        diff = forecast_mes1 - meta
        if diff < 0:
            rec.append(
                f"Próximo mes proyectado: ${forecast_mes1:,.0f} "
                f"(${abs(diff):,.0f} bajo la meta). Activar promociones con anticipación."
            )
        else:
            rec.append(
                f"Próximo mes proyectado: ${forecast_mes1:,.0f} "
                f"(${diff:,.0f} sobre la meta). Mantener operaciones y vigilar ticket promedio."
            )
    if extra:
        rec.append(extra)
    lineas = ["\nRECOMENDACIONES AUTOMÁTICAS DEL SISTEMA ML:"]
    for i, r in enumerate(rec, 1):
        lineas.append(f"  {i}. {r}")
    return "\n".join(lineas)


def _llamar_ollama(
    prompt: str, temperature: float = 0.15, num_predict: int = 900
) -> str:
    if not _GROQ_API_KEY:
        return "⚠️ La IA no está configurada. Contacta al administrador (falta GROQ_API_KEY)."

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {_GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": num_predict,
            },
            timeout=60,
        )

        if response.status_code == 401:
            logger.error("Groq: API key inválida o expirada")
            return "Error de autenticación con la IA. Verifica la GROQ_API_KEY."

        if response.status_code == 429:
            logger.warning("Groq: rate limit alcanzado")
            return "La IA está recibiendo demasiadas solicitudes. Intenta en unos segundos."

        if response.status_code == 503:
            logger.warning("Groq: servicio no disponible")
            return "El servicio de IA no está disponible en este momento. Intenta más tarde."

        if response.status_code != 200:
            logger.error(
                f"Groq: error inesperado {response.status_code} — {response.text[:200]}"
            )
            return f"Error al conectar con la IA (código: {response.status_code})."

        data = response.json()

        if "choices" not in data or not data["choices"]:
            logger.error(f"Groq: respuesta sin choices — {data}")
            return "La IA devolvió una respuesta inesperada. Intenta de nuevo."

        return data["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout:
        logger.error("Groq: timeout después de 60s")
        return "La IA tardó demasiado en responder. Intenta de nuevo."

    except requests.exceptions.ConnectionError:
        logger.error("Groq: error de conexión")
        return "No se pudo conectar con el servicio de IA. Verifica tu conexión a internet."

    except Exception as e:
        logger.error(f"Groq: error inesperado — {e}")
        return f"Error inesperado al procesar la respuesta de la IA."


def generar_respuesta_chat(contexto: str, pregunta: str) -> str:
    prompt = f"""
Eres un consultor experto en pollerías y restaurantes.

CONTEXTO DE VENTAS:
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}

Responde de forma clara, concisa y orientada a acciones. Si el usuario pide recomendaciones, sé específico.
Usa formato markdown básico si es útil para la legibilidad.
"""
    return _llamar_ollama(prompt, temperature=0.2, num_predict=800)


def generar_analisis_ia(df: pd.DataFrame, meta_mensual: float) -> str:
    if df.empty or len(df) < 2:
        return "No hay datos suficientes para generar un análisis consistente."

    ventas_totales = df["ventas_mes"].sum()
    promedio = df["ventas_mes"].mean()
    desviacion = df["ventas_mes"].std()
    mejor_mes = df.loc[df["ventas_mes"].idxmax()]
    peor_mes = df.loc[df["ventas_mes"].idxmin()]
    crecimiento = (
        (df["ventas_mes"].iloc[-1] - df["ventas_mes"].iloc[0])
        / df["ventas_mes"].iloc[0]
    ) * 100
    df["cumple_meta"] = df["ventas_mes"] >= meta_mensual
    meses_bajo_meta = (~df["cumple_meta"]).sum()
    cumplimiento_pct = (df["ventas_mes"] / meta_mensual).mean() * 100
    variabilidad = "Alta" if desviacion > promedio * 0.25 else "Estable"

    resumen = f"""
    - Ventas totales del periodo: ${ventas_totales:,.0f}
    - Venta promedio mensual: ${promedio:,.0f}
    - Variabilidad de ventas: {variabilidad}
    - Mes con mayor venta: {mejor_mes['nombre_mes']}
    - Mes con menor venta: {peor_mes['nombre_mes']}
    - Crecimiento entre primer y último mes: {crecimiento:.2f}%
    - Meses que no alcanzaron la meta mensual: {meses_bajo_meta}
    - Cumplimiento promedio de meta mensual: {cumplimiento_pct:.2f}%
    """

    features = ["ventas_mes", "ticket_promedio", "cantidad_ventas"]
    cols_ok = [c for c in features if c in df.columns]
    df_ml = df.copy()
    bloque_anom = ""
    n_anom = 0

    if len(cols_ok) >= 1 and len(df) >= 6:
        X = df[cols_ok].fillna(df[cols_ok].median()).values
        pipeline = _cargar_o_entrenar("ventas", X)
        flags, scores = _detectar(pipeline, X)
        df_ml["_es_anomalia"] = flags
        df_ml["_score_anomalia"] = scores
        media_v = df["ventas_mes"].mean()
        std_v = df["ventas_mes"].std()

        def _razon_v(row):
            if not row["_es_anomalia"]:
                return ""
            partes = []
            r = _zscore_razon(row["ventas_mes"], media_v, std_v, "venta mensual")
            if r:
                partes.append(r)
            if "ticket_promedio" in df.columns:
                r2 = _zscore_razon(
                    row["ticket_promedio"],
                    df["ticket_promedio"].mean(),
                    df["ticket_promedio"].std(),
                    "ticket promedio",
                )
                if r2:
                    partes.append(r2)
            return "; ".join(partes) if partes else "combinación inusual de indicadores"

        df_ml["_razon_anomalia"] = df_ml.apply(_razon_v, axis=1)
        bloque_anom = _bloque_anomalias_texto(df_ml)
        n_anom = int(flags.sum())

    bloque_forecast = _bloque_forecast_texto(df, "ventas_mes", n=3)
    forecast_mes1 = None
    if len(df) >= 4:
        df_c = df[["ventas_mes"]].dropna().reset_index(drop=True)
        X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        m_lr = LinearRegression().fit(X_h, df_c["ventas_mes"].values)
        forecast_mes1 = m_lr.predict([[len(df_c) + 1]])[0]

    tend_pct = 0.0
    if len(df) >= 4:
        rec_3 = df["ventas_mes"].iloc[-3:].mean()
        ant = df["ventas_mes"].iloc[:-3].mean()
        tend_pct = ((rec_3 - ant) / ant * 100) if ant != 0 else 0.0

    bloque_rec = _recomendaciones_ml(
        n_anomalias=n_anom,
        tendencia_pct=tend_pct,
        cumplimiento_meta_pct=cumplimiento_pct,
        forecast_mes1=forecast_mes1,
        meta=meta_mensual,
    )

    prompt = f"""
Eres un consultor senior en inteligencia de negocio especializado en restaurantes.

Analiza este resumen de desempeño mensual de una pollería.

RESUMEN ESTADÍSTICO:
{resumen}
{bloque_anom}
{bloque_forecast}
{bloque_rec}

INSTRUCCIONES IMPORTANTES:
- NO copies el resumen ni enumeres todos los valores otra vez
- SÍ puedes usar cifras cuando sean relevantes para justificar decisiones
- Interpreta el comportamiento del negocio integrando los hallazgos de ML
- Si hay anomalías o alertas ML, mencionarlas en el análisis
- Habla como asesor estratégico, no como reporte técnico

FORMATO DE RESPUESTA (OBLIGATORIO):

Tendencias del negocio
(Explica patrones, estabilidad o variaciones incluyendo lo que el ML detectó.)

Riesgos o alertas detectadas
(Señala problemas reales, incluidas las anomalías ML.)

Oportunidades de mejora
(Áreas donde el negocio puede crecer o estabilizarse.)

Recomendaciones estratégicas accionables
(Acciones concretas basadas en estadísticas + ML.)
"""
    return _llamar_ollama(prompt)


def generar_analisis_ticket_promedio(df: pd.DataFrame, meta_ticket: float) -> str:
    if df.empty or len(df) < 2:
        return "No hay datos suficientes para generar un análisis consistente."

    ticket_promedio = df["ticket_promedio"].mean()
    ticket_max = df["ticket_promedio"].max()
    ticket_min = df["ticket_promedio"].min()
    desviacion = df["ticket_promedio"].std()
    mejor_mes = df.loc[df["ticket_promedio"].idxmax()]
    peor_mes = df.loc[df["ticket_promedio"].idxmin()]
    crecimiento = (
        (df["ticket_promedio"].iloc[-1] - df["ticket_promedio"].iloc[0])
        / df["ticket_promedio"].iloc[0]
    ) * 100
    df["cumple_meta"] = df["ticket_promedio"] >= meta_ticket
    meses_bajo_meta = (~df["cumple_meta"]).sum()
    cumplimiento_pct = (df["ticket_promedio"] / meta_ticket).mean() * 100
    variabilidad = "Alta" if desviacion > ticket_promedio * 0.15 else "Estable"

    resumen = f"""
    - Ticket promedio del periodo: ${ticket_promedio:,.2f}
    - Meta establecida: ${meta_ticket:,.2f}
    - Cumplimiento promedio de meta: {cumplimiento_pct:.2f}%
    - Variabilidad del ticket: {variabilidad}
    - Mejor mes: {mejor_mes['nombre_mes']} (${ticket_max:,.2f})
    - Peor mes: {peor_mes['nombre_mes']} (${ticket_min:,.2f})
    - Crecimiento entre primer y último mes: {crecimiento:.2f}%
    - Meses por debajo de la meta: {meses_bajo_meta}
    - Ventas totales: ${df['ventas_totales'].sum():,.0f}
    - Cantidad total de ventas: {df['cantidad_ventas'].sum():,.0f}
    """

    features = ["ticket_promedio", "ventas_totales", "cantidad_ventas"]
    cols_ok = [c for c in features if c in df.columns]
    df_ml = df.copy()
    bloque_anom = ""
    n_anom = 0

    if len(cols_ok) >= 1 and len(df) >= 6:
        X = df[cols_ok].fillna(df[cols_ok].median()).values
        pipeline = _cargar_o_entrenar("ticket", X)
        flags, scores = _detectar(pipeline, X)
        df_ml["_es_anomalia"] = flags
        df_ml["_score_anomalia"] = scores
        media_t = df["ticket_promedio"].mean()
        std_t = df["ticket_promedio"].std()
        df_ml["_razon_anomalia"] = df_ml.apply(
            lambda r: _zscore_razon(
                r["ticket_promedio"], media_t, std_t, "ticket promedio"
            )
            or ("combinación inusual" if r["_es_anomalia"] else ""),
            axis=1,
        )
        bloque_anom = _bloque_anomalias_texto(df_ml)
        n_anom = int(flags.sum())

    bloque_forecast = _bloque_forecast_texto(df, "ticket_promedio", n=3)
    forecast_mes1 = None
    if len(df) >= 4:
        df_c = df[["ticket_promedio"]].dropna().reset_index(drop=True)
        X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        m_lr = LinearRegression().fit(X_h, df_c["ticket_promedio"].values)
        forecast_mes1 = m_lr.predict([[len(df_c) + 1]])[0]

    tend_pct = 0.0
    if len(df) >= 4:
        rec_3 = df["ticket_promedio"].iloc[-3:].mean()
        ant = df["ticket_promedio"].iloc[:-3].mean()
        tend_pct = ((rec_3 - ant) / ant * 100) if ant != 0 else 0.0

    bloque_rec = _recomendaciones_ml(
        n_anomalias=n_anom,
        tendencia_pct=tend_pct,
        cumplimiento_meta_pct=cumplimiento_pct,
        forecast_mes1=forecast_mes1,
        meta=meta_ticket,
        extra="Estrategias de upselling: combos con bebidas, postres de alto margen o promociones de tamaño familiar.",
    )

    prompt = f"""
Eres un consultor senior en inteligencia de negocio especializado en restaurantes, con foco en ticket promedio.

RESUMEN ESTADÍSTICO:
{resumen}
{bloque_anom}
{bloque_forecast}
{bloque_rec}

INSTRUCCIONES IMPORTANTES:
- NO copies el resumen ni enumeres todos los valores otra vez
- Interpreta el comportamiento desde la perspectiva del valor por transacción
- Integra los hallazgos de ML en tu análisis
- Habla como asesor estratégico

FORMATO DE RESPUESTA (OBLIGATORIO):

Patrones de Comportamiento del Cliente
(¿Qué dicen estos datos sobre los hábitos de compra, considerando las anomalías detectadas?)

Riesgos en la Rentabilidad por Transacción
(¿Qué meses o patrones representan riesgo, incluyendo alertas ML?)

Oportunidades para Incrementar el Valor
(¿Cómo aumentar el ticket en meses bajos?)

Recomendaciones Estratégicas Accionables
(Acciones específicas: combos, upselling, etc., apoyadas en el forecast ML.)
"""
    return _llamar_ollama(prompt)


def generar_analisis_cantidad_ventas(df: pd.DataFrame, meta_mensual: float) -> str:
    if df.empty or len(df) < 2:
        return "No hay datos suficientes para generar un análisis consistente."

    ventas_totales = df["cantidad_ventas"].sum()
    promedio = df["cantidad_ventas"].mean()
    desviacion = df["cantidad_ventas"].std()
    mejor_mes = df.loc[df["cantidad_ventas"].idxmax()]
    peor_mes = df.loc[df["cantidad_ventas"].idxmin()]
    crecimiento = (
        (df["cantidad_ventas"].iloc[-1] - df["cantidad_ventas"].iloc[0])
        / df["cantidad_ventas"].iloc[0]
    ) * 100
    df["cumple_meta"] = df["cantidad_ventas"] >= meta_mensual
    meses_bajo_meta = (~df["cumple_meta"]).sum()
    cumplimiento_pct = (df["cantidad_ventas"] / meta_mensual).mean() * 100
    variabilidad = "Alta" if desviacion > promedio * 0.25 else "Estable"

    resumen = f"""
    - Ventas totales del periodo: {ventas_totales:,.0f} transacciones
    - Promedio mensual: {promedio:,.0f} ventas
    - Meta mensual establecida: {meta_mensual:,.0f} ventas
    - Cumplimiento promedio de meta: {cumplimiento_pct:.2f}%
    - Variabilidad de ventas: {variabilidad}
    - Mejor mes: {mejor_mes['nombre_mes']} ({mejor_mes['cantidad_ventas']:,.0f} ventas)
    - Peor mes: {peor_mes['nombre_mes']} ({peor_mes['cantidad_ventas']:,.0f} ventas)
    - Crecimiento entre primer y último mes: {crecimiento:.2f}%
    - Meses que no alcanzaron la meta: {meses_bajo_meta}
    - Ventas acumuladas totales: {df['cantidad_acumulada'].iloc[-1]:,.0f}
    - Meta acumulada total: {df['meta_acumulada'].iloc[-1]:,.0f}
    """

    cols_base = ["cantidad_ventas"]
    if "cantidad_acumulada" in df.columns:
        cols_base.append("cantidad_acumulada")

    df_ml = df.copy()
    bloque_anom = ""
    n_anom = 0

    if len(df) >= 6:
        X = df[cols_base].fillna(df[cols_base].median()).values
        pipeline = _cargar_o_entrenar("cantidad", X)
        flags, scores = _detectar(pipeline, X)
        df_ml["_es_anomalia"] = flags
        df_ml["_score_anomalia"] = scores
        media_c = df["cantidad_ventas"].mean()
        std_c = df["cantidad_ventas"].std()
        df_ml["_razon_anomalia"] = df_ml.apply(
            lambda r: _zscore_razon(
                r["cantidad_ventas"], media_c, std_c, "cantidad de ventas"
            )
            or ("combinación inusual" if r["_es_anomalia"] else ""),
            axis=1,
        )
        bloque_anom = _bloque_anomalias_texto(df_ml)
        n_anom = int(flags.sum())

    bloque_forecast = _bloque_forecast_texto(df, "cantidad_ventas", n=3)
    forecast_mes1 = None
    if len(df) >= 4:
        df_c = df[["cantidad_ventas"]].dropna().reset_index(drop=True)
        X_h = np.arange(1, len(df_c) + 1).reshape(-1, 1)
        m_lr = LinearRegression().fit(X_h, df_c["cantidad_ventas"].values)
        forecast_mes1 = m_lr.predict([[len(df_c) + 1]])[0]

    tend_pct = 0.0
    if len(df) >= 4:
        rec_3 = df["cantidad_ventas"].iloc[-3:].mean()
        ant = df["cantidad_ventas"].iloc[:-3].mean()
        tend_pct = ((rec_3 - ant) / ant * 100) if ant != 0 else 0.0

    bloque_rec = _recomendaciones_ml(
        n_anomalias=n_anom,
        tendencia_pct=tend_pct,
        cumplimiento_meta_pct=cumplimiento_pct,
        forecast_mes1=forecast_mes1,
        meta=meta_mensual,
        extra="Para aumentar el tráfico: programas de fidelidad, delivery en horas valle y eventos temáticos.",
    )

    prompt = f"""
Eres un consultor senior en inteligencia de negocio especializado en restaurantes, con foco en volumen de transacciones.

RESUMEN ESTADÍSTICO:
{resumen}
{bloque_anom}
{bloque_forecast}
{bloque_rec}

INSTRUCCIONES IMPORTANTES:
- NO copies el resumen
- Interpreta el comportamiento desde la perspectiva del volumen de transacciones
- Integra hallazgos ML en tu análisis
- Enfócate en tráfico de clientes, no en valor monetario

FORMATO DE RESPUESTA (OBLIGATORIO):

Patrones de Tráfico de Clientes
(¿Qué dicen los datos sobre el flujo, considerando anomalías ML?)

Riesgos en el Volumen de Transacciones
(¿Qué meses o patrones representan riesgo, incluidas alertas ML?)

Oportunidades para Incrementar el Tráfico
(¿Cómo atraer más clientes en meses bajos?)

Recomendaciones Estratégicas Accionables
(Acciones específicas basadas en el volumen + forecast ML.)
"""
    return _llamar_ollama(prompt)


def generar_analisis_crecimiento_ventas(
    df: pd.DataFrame, meta_crecimiento: float
) -> str:
    if df.empty or len(df) < 2:
        return "No hay datos suficientes para generar un análisis consistente del crecimiento."

    crecimiento_promedio = df["crecimiento_ventas"].mean() * 100
    crecimiento_max = df["crecimiento_ventas"].max() * 100
    crecimiento_min = df["crecimiento_ventas"].min() * 100
    desviacion = df["crecimiento_ventas"].std() * 100
    mejor_mes = df.loc[df["crecimiento_ventas"].idxmax()]
    peor_mes = df.loc[df["crecimiento_ventas"].idxmin()]
    meses_sobre_meta = df["cumple_meta"].sum()
    total_meses = len(df)
    porcentaje_sobre_meta = (meses_sobre_meta / total_meses) * 100
    estabilidad = (
        "Alta" if desviacion < 5 else "Moderada" if desviacion < 10 else "Baja"
    )

    tendencia = "estable"
    if len(df) >= 3:
        rec = df["crecimiento_ventas"].iloc[-3:].mean() * 100
        ini = df["crecimiento_ventas"].iloc[:3].mean() * 100
        tendencia = (
            "mejorando" if rec > ini else ("deteriorándose" if rec < ini else "estable")
        )

    resumen = f"""
    - Crecimiento promedio: {crecimiento_promedio:.2f}%
    - Meta establecida: {meta_crecimiento*100:.0f}% mensual
    - Cumplimiento de meta: {porcentaje_sobre_meta:.1f}% de los meses
    - Estabilidad del crecimiento: {estabilidad}
    - Mejor mes: {mejor_mes['nombre_mes']} ({crecimiento_max:.2f}%)
    - Peor mes: {peor_mes['nombre_mes']} ({crecimiento_min:.2f}%)
    - Variabilidad (desviación): {desviacion:.2f}%
    - Meses sobre la meta: {meses_sobre_meta} de {total_meses}
    - Tendencia reciente: {tendencia}
    - Ventas promedio mensuales: ${df['ventas_mes'].mean():,.0f}
    - Cantidad promedio de ventas: {df['cantidad_ventas_mes'].mean():,.0f}
    """

    df_ml = df.copy()
    bloque_anom = ""
    n_anom = 0

    if len(df) >= 6:
        X = df[["crecimiento_ventas"]].fillna(0).values
        pipeline = _cargar_o_entrenar("crecimiento", X, contamination=0.1)
        flags, scores = _detectar(pipeline, X)
        df_ml["_es_anomalia"] = flags
        df_ml["_score_anomalia"] = scores
        media_c = df["crecimiento_ventas"].mean()
        std_c = df["crecimiento_ventas"].std()
        df_ml["_razon_anomalia"] = df_ml.apply(
            lambda r: _zscore_razon(
                r["crecimiento_ventas"], media_c, std_c, "crecimiento mensual"
            )
            or ("variación inusual" if r["_es_anomalia"] else ""),
            axis=1,
        )
        bloque_anom = _bloque_anomalias_texto(df_ml)
        n_anom = int(flags.sum())

    bloque_forecast = _bloque_forecast_texto(df, "ventas_mes", n=3)
    tend_val = (
        (
            df["crecimiento_ventas"].iloc[-3:].mean()
            - df["crecimiento_ventas"].iloc[:3].mean()
        )
        * 100
        if len(df) >= 6
        else 0.0
    )

    bloque_rec = _recomendaciones_ml(
        n_anomalias=n_anom,
        tendencia_pct=tend_val,
        cumplimiento_meta_pct=porcentaje_sobre_meta,
        forecast_mes1=None,
        meta=meta_crecimiento * 100,
        extra="Para estabilizar el crecimiento: fijar micro-metas semanales y revisar KPIs cada 15 días.",
    )

    prompt = f"""
Eres un consultor senior en inteligencia de negocio especializado en análisis de crecimiento.

RESUMEN ESTADÍSTICO:
{resumen}
{bloque_anom}
{bloque_forecast}
{bloque_rec}

INSTRUCCIONES IMPORTANTES:
- NO copies el resumen
- Interpreta el comportamiento del crecimiento integrando los hallazgos ML
- Piensa en sostenibilidad, no en picos aislados
- Habla como asesor estratégico

FORMATO DE RESPUESTA (OBLIGATORIO):

Evaluación de la Sostenibilidad del Crecimiento
(¿Es sostenible o depende de meses excepcionales? ¿Qué detectó el ML?)

Análisis de Consistencia vs Meta
(¿Qué tan consistente es el negocio en alcanzar la meta?)

Patrones Estacionales o Temporales Detectados
(¿Hay patrones repetitivos, incluyendo meses anómalos ML?)

Recomendaciones para Crecimiento Sostenible
(Acciones concretas para mantener/mejorar el crecimiento.)
"""
    return _llamar_ollama(prompt)


def generar_analisis_productos_vendidos(df: pd.DataFrame, meta_productos: float) -> str:
    if df.empty:
        return "No hay datos suficientes para generar un análisis."

    total_productos = df["cantidad_vendida"].sum()
    total_ventas = df["cantidad_ventas"].sum()
    total_tipos = df["tipo_plato"].nunique()
    promedio_x_venta = total_productos / total_ventas if total_ventas > 0 else 0

    df_agrupado = df.groupby("tipo_plato", as_index=False).agg(
        {"cantidad_vendida": "sum", "ventas_totales": "sum"}
    )
    mejor_tipo = df_agrupado.loc[df_agrupado["cantidad_vendida"].idxmax()]
    peor_tipo = df_agrupado.loc[df_agrupado["cantidad_vendida"].idxmin()]
    df_mensual = df.groupby("mes", as_index=False)["cantidad_vendida"].sum()
    meses_sobre_meta = (df_mensual["cantidad_vendida"] >= meta_productos).sum()
    total_meses = len(df_mensual)
    pct_meta = (meses_sobre_meta / total_meses * 100) if total_meses > 0 else 0
    part_mejor = (mejor_tipo["cantidad_vendida"] / total_productos) * 100
    part_peor = (peor_tipo["cantidad_vendida"] / total_productos) * 100

    resumen = f"""
    - Total productos vendidos: {total_productos:,.0f}
    - Meta establecida: {meta_productos:,.0f} productos
    - Meses sobre meta: {meses_sobre_meta} de {total_meses} ({pct_meta:.1f}%)
    - Tipos de plato analizados: {total_tipos}
    - Ventas totales (transacciones): {total_ventas:,.0f}
    - Promedio productos por venta: {promedio_x_venta:.1f}
    - Mejor tipo: {mejor_tipo['tipo_plato']} ({mejor_tipo['cantidad_vendida']:,.0f}, {part_mejor:.1f}%)
    - Peor tipo: {peor_tipo['tipo_plato']} ({peor_tipo['cantidad_vendida']:,.0f}, {part_peor:.1f}%)
    - Ingresos totales: ${df['ventas_totales'].sum():,.0f}
    - Ingresos promedio por tipo: ${df_agrupado['ventas_totales'].mean():,.0f}
    """

    df_ml = df_agrupado.copy()
    bloque_anom = ""
    n_anom = 0

    if len(df_agrupado) >= 4:
        X = df_agrupado[["cantidad_vendida", "ventas_totales"]].fillna(0).values
        pipeline = _cargar_o_entrenar("productos", X, contamination=0.15)
        flags, scores = _detectar(pipeline, X)
        df_ml["_es_anomalia"] = flags
        df_ml["_score_anomalia"] = scores
        media_p = df_agrupado["cantidad_vendida"].mean()
        std_p = df_agrupado["cantidad_vendida"].std()
        df_ml["_razon_anomalia"] = df_ml.apply(
            lambda r: _zscore_razon(
                r["cantidad_vendida"], media_p, std_p, "ventas del tipo"
            )
            or ("participación atípica" if r["_es_anomalia"] else ""),
            axis=1,
        )
        df_ml["nombre_mes"] = df_ml["tipo_plato"]
        bloque_anom = _bloque_anomalias_texto(df_ml)
        n_anom = int(flags.sum())

    bloque_rec = _recomendaciones_ml(
        n_anomalias=n_anom,
        tendencia_pct=0.0,
        cumplimiento_meta_pct=pct_meta,
        forecast_mes1=None,
        meta=meta_productos,
        extra=f"El tipo '{mejor_tipo['tipo_plato']}' lidera con {part_mejor:.1f}%. "
        f"Evaluar bundle con '{peor_tipo['tipo_plato']}' para impulsar categorías débiles.",
    )

    prompt = f"""
Eres un consultor senior en inteligencia de negocio especializado en análisis de productos.

RESUMEN ESTADÍSTICO:
{resumen}
{bloque_anom}
{bloque_rec}

INSTRUCCIONES IMPORTANTES:
- NO copies el resumen
- Interpreta el mix de productos integrando hallazgos ML
- Habla como estratega de producto, no como reporte técnico

FORMATO DE RESPUESTA (OBLIGATORIO):

Análisis del Mix de Productos
(¿Cómo está compuesto el portafolio? ¿Qué detectó el ML sobre categorías atípicas?)

Desempeño por Categoría vs Meta
(¿Qué tipos cumplen consistentemente y cuáles necesitan atención?)

Oportunidades de Optimización de Producto
(¿Cómo mejorar el desempeño de categorías débiles?)

Recomendaciones Estratégicas de Producto
(Acciones específicas basadas en el análisis ML por tipo de plato.)
"""
    return _llamar_ollama(prompt)


def generar_analisis_ventas_promedio_vendedor(
    df: pd.DataFrame, meta_venta_vendedor: float
) -> str:
    if df.empty:
        return "No hay datos suficientes para generar un análisis."

    total_ventas = df["ventas_totales"].sum()
    vendedores_unicos = df["vendedor"].nunique()
    promedio_general = total_ventas / vendedores_unicos if vendedores_unicos > 0 else 0
    vendedores_sobre_meta = df[df["cumple_meta"]].shape[0]
    total_v_meses = df.shape[0]
    pct_sobre_meta = (
        (vendedores_sobre_meta / total_v_meses * 100) if total_v_meses > 0 else 0
    )
    mejor_vendedor = df.loc[df["ventas_totales"].idxmax()]
    peor_vendedor = df.loc[df["ventas_totales"].idxmin()]

    df_mensual = df.groupby(["mes", "nombre_mes"], as_index=False).agg(
        {"ventas_promedio_por_vendedor": "first", "cumple_meta_promedio": "first"}
    )
    meses_sobre_meta = df_mensual["cumple_meta_promedio"].sum()
    total_meses = len(df_mensual)
    promedio_x_venta = df["promedio_por_venta"].mean()
    clientes_prom = df["clientes_atendidos"].sum() / vendedores_unicos
    productos_prom = df["productos_vendidos"].sum() / vendedores_unicos

    resumen = f"""
    - Ventas promedio por vendedor: ${promedio_general:,.0f}
    - Meta por vendedor: ${meta_venta_vendedor:,.0f}
    - Cumplimiento de meta: {pct_sobre_meta:.1f}% de vendedores-mes
    - Vendedores únicos: {vendedores_unicos}
    - Meses que cumplen meta promedio: {meses_sobre_meta} de {total_meses}
    - Mejor vendedor: {mejor_vendedor['vendedor']} (${mejor_vendedor['ventas_totales']:,.0f})
    - Peor vendedor: {peor_vendedor['vendedor']} (${peor_vendedor['ventas_totales']:,.0f})
    - Promedio por venta: ${promedio_x_venta:,.0f}
    - Clientes promedio por vendedor: {clientes_prom:.0f}
    - Productos promedio por vendedor: {productos_prom:.0f}
    - Ventas totales del equipo: ${total_ventas:,.0f}
    """

    df_vendedor = df.groupby("vendedor", as_index=False).agg(
        {
            "ventas_totales": "sum",
            "clientes_atendidos": "sum",
            "productos_vendidos": "sum",
        }
    )
    df_ml = df_vendedor.copy()
    bloque_anom = ""
    n_anom = 0

    if len(df_vendedor) >= 4:
        X = df_vendedor[["ventas_totales", "clientes_atendidos"]].fillna(0).values
        pipeline = _cargar_o_entrenar("vendedores", X, contamination=0.15)
        flags, scores = _detectar(pipeline, X)
        df_ml["_es_anomalia"] = flags
        df_ml["_score_anomalia"] = scores
        media_v = df_vendedor["ventas_totales"].mean()
        std_v = df_vendedor["ventas_totales"].std()
        df_ml["_razon_anomalia"] = df_ml.apply(
            lambda r: _zscore_razon(
                r["ventas_totales"], media_v, std_v, "ventas del vendedor"
            )
            or ("desempeño atípico" if r["_es_anomalia"] else ""),
            axis=1,
        )
        df_ml["nombre_mes"] = df_ml["vendedor"]
        bloque_anom = _bloque_anomalias_texto(df_ml)
        n_anom = int(flags.sum())

    brecha_pct = (
        (mejor_vendedor["ventas_totales"] - peor_vendedor["ventas_totales"])
        / peor_vendedor["ventas_totales"]
        * 100
        if peor_vendedor["ventas_totales"] > 0
        else 0
    )

    bloque_rec = _recomendaciones_ml(
        n_anomalias=n_anom,
        tendencia_pct=0.0,
        cumplimiento_meta_pct=pct_sobre_meta,
        forecast_mes1=None,
        meta=meta_venta_vendedor,
        extra=f"Brecha entre mejor y peor vendedor: {brecha_pct:.0f}%. "
        "Implementar mentoring entre pares y revisar la distribución de zonas/mesas.",
    )

    prompt = f"""
Eres un consultor senior en inteligencia de negocio especializado en gestión de equipos de ventas.

RESUMEN ESTADÍSTICO:
{resumen}
{bloque_anom}
{bloque_rec}

INSTRUCCIONES IMPORTANTES:
- NO copies el resumen
- Interpreta el desempeño del equipo integrando anomalías ML (vendedores atípicos)
- Piensa en estrategias de gestión y motivación
- Habla como asesor de ventas senior

FORMATO DE RESPUESTA (OBLIGATORIO):

Evaluación del Desempeño del Equipo
(¿Cómo funciona el equipo como conjunto? ¿Qué detectó el ML sobre vendedores atípicos?)

Análisis de Consistencia vs Meta
(¿Qué tan consistente es el equipo en alcanzar la meta?)

Identificación de Oportunidades de Mejora
(¿Qué áreas del equipo necesitan más desarrollo?)

Recomendaciones para Gestión de Equipos
(Acciones específicas para mejorar el desempeño promedio, basadas en ML.)
"""
    return _llamar_ollama(prompt)


def generar_analisis_participacion_productos(
    df: pd.DataFrame, meta_participacion: float
) -> str:
    if df.empty:
        return "No hay datos suficientes para generar un análisis."

    total_productos = df.shape[0]
    productos_sobre_meta = df[df["cumple_meta"]].shape[0]
    pct_sobre_meta = (
        (productos_sobre_meta / total_productos * 100) if total_productos > 0 else 0
    )
    producto_lider = df.loc[df["participacion_porcentual"].idxmax()]
    producto_debil = df.loc[df["participacion_porcentual"].idxmin()]
    part_promedio = df["participacion_porcentual"].mean()
    desv_part = df["participacion_porcentual"].std()
    concentracion_top3 = df.nlargest(3, "participacion_porcentual")[
        "participacion_porcentual"
    ].sum()
    concentracion_top5 = df.nlargest(5, "participacion_porcentual")[
        "participacion_porcentual"
    ].sum()
    concentracion_top10 = df.nlargest(10, "participacion_porcentual")[
        "participacion_porcentual"
    ].sum()

    df_cat = df.groupby("categoria", as_index=False).agg(
        {"participacion_porcentual": "sum", "producto": "count"}
    )
    cat_lider = df_cat.loc[df_cat["participacion_porcentual"].idxmax()]
    cat_debil = df_cat.loc[df_cat["participacion_porcentual"].idxmin()]
    diversificacion = (
        "Alta"
        if concentracion_top3 < 50
        else "Media" if concentracion_top3 < 70 else "Baja"
    )

    resumen = f"""
    - Total productos analizados: {total_productos}
    - Productos sobre meta ({meta_participacion*100:.0f}%): {productos_sobre_meta} ({pct_sobre_meta:.1f}%)
    - Participación promedio: {part_promedio:.2f}%
    - Desviación estándar: {desv_part:.2f}%
    - Concentración Top 3: {concentracion_top3:.1f}% | Top 5: {concentracion_top5:.1f}% | Top 10: {concentracion_top10:.1f}%
    - Producto líder: {producto_lider['producto']} ({producto_lider['participacion_porcentual']:.2f}%)
    - Producto más débil: {producto_debil['producto']} ({producto_debil['participacion_porcentual']:.2f}%)
    - Categoría líder: {cat_lider['categoria']} ({cat_lider['participacion_porcentual']:.1f}%, {cat_lider['producto']} productos)
    - Categoría más débil: {cat_debil['categoria']} ({cat_debil['participacion_porcentual']:.1f}%, {cat_debil['producto']} productos)
    - Nivel de diversificación: {diversificacion}
    - Ventas totales: ${df['ventas_totales'].sum():,.0f}
    - Unidades totales vendidas: {df['unidades_vendidas'].sum():,.0f}
    """

    df_ml = df.copy()
    bloque_anom = ""
    n_anom = 0

    if len(df) >= 6:
        X = df[["participacion_porcentual", "unidades_vendidas"]].fillna(0).values
        pipeline = _cargar_o_entrenar("productos", X, contamination=0.1)
        flags, scores = _detectar(pipeline, X)
        df_ml["_es_anomalia"] = flags
        df_ml["_score_anomalia"] = scores
        media_p = df["participacion_porcentual"].mean()
        std_p = df["participacion_porcentual"].std()
        df_ml["_razon_anomalia"] = df_ml.apply(
            lambda r: _zscore_razon(
                r["participacion_porcentual"], media_p, std_p, "participación"
            )
            or ("participación atípica" if r["_es_anomalia"] else ""),
            axis=1,
        )
        df_ml["nombre_mes"] = df_ml["producto"]
        bloque_anom = _bloque_anomalias_texto(df_ml)
        n_anom = int(flags.sum())

    bloque_rec = _recomendaciones_ml(
        n_anomalias=n_anom,
        tendencia_pct=0.0,
        cumplimiento_meta_pct=pct_sobre_meta,
        forecast_mes1=None,
        meta=meta_participacion * 100,
        extra=f"Diversificación {diversificacion}. "
        + (
            "Top 3 concentran demasiado: crear combos que incluyan productos de menor participación."
            if diversificacion == "Baja"
            else "Potenciar categorías débiles con promociones cruzadas."
        ),
    )

    prompt = f"""
Eres un consultor senior en inteligencia de negocio especializado en análisis de portafolio de productos.

RESUMEN ESTADÍSTICO:
{resumen}
{bloque_anom}
{bloque_rec}

INSTRUCCIONES IMPORTANTES:
- NO copies el resumen
- Interpreta la estructura del portafolio integrando los hallazgos ML
- Habla como estratega de producto senior

FORMATO DE RESPUESTA (OBLIGATORIO):

Evaluación del Balance del Portafolio
(¿Cómo está distribuida la participación? ¿Qué productos son estadísticamente atípicos según el ML?)

Análisis de Concentración vs Diversificación
(¿El portafolio depende de pocos productos? Considera las alertas ML.)

Identificación de Riesgos y Oportunidades
(¿Qué productos/categorías representan riesgo u oportunidad, cruzando datos con ML?)

Recomendaciones Estratégicas de Portafolio
(Acciones específicas para optimizar la participación, apoyadas en el análisis ML.)
"""
    return _llamar_ollama(prompt)
