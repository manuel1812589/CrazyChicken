import pandas as pd
from sqlalchemy import text
from app.db.connection_mart import get_mart_engine


def get_ventas_mensuales_2024(mes_filtro=None):
    engine = get_mart_engine()

    query = text(
        """
        SELECT
            dt.año,
            dt.mes,
            dt.nombre_mes,
            SUM(hv.total_linea) AS ventas_mes
        FROM hechos_ventas hv
        INNER JOIN dim_tiempo dt ON hv.id_tiempo = dt.id_tiempo
        WHERE dt.año = 2024
    """
    )

    if mes_filtro:
        if isinstance(mes_filtro, list):
            mes_list = ", ".join(str(m) for m in mes_filtro)
            query = text(
                f"""
                {query.text}
                AND dt.mes IN ({mes_list})
            """
            )
        else:
            query = text(
                f"""
                {query.text}
                AND dt.mes = {mes_filtro}
            """
            )

    query = text(
        f"""
        {query.text}
        GROUP BY dt.año, dt.mes, dt.nombre_mes
        ORDER BY dt.mes
    """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return df

    df = df.sort_values("mes").reset_index(drop=True)

    df["ventas_acumuladas"] = df["ventas_mes"].cumsum()

    meta_mensual = df["ventas_mes"].mean() * 1.10

    df["meta_mensual"] = meta_mensual
    df["meta_acumulada"] = meta_mensual * (df.index + 1)

    df["desviacion_meta"] = df["ventas_acumuladas"] - df["meta_acumulada"]

    return df


def get_ticket_promedio_global_2024(mes_filtro=None):
    engine = get_mart_engine()

    query = text(
        """
        SELECT
            dt.año,
            dt.mes,
            dt.nombre_mes,
            SUM(hv.total_linea) AS ventas_totales,
            COUNT(DISTINCT hv.id_venta) AS cantidad_ventas,
            CASE 
                WHEN COUNT(DISTINCT hv.id_venta) > 0 
                THEN CAST(SUM(hv.total_linea) AS FLOAT) / COUNT(DISTINCT hv.id_venta)
                ELSE 0 
            END AS ticket_promedio
        FROM hechos_ventas hv
        INNER JOIN dim_tiempo dt ON hv.id_tiempo = dt.id_tiempo
        WHERE dt.año = 2024
    """
    )

    if mes_filtro:
        if isinstance(mes_filtro, list):
            mes_list = ", ".join(str(m) for m in mes_filtro)
            query = text(
                f"""
                {query.text}
                AND dt.mes IN ({mes_list})
            """
            )
        else:
            query = text(
                f"""
                {query.text}
                AND dt.mes = {mes_filtro}
            """
            )

    query = text(
        f"""
        {query.text}
        GROUP BY dt.año, dt.mes, dt.nombre_mes
        ORDER BY dt.mes
    """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    df["meta_ticket"] = 70

    return df


def get_cantidad_ventas_mensuales_2024(mes_filtro=None):
    engine = get_mart_engine()

    query = text(
        """
        SELECT
            dt.año,
            dt.mes,
            dt.nombre_mes,
            COUNT(DISTINCT hv.id_venta) AS cantidad_ventas
        FROM hechos_ventas hv
        INNER JOIN dim_tiempo dt ON hv.id_tiempo = dt.id_tiempo
        WHERE dt.año = 2024
        """
    )

    if mes_filtro:
        if isinstance(mes_filtro, list):
            mes_list = ", ".join(str(m) for m in mes_filtro)
            query = text(
                f"""
                {query.text}
                AND dt.mes IN ({mes_list})
                """
            )
        else:
            query = text(
                f"""
                {query.text}
                AND dt.mes = {mes_filtro}
                """
            )

    query = text(
        f"""
        {query.text}
        GROUP BY dt.año, dt.mes, dt.nombre_mes
        ORDER BY dt.mes
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    df["cantidad_acumulada"] = df["cantidad_ventas"].cumsum()
    df["meta_mensual"] = 2400
    df["meta_acumulada"] = 2400 * (df.index + 1)

    return df


def get_crecimiento_ventas_mensual_2024(mes_filtro=None):
    """
    Obtiene el crecimiento porcentual de ventas mensuales comparando con el mes anterior.
    Meta de crecimiento: 2% (0.02)
    """
    engine = get_mart_engine()

    query = text(
        """
        SELECT
            dt.año,
            dt.mes,
            dt.nombre_mes,
            SUM(hv.total_linea) AS ventas_mes,
            LAG(SUM(hv.total_linea), 1) OVER (ORDER BY dt.mes) AS ventas_mes_anterior,
            COUNT(DISTINCT hv.id_venta) AS cantidad_ventas_mes,
            LAG(COUNT(DISTINCT hv.id_venta), 1) OVER (ORDER BY dt.mes) AS cantidad_ventas_mes_anterior
        FROM hechos_ventas hv
        INNER JOIN dim_tiempo dt ON hv.id_tiempo = dt.id_tiempo
        WHERE dt.año = 2024
        GROUP BY dt.año, dt.mes, dt.nombre_mes
        ORDER BY dt.mes
    """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return df

    df["crecimiento_ventas"] = df.apply(
        lambda row: (
            (row["ventas_mes"] - row["ventas_mes_anterior"])
            / row["ventas_mes_anterior"]
            if row["ventas_mes_anterior"] and row["ventas_mes_anterior"] > 0
            else None
        ),
        axis=1,
    )

    df["crecimiento_cantidad"] = df.apply(
        lambda row: (
            (row["cantidad_ventas_mes"] - row["cantidad_ventas_mes_anterior"])
            / row["cantidad_ventas_mes_anterior"]
            if row["cantidad_ventas_mes_anterior"]
            and row["cantidad_ventas_mes_anterior"] > 0
            else None
        ),
        axis=1,
    )

    df["meta_crecimiento"] = 0.02  # 2%

    df["cumple_meta"] = df["crecimiento_ventas"] >= df["meta_crecimiento"]

    df["crecimiento_ventas_pct"] = df["crecimiento_ventas"].apply(
        lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
    )

    df["crecimiento_cantidad_pct"] = df["crecimiento_cantidad"].apply(
        lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A"
    )

    if mes_filtro and mes_filtro != "all":
        if isinstance(mes_filtro, list):
            df = df[df["mes"].isin(mes_filtro)]
        else:
            df = df[df["mes"] == int(mes_filtro)]

    return df


def get_productos_vendidos_por_tipo_2024(mes_filtro=None, tipo_filtro=None):
    engine = get_mart_engine()

    query = """
        SELECT
            dt.mes,
            dt.nombre_mes,
            dp.tipo AS tipo_plato,
            SUM(hv.cantidad_vendida) AS cantidad_vendida,
            COUNT(DISTINCT hv.id_venta) AS cantidad_ventas,
            SUM(hv.total_linea) AS ventas_totales
        FROM hechos_ventas hv
        INNER JOIN dim_tiempo dt ON hv.id_tiempo = dt.id_tiempo
        INNER JOIN dim_productos dp ON hv.id_producto = dp.id_producto
        WHERE dt.año = 2024
    """

    params = {}

    if mes_filtro and mes_filtro != "all":
        query += " AND dt.mes = :mes"
        params["mes"] = int(mes_filtro)

    if tipo_filtro and tipo_filtro != "all":
        query += " AND dp.tipo = :tipo"
        params["tipo"] = tipo_filtro

    query += """
        GROUP BY dt.mes, dt.nombre_mes, dp.tipo
        ORDER BY dt.mes, cantidad_vendida DESC
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        return df

    df["meta_productos"] = 1600
    df["cumple_meta"] = df["cantidad_vendida"] >= df["meta_productos"]

    return df


def get_ventas_promedio_por_vendedor_2024(mes_filtro=None):
    engine = get_mart_engine()

    query_base = """
        SELECT
            dt.año,
            dt.mes,
            dt.nombre_mes,
            dv.nombre_completo AS vendedor,
            COUNT(DISTINCT hv.id_venta) AS cantidad_ventas,
            SUM(hv.total_linea) AS ventas_totales,
            COUNT(DISTINCT hv.id_cliente) AS clientes_atendidos,
            AVG(hv.total_linea) AS promedio_por_venta,
            SUM(hv.cantidad_vendida) AS productos_vendidos
        FROM hechos_ventas hv
        INNER JOIN dim_tiempo dt ON hv.id_tiempo = dt.id_tiempo
        INNER JOIN dim_vendedores dv ON hv.id_vendedor = dv.id_vendedor
        WHERE dt.año = 2024
    """

    if mes_filtro and mes_filtro != "all":
        if isinstance(mes_filtro, list):
            mes_list = ", ".join(str(m) for m in mes_filtro)
            query_base += f" AND dt.mes IN ({mes_list})"
        else:
            query_base += f" AND dt.mes = {mes_filtro}"

    query_base += """
        GROUP BY dt.año, dt.mes, dt.nombre_mes, dv.nombre_completo
        ORDER BY dt.mes, dv.nombre_completo
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query_base), conn)

    if not df.empty:
        df["meta_venta_vendedor"] = 20000
        df["cumple_meta"] = df["ventas_totales"] >= df["meta_venta_vendedor"]

        df_mensual = df.groupby(["mes", "nombre_mes"], as_index=False).agg(
            {"ventas_totales": "sum", "vendedor": "nunique"}
        )

        df_mensual["ventas_promedio_por_vendedor"] = (
            df_mensual["ventas_totales"] / df_mensual["vendedor"]
        )
        df_mensual["cumple_meta_promedio"] = (
            df_mensual["ventas_promedio_por_vendedor"] >= 20000
        )

        df = pd.merge(
            df,
            df_mensual[["mes", "ventas_promedio_por_vendedor", "cumple_meta_promedio"]],
            on="mes",
        )

    return df


def get_participacion_productos_2024(mes_filtro=None):
    engine = get_mart_engine()

    query_base = """
        SELECT
            dp.nombre AS producto,
            dp.tipo AS categoria,
            COUNT(DISTINCT hv.id_venta) AS cantidad_ventas,
            SUM(hv.cantidad_vendida) AS unidades_vendidas,
            SUM(hv.total_linea) AS ventas_totales,
            AVG(hv.precio_unitario) AS precio_promedio,
            COUNT(DISTINCT hv.id_cliente) AS clientes_unicos
        FROM hechos_ventas hv
        INNER JOIN dim_tiempo dt ON hv.id_tiempo = dt.id_tiempo
        INNER JOIN dim_productos dp ON hv.id_producto = dp.id_producto
        WHERE dt.año = 2024
    """

    if mes_filtro and mes_filtro != "all":
        if isinstance(mes_filtro, list):
            mes_list = ", ".join(str(m) for m in mes_filtro)
            query_base += f" AND dt.mes IN ({mes_list})"
        else:
            query_base += f" AND dt.mes = {mes_filtro}"

    query_base += """
        GROUP BY dp.nombre, dp.tipo
        ORDER BY ventas_totales DESC
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query_base), conn)

    if not df.empty:
        total_ventas = df["ventas_totales"].sum()
        df["participacion_porcentual"] = (df["ventas_totales"] / total_ventas) * 100

        df["participacion_decimal"] = df["ventas_totales"] / total_ventas

        df["meta_participacion"] = 0.10

        df["cumple_meta"] = df["participacion_decimal"] >= df["meta_participacion"]

        df["participacion_formateada"] = df["participacion_porcentual"].apply(
            lambda x: f"{x:.2f}%"
        )

        df["ranking"] = (
            df["ventas_totales"].rank(method="dense", ascending=False).astype(int)
        )

        df["clasificacion"] = pd.cut(
            df["participacion_decimal"],
            bins=[0, 0.05, 0.10, 0.20, 1.0],
            labels=["Baja", "Media", "Alta", "Muy Alta"],
        )

    return df
