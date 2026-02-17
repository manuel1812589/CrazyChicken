import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.services.data_service import get_productos_vendidos_por_tipo_2024
from app.components.graph_card import create_graph_card
from app.services.ai_service import (
    generar_analisis_productos_vendidos,
    generar_respuesta_chat,
)
from app.services.ml_service import generar_contexto_ml_productos
import time

dash.register_page(__name__, path="/kpi-productos-vendidos")

df_all = get_productos_vendidos_por_tipo_2024()
meses_disponibles = [
    {"label": row["nombre_mes"], "value": row["mes"]}
    for _, row in df_all.sort_values("mes").drop_duplicates("mes").iterrows()
]

tipos_disponibles = [
    {"label": tipo, "value": tipo} for tipo in sorted(df_all["tipo_plato"].unique())
]

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H1(
                                    "Productos Vendidos por Tipo",
                                    className="page-title mb-2",
                                ),
                                html.P(
                                    "Análisis de productos vendidos por tipo de plato y comparación con meta",
                                    className="page-subtitle text-muted",
                                ),
                            ],
                            className="page-header",
                        )
                    ]
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        create_graph_card(
                            graph=html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "Filtrar por mes:",
                                                className="form-label",
                                            ),
                                            dcc.Dropdown(
                                                id="productos-mes-filter",
                                                options=[
                                                    {
                                                        "label": "Todos los meses",
                                                        "value": "all",
                                                    }
                                                ]
                                                + meses_disponibles,
                                                value="all",
                                                placeholder="Selecciona un mes...",
                                                className="mb-3",
                                            ),
                                        ],
                                        style={
                                            "position": "relative",
                                            "zIndex": 20,
                                            "marginBottom": "20px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "Filtrar por tipo de plato:",
                                                className="form-label",
                                            ),
                                            dcc.Dropdown(
                                                id="productos-tipo-filter",
                                                options=[
                                                    {
                                                        "label": "Todos los tipos",
                                                        "value": "all",
                                                    }
                                                ]
                                                + tipos_disponibles,
                                                value="all",
                                                placeholder="Selecciona un tipo de plato...",
                                                className="mb-3",
                                            ),
                                        ],
                                        style={
                                            "position": "relative",
                                            "zIndex": 10,
                                            "marginBottom": "10px",
                                        },
                                    ),
                                    html.Small(
                                        "Nota: El año siempre será 2024 • Meta: 1,600 productos vendidos",
                                        className="text-muted",
                                    ),
                                ],
                                style={"position": "relative"},
                            ),
                            title="Filtros",
                            description="Selecciona meses y tipos específicos para analizar",
                            height=250,
                        )
                    ],
                    lg=6,
                    md=12,
                    className="mb-4",
                )
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Análisis Inteligente con IA", className="section-title mb-3"),
                dbc.Spinner(
                    html.Div(
                        id="productos-ai-analysis-output",
                        className="p-3 border rounded bg-light",
                        style={"whiteSpace": "pre-line"},
                    ),
                    id="productos-analysis-spinner",
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Asistente IA de Negocio", className="section-title mb-3"),
                dcc.Store(id="productos-chat-history", data=[]),
                dcc.Store(id="productos-ai-thinking", data=False),
                html.Div(
                    id="productos-chat-window",
                    className="chat-window mb-3",
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
                html.Div(
                    id="productos-typing-indicator",
                    className="typing-indicator",
                    style={"display": "none"},
                    children=html.Div(
                        [
                            dbc.Spinner(size="sm", color="primary"),
                            html.Span(" Asistente está pensando...", className="ms-2"),
                        ]
                    ),
                ),
                dbc.InputGroup(
                    [
                        dbc.Input(
                            id="productos-chat-input",
                            placeholder="Haz una pregunta sobre productos vendidos por tipo...",
                            type="text",
                            disabled=False,
                        ),
                        dbc.Button(
                            "Enviar",
                            id="productos-chat-send",
                            color="primary",
                            disabled=False,
                        ),
                    ],
                    className="mb-3",
                ),
            ],
            className="chat-container mb-5",
        ),
        html.Div(
            [
                html.H4(
                    "Distribución por Tipo de Plato", className="section-title mb-4"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="productos-bar-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Productos Vendidos por Tipo",
                                description="Cantidad de productos vendidos por tipo de plato",
                                height=450,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="productos-pie-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Distribución por Tipo",
                                description="Participación porcentual de cada tipo de plato",
                                height=450,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                    ],
                    className="mb-4",
                ),
            ]
        ),
        html.Div(
            [
                html.H4("Análisis de Tendencias", className="section-title mb-4"),
                create_graph_card(
                    graph=html.Div(
                        [
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        id="productos-view-toggle",
                                        className="btn-group",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[
                                            {
                                                "label": "Por Mes",
                                                "value": "mensual",
                                            },
                                            {
                                                "label": "Comparación con Meta",
                                                "value": "meta",
                                            },
                                            {
                                                "label": "Evolución por Tipo",
                                                "value": "tipo",
                                            },
                                        ],
                                        value="mensual",
                                    ),
                                ],
                                className="radio-group mb-4",
                            ),
                            html.Div(id="productos-dynamic-graph-container"),
                        ]
                    ),
                    title="Análisis Detallado",
                    description="Diferentes perspectivas del desempeño de productos",
                    height=500,
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("KPIs de Productos", className="section-title mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=html.Div(
                                    [
                                        html.Div(
                                            id="productos-kpi-cards",
                                            className="kpi-cards-grid",
                                        ),
                                    ]
                                ),
                                title="Métricas Clave",
                                description="Resumen del desempeño de productos vendidos",
                                height=200,
                            ),
                            lg=12,
                            md=12,
                            className="mb-4",
                        ),
                    ],
                    className="mb-4",
                ),
            ]
        ),
        html.Div(
            [
                html.Hr(className="my-4"),
                html.Div(
                    [
                        html.Small(
                            "Dashboard de Productos Vendidos • Actualizado: Diciembre 2024",
                            className="text-muted",
                        ),
                    ],
                    className="text-center",
                ),
            ],
            className="app-footer mt-5",
        ),
    ],
    className="kpi-page-container",
)


@callback(
    [
        Output("productos-bar-graph", "figure"),
        Output("productos-pie-graph", "figure"),
    ],
    [
        Input("productos-mes-filter", "value"),
        Input("productos-tipo-filter", "value"),
    ],
)
def update_static_graphs(mes_filtro, tipo_filtro):
    df = get_productos_vendidos_por_tipo_2024(mes_filtro, tipo_filtro)

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_text="No hay datos para los filtros seleccionados",
            annotations=[
                dict(
                    text="No hay datos",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return empty_fig, empty_fig

    df_agrupado = df.groupby("tipo_plato", as_index=False).agg(
        {"cantidad_vendida": "sum", "ventas_totales": "sum", "cantidad_ventas": "sum"}
    )

    bar_fig = px.bar(
        df_agrupado,
        x="tipo_plato",
        y="cantidad_vendida",
        title="",
        labels={
            "cantidad_vendida": "Cantidad Vendida",
            "tipo_plato": "Tipo de Plato",
            "ventas_totales": "Ventas Totales",
        },
        color="cantidad_vendida",
        color_continuous_scale="Viridis",
    )

    if mes_filtro and mes_filtro != "all":
        title_text = f"Productos Vendidos - Mes Seleccionado"
    else:
        title_text = "Productos Vendidos por Tipo de Plato 2024"

    bar_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        title_text=title_text,
        title_font=dict(size=18, color="var(--primary-color)"),
        xaxis_title="Tipo de Plato",
        yaxis_title="Cantidad Vendida",
        coloraxis_showscale=False,
    )
    bar_fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Cantidad: %{y:,.0f}<extra></extra>",
    )

    pie_fig = px.pie(
        df_agrupado,
        names="tipo_plato",
        values="cantidad_vendida",
        title="",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Plasma,
    )

    pie_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.2),
        title_text="Distribución por Tipo de Plato",
        title_font=dict(size=18, color="var(--primary-color)"),
    )
    pie_fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Cantidad: %{value:,.0f}<br>%{percent}<extra></extra>",
    )

    return bar_fig, pie_fig


@callback(
    Output("productos-dynamic-graph-container", "children"),
    [
        Input("productos-view-toggle", "value"),
        Input("productos-mes-filter", "value"),
        Input("productos-tipo-filter", "value"),
    ],
)
def update_dynamic_graph(view_type, mes_filtro, tipo_filtro):
    df = get_productos_vendidos_por_tipo_2024(mes_filtro, tipo_filtro)

    if df.empty:
        return html.Div(
            "No hay datos para los filtros seleccionados",
            className="text-center text-muted py-5",
        )

    if view_type == "mensual":
        df_mensual = (
            df.groupby(["nombre_mes", "mes"], as_index=False)
            .agg(
                {
                    "cantidad_vendida": "sum",
                    "ventas_totales": "sum",
                    "cantidad_ventas": "sum",
                }
            )
            .sort_values("mes")
        )

        df_mensual["meta_productos"] = 1600

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df_mensual["nombre_mes"],
                    y=df_mensual["cantidad_vendida"],
                    mode="lines+markers",
                    name="Productos Vendidos",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=10, color="#3498db"),
                    hovertemplate="<b>%{x}</b><br>Cantidad: %{y:,.0f}<extra></extra>",
                ),
                go.Scatter(
                    x=df_mensual["nombre_mes"],
                    y=df_mensual["meta_productos"],
                    mode="lines",
                    name="Meta (1,600)",
                    line=dict(color="#e74c3c", width=2, dash="dash"),
                    hovertemplate="Meta: 1,600<extra></extra>",
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Productos Vendidos por Mes vs Meta",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Cantidad de Productos",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            ),
        )

    elif view_type == "meta":
        df_agrupado = df.groupby("tipo_plato", as_index=False).agg(
            {"cantidad_vendida": "sum", "meta_productos": "first"}
        )

        df_agrupado["cumple_meta"] = (
            df_agrupado["cantidad_vendida"] >= df_agrupado["meta_productos"]
        )

        colores = [
            "#2ecc71" if cumple else "#e74c3c" for cumple in df_agrupado["cumple_meta"]
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_agrupado["tipo_plato"],
                    y=df_agrupado["cantidad_vendida"],
                    name="Vendidos",
                    marker_color=colores,
                    hovertemplate="<b>%{x}</b><br>Cantidad: %{y:,.0f}<extra></extra>",
                ),
                go.Scatter(
                    x=df_agrupado["tipo_plato"],
                    y=df_agrupado["meta_productos"],
                    mode="lines+markers",
                    name="Meta (1,600)",
                    line=dict(color="#34495e", width=2, dash="dot"),
                    marker=dict(size=8, color="#34495e"),
                    hovertemplate="Meta: 1,600<extra></extra>",
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Productos Vendidos por Tipo vs Meta",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Tipo de Plato",
                yaxis_title="Cantidad",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            ),
        )

    else:
        if len(df["tipo_plato"].unique()) > 1:
            fig = px.line(
                df.sort_values(["mes", "tipo_plato"]),
                x="nombre_mes",
                y="cantidad_vendida",
                color="tipo_plato",
                title="",
                markers=True,
                labels={
                    "cantidad_vendida": "Cantidad Vendida",
                    "nombre_mes": "Mes",
                    "tipo_plato": "Tipo de Plato",
                },
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Evolución de Productos Vendidos por Tipo",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Cantidad Vendida",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            )

            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{fullData.name}</b><br>Mes: %{x}<br>Cantidad: %{y:,.0f}<extra></extra>",
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_text="Selecciona 'Todos los tipos' para ver la evolución por tipo",
                annotations=[
                    dict(
                        text="Se necesita más de un tipo para comparar",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16),
                    )
                ],
            )

    return dcc.Graph(
        id="productos-line-graph",
        figure=fig,
        config={"displayModeBar": True},
        className="dynamic-graph",
    )


@callback(
    Output("productos-kpi-cards", "children"),
    [
        Input("productos-mes-filter", "value"),
        Input("productos-tipo-filter", "value"),
    ],
)
def update_kpi_cards(mes_filtro, tipo_filtro):
    df = get_productos_vendidos_por_tipo_2024(mes_filtro, tipo_filtro)

    if df.empty:
        return html.Div(
            "No hay datos para los filtros seleccionados",
            className="text-center text-muted py-3",
        )

    total_productos = df["cantidad_vendida"].sum()
    total_ventas = df["cantidad_ventas"].sum()
    total_tipos = df["tipo_plato"].nunique()
    promedio_por_venta = total_productos / total_ventas if total_ventas > 0 else 0

    meta = 1600
    meses_sobre_meta = (df.groupby("mes")["cantidad_vendida"].sum() >= meta).sum()
    total_meses = df["mes"].nunique()

    mejor_tipo = (
        df.loc[df["cantidad_vendida"].idxmax(), "tipo_plato"] if not df.empty else "N/A"
    )
    mejor_cantidad = df["cantidad_vendida"].max() if not df.empty else 0

    cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(f"{total_productos:,.0f}", className="card-title"),
                        html.P(
                            "Total Productos Vendidos", className="card-text text-muted"
                        ),
                    ]
                ),
                className="text-center h-100 border-primary",
            ),
            md=3,
            sm=6,
            className="mb-3",
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(
                            f"{meses_sobre_meta}/{total_meses}", className="card-title"
                        ),
                        html.P("Meses sobre Meta", className="card-text text-muted"),
                    ]
                ),
                className="text-center h-100 border-success",
            ),
            md=3,
            sm=6,
            className="mb-3",
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(f"{total_tipos}", className="card-title"),
                        html.P("Tipos de Plato", className="card-text text-muted"),
                    ]
                ),
                className="text-center h-100 border-info",
            ),
            md=3,
            sm=6,
            className="mb-3",
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(f"{promedio_por_venta:.1f}", className="card-title"),
                        html.P("Productos/Venta", className="card-text text-muted"),
                    ]
                ),
                className="text-center h-100 border-warning",
            ),
            md=3,
            sm=6,
            className="mb-3",
        ),
    ]

    return dbc.Row(cards)


@callback(
    Output("productos-ai-analysis-output", "children"),
    Output("productos-analysis-spinner", "type"),
    [
        Input("productos-mes-filter", "value"),
        Input("productos-tipo-filter", "value"),
    ],
)
def update_ai_analysis(mes_filtro, tipo_filtro):
    df = get_productos_vendidos_por_tipo_2024(mes_filtro, tipo_filtro)

    if df.empty:
        return "No hay datos para generar análisis.", "border"

    meta_productos = 1600
    analisis = generar_analisis_productos_vendidos(df, meta_productos)

    return analisis, "border"


@callback(
    Output("productos-chat-history", "data", allow_duplicate=True),
    Output("productos-chat-input", "value"),
    Output("productos-typing-indicator", "style"),
    Output("productos-ai-thinking", "data"),
    Output("productos-chat-input", "disabled"),
    Output("productos-chat-send", "disabled"),
    Input("productos-chat-send", "n_clicks"),
    State("productos-chat-input", "value"),
    State("productos-chat-history", "data"),
    State("productos-ai-thinking", "data"),
    prevent_initial_call=True,
)
def handle_user_message(n_clicks, user_msg, history, is_thinking):
    if not user_msg or user_msg.strip() == "" or is_thinking:
        return dash.no_update, "", {"display": "none"}, is_thinking, False, False

    history = history or []

    history.append(
        {"role": "user", "text": user_msg, "id": len(history), "timestamp": time.time()}
    )

    history.append(
        {
            "role": "ai",
            "text": "",
            "id": len(history),
            "timestamp": time.time(),
            "is_loading": True,
        }
    )

    return (
        history,
        "",
        {"display": "flex", "alignItems": "center"},
        True,
        True,
        True,
    )


@callback(
    Output("productos-chat-history", "data"),
    Output("productos-typing-indicator", "style", allow_duplicate=True),
    Output("productos-ai-thinking", "data", allow_duplicate=True),
    Output("productos-chat-input", "disabled", allow_duplicate=True),
    Output("productos-chat-send", "disabled", allow_duplicate=True),
    Input("productos-ai-thinking", "data"),
    State("productos-chat-history", "data"),
    State("productos-mes-filter", "value"),
    State("productos-tipo-filter", "value"),
    prevent_initial_call=True,
)
def generate_ai_response(is_thinking, history, mes_filtro, tipo_filtro):
    if not is_thinking or not history:
        return dash.no_update, {"display": "none"}, False, False, False

    user_messages = [msg for msg in history if msg.get("role") == "user"]
    if not user_messages:
        return dash.no_update, {"display": "none"}, False, False, False

    last_user_msg = user_messages[-1]["text"]

    df = get_productos_vendidos_por_tipo_2024(mes_filtro, tipo_filtro)

    if df.empty:
        ai_response = "No hay datos disponibles para responder tu pregunta."
    else:
        meta_productos = 1600
        contexto = generar_contexto_ml_productos(df, meta_productos)
        ai_response = generar_respuesta_chat(contexto, last_user_msg)

    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "ai" and history[i].get("is_loading"):
            history[i]["text"] = ai_response
            history[i]["is_loading"] = False
            break

    return (
        history,
        {"display": "none"},
        False,
        False,
        False,
    )


@callback(
    Output("productos-chat-window", "children"),
    Input("productos-chat-history", "data"),
)
def render_chat(history):
    bubbles = []

    if not history:
        welcome_msg = "¡Hola! Soy tu asistente de análisis de productos vendidos. Puedes preguntarme sobre tipos de plato más vendidos, comparaciones con la meta de 1,600 productos, distribución por categorías, tendencias mensuales por tipo, o cualquier otra consulta relacionada con el desempeño de productos por tipo de plato."
        bubbles.append(
            html.Div(
                [
                    html.Div(welcome_msg, className="chat-text"),
                    html.Div("Asistente IA", className="chat-sender text-muted"),
                ],
                className="chat-bubble ai-bubble",
            )
        )
        return bubbles

    for msg in history:
        if msg["role"] == "user":
            bubbles.append(
                html.Div(
                    [
                        html.Div(msg["text"], className="chat-text"),
                        html.Div("Tú", className="chat-sender text-primary"),
                    ],
                    className="chat-bubble user-bubble",
                )
            )
        else:
            text_to_show = msg.get("text", "")
            is_loading = msg.get("is_loading", False)

            if is_loading:
                loading_content = html.Div(
                    [
                        dbc.Spinner(size="sm", color="primary"),
                        html.Span(" Pensando...", className="ms-2 text-muted"),
                    ]
                )

                bubbles.append(
                    html.Div(
                        [
                            html.Div(loading_content, className="chat-text"),
                            html.Div(
                                "Asistente IA", className="chat-sender text-muted"
                            ),
                        ],
                        className="chat-bubble ai-bubble",
                        style={"opacity": 0.7},
                    )
                )
            else:
                bubbles.append(
                    html.Div(
                        [
                            html.Div(text_to_show, className="chat-text"),
                            html.Div(
                                "Asistente IA", className="chat-sender text-muted"
                            ),
                        ],
                        className="chat-bubble ai-bubble",
                    )
                )

    return html.Div(
        bubbles,
        id="productos-chat-messages-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minHeight": "100px",
        },
    )
