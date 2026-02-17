import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.services.data_service import get_crecimiento_ventas_mensual_2024
from app.components.graph_card import create_graph_card
from app.services.ai_service import (
    generar_analisis_crecimiento_ventas,
    generar_respuesta_chat,
)
import time

dash.register_page(__name__, path="/kpi-crecimiento-ventas")

df_all = get_crecimiento_ventas_mensual_2024()
meses_disponibles = [
    {"label": row["nombre_mes"], "value": row["mes"]}
    for _, row in df_all.sort_values("mes").iterrows()
    if pd.notnull(row["crecimiento_ventas"])
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
                                    "Crecimiento de Ventas",
                                    className="page-title mb-2",
                                ),
                                html.P(
                                    "Análisis del crecimiento porcentual mensual comparado con el mes anterior",
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
                                    html.Label(
                                        "Filtrar por mes:", className="form-label"
                                    ),
                                    dcc.Dropdown(
                                        id="crecimiento-mes-filter",
                                        options=[
                                            {"label": "Todos los meses", "value": "all"}
                                        ]
                                        + meses_disponibles,
                                        value="all",
                                        placeholder="Selecciona un mes...",
                                        className="mb-3",
                                    ),
                                    html.Small(
                                        "Nota: El año siempre será 2024 • Meta de crecimiento: 2% mensual",
                                        className="text-muted",
                                    ),
                                ]
                            ),
                            title="Filtros",
                            description="Selecciona meses específicos para analizar",
                            height=200,
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
                        id="crecimiento-ai-analysis-output",
                        className="p-3 border rounded bg-light",
                        style={"whiteSpace": "pre-line"},
                    ),
                    id="crecimiento-analysis-spinner",
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Asistente IA de Negocio", className="section-title mb-3"),
                dcc.Store(id="crecimiento-chat-history", data=[]),
                dcc.Store(id="crecimiento-ai-thinking", data=False),
                html.Div(
                    id="crecimiento-chat-window",
                    className="chat-window mb-3",
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
                html.Div(
                    id="crecimiento-typing-indicator",
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
                            id="crecimiento-chat-input",
                            placeholder="Haz una pregunta sobre el crecimiento de ventas...",
                            type="text",
                            disabled=False,
                        ),
                        dbc.Button(
                            "Enviar",
                            id="crecimiento-chat-send",
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
                    "Crecimiento Porcentual Mensual", className="section-title mb-4"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="crecimiento-bar-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Crecimiento de Ventas (%)",
                                description="Crecimiento porcentual mensual comparado con el mes anterior",
                                height=450,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="crecimiento-heatmap-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Mapa de Calor de Crecimiento",
                                description="Visualización de meses con mayor y menor crecimiento",
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
                html.H4("Análisis Comparativo", className="section-title mb-4"),
                create_graph_card(
                    graph=html.Div(
                        [
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        id="crecimiento-view-toggle",
                                        className="btn-group",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[
                                            {
                                                "label": "Crecimiento vs Meta",
                                                "value": "crecimiento",
                                            },
                                            {
                                                "label": "Ventas Absolutas",
                                                "value": "ventas",
                                            },
                                            {
                                                "label": "Cantidad de Ventas",
                                                "value": "cantidad",
                                            },
                                        ],
                                        value="crecimiento",
                                    ),
                                ],
                                className="radio-group mb-4",
                            ),
                            html.Div(id="crecimiento-dynamic-graph-container"),
                        ]
                    ),
                    title="Análisis Detallado",
                    description="Compara diferentes perspectivas del crecimiento",
                    height=500,
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Resumen de Desempeño", className="section-title mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=html.Div(
                                    [
                                        html.Div(
                                            id="crecimiento-kpi-cards",
                                            className="kpi-cards-grid",
                                        ),
                                    ]
                                ),
                                title="KPIs de Crecimiento",
                                description="Métricas clave del desempeño de crecimiento",
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
                            "Dashboard de Crecimiento de Ventas • Actualizado: Diciembre 2024",
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
        Output("crecimiento-bar-graph", "figure"),
        Output("crecimiento-heatmap-graph", "figure"),
    ],
    [Input("crecimiento-mes-filter", "value")],
)
def update_static_graphs(mes_filtro):
    if mes_filtro == "all" or not mes_filtro:
        df = get_crecimiento_ventas_mensual_2024()
    else:
        df = get_crecimiento_ventas_mensual_2024(mes_filtro)

    df_filtered = df[pd.notnull(df["crecimiento_ventas"])].copy()

    if df_filtered.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_text="No hay datos de crecimiento disponibles",
            annotations=[
                dict(
                    text="No hay datos suficientes",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=16),
                )
            ],
        )
        return empty_fig, empty_fig

    colores = []
    for crecimiento in df_filtered["crecimiento_ventas"]:
        if crecimiento >= 0.02:
            colores.append("#2ecc71")
        elif crecimiento >= 0:
            colores.append("#f39c12")
        else:
            colores.append("#e74c3c")

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=df_filtered["nombre_mes"],
                y=df_filtered["crecimiento_ventas"] * 100,
                marker_color=colores,
                text=df_filtered["crecimiento_ventas_pct"],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Crecimiento: %{text}<extra></extra>",
            ),
            go.Scatter(
                x=df_filtered["nombre_mes"],
                y=[2] * len(df_filtered),
                mode="lines",
                name="Meta (2%)",
                line=dict(color="#3498db", width=2, dash="dash"),
                hovertemplate="Meta: 2%<extra></extra>",
            ),
        ],
        layout=go.Layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=50),
            title_text="Crecimiento Mensual de Ventas (%)",
            title_font=dict(size=18, color="var(--primary-color)"),
            xaxis_title="Mes",
            yaxis_title="Crecimiento (%)",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        ),
    )

    heatmap_data = df_filtered[["nombre_mes", "crecimiento_ventas"]].copy()
    heatmap_data["mes_num"] = range(len(heatmap_data))

    heatmap_fig = px.density_heatmap(
        heatmap_data,
        x="nombre_mes",
        y="crecimiento_ventas",
        z="crecimiento_ventas",
        title="",
        color_continuous_scale="RdBu",
        range_color=[
            df_filtered["crecimiento_ventas"].min(),
            df_filtered["crecimiento_ventas"].max(),
        ],
        labels={"crecimiento_ventas": "Crecimiento", "nombre_mes": "Mes"},
    )

    heatmap_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(l=50, r=50, t=50, b=50),
        title_text="Mapa de Calor: Intensidad de Crecimiento",
        title_font=dict(size=18, color="var(--primary-color)"),
        xaxis_title="Mes",
        yaxis_title="Crecimiento",
        coloraxis_colorbar=dict(title="Intensidad"),
    )

    return bar_fig, heatmap_fig


@callback(
    Output("crecimiento-dynamic-graph-container", "children"),
    [
        Input("crecimiento-view-toggle", "value"),
        Input("crecimiento-mes-filter", "value"),
    ],
)
def update_dynamic_graph(view_type, mes_filtro):
    if mes_filtro == "all" or not mes_filtro:
        df = get_crecimiento_ventas_mensual_2024()
    else:
        df = get_crecimiento_ventas_mensual_2024(mes_filtro)

    df_filtered = df[pd.notnull(df["crecimiento_ventas"])].copy()

    if df_filtered.empty:
        return html.Div(
            "No hay datos suficientes para mostrar análisis comparativo",
            className="text-center text-muted py-5",
        )

    if view_type == "crecimiento":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df_filtered["nombre_mes"],
                    y=df_filtered["crecimiento_ventas"] * 100,
                    mode="lines+markers+text",
                    name="Crecimiento Real",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=10, color="#3498db"),
                    text=df_filtered["crecimiento_ventas_pct"],
                    textposition="top center",
                ),
                go.Scatter(
                    x=df_filtered["nombre_mes"],
                    y=[2] * len(df_filtered),
                    mode="lines",
                    name="Meta (2%)",
                    line=dict(color="#2ecc71", width=2, dash="dash"),
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Tendencia de Crecimiento vs Meta",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Crecimiento (%)",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
            ),
        )

    elif view_type == "ventas":
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_filtered["nombre_mes"],
                    y=df_filtered["ventas_mes"],
                    name="Ventas del Mes",
                    marker_color="#3498db",
                ),
                go.Scatter(
                    x=df_filtered["nombre_mes"],
                    y=df_filtered["ventas_mes_anterior"],
                    mode="lines+markers",
                    name="Ventas Mes Anterior",
                    line=dict(color="#e74c3c", width=2, dash="dot"),
                    marker=dict(size=8, color="#e74c3c"),
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Ventas Mensuales vs Mes Anterior",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Ventas ($)",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                yaxis=dict(tickformat="$,.0f"),
            ),
        )

    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_filtered["nombre_mes"],
                    y=df_filtered["cantidad_ventas_mes"],
                    name="Cantidad del Mes",
                    marker_color="#9b59b6",
                ),
                go.Scatter(
                    x=df_filtered["nombre_mes"],
                    y=df_filtered["cantidad_ventas_mes_anterior"],
                    mode="lines+markers",
                    name="Cantidad Mes Anterior",
                    line=dict(color="#34495e", width=2, dash="dot"),
                    marker=dict(size=8, color="#34495e"),
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Cantidad de Ventas vs Mes Anterior",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Cantidad de Ventas",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                yaxis=dict(tickformat=","),
            ),
        )

    return dcc.Graph(
        id="crecimiento-line-graph",
        figure=fig,
        config={"displayModeBar": True},
        className="dynamic-graph",
    )


@callback(
    Output("crecimiento-kpi-cards", "children"),
    [Input("crecimiento-mes-filter", "value")],
)
def update_kpi_cards(mes_filtro):
    if mes_filtro == "all" or not mes_filtro:
        df = get_crecimiento_ventas_mensual_2024()
    else:
        df = get_crecimiento_ventas_mensual_2024(mes_filtro)

    df_filtered = df[pd.notnull(df["crecimiento_ventas"])].copy()

    if df_filtered.empty:
        return html.Div(
            "No hay datos disponibles para mostrar KPIs",
            className="text-center text-muted py-3",
        )

    crecimiento_promedio = df_filtered["crecimiento_ventas"].mean() * 100
    meses_sobre_meta = df_filtered["cumple_meta"].sum()
    total_meses = len(df_filtered)
    mejor_crecimiento = df_filtered["crecimiento_ventas"].max() * 100
    peor_crecimiento = df_filtered["crecimiento_ventas"].min() * 100
    crecimiento_actual = (
        df_filtered["crecimiento_ventas"].iloc[-1] * 100 if len(df_filtered) > 0 else 0
    )

    mejor_mes = (
        df_filtered.loc[df_filtered["crecimiento_ventas"].idxmax(), "nombre_mes"]
        if not df_filtered.empty
        else "N/A"
    )
    peor_mes = (
        df_filtered.loc[df_filtered["crecimiento_ventas"].idxmin(), "nombre_mes"]
        if not df_filtered.empty
        else "N/A"
    )

    cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(f"{crecimiento_promedio:.2f}%", className="card-title"),
                        html.P(
                            "Crecimiento Promedio", className="card-text text-muted"
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
                        html.H5(f"{crecimiento_actual:.2f}%", className="card-title"),
                        html.P("Crecimiento Actual", className="card-text text-muted"),
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
                        html.H5(f"{mejor_crecimiento:.2f}%", className="card-title"),
                        html.P(f"Mejor: {mejor_mes}", className="card-text text-muted"),
                    ]
                ),
                className="text-center h-100 border-success",
            ),
            md=3,
            sm=6,
            className="mb-3",
        ),
    ]

    return dbc.Row(cards)


@callback(
    Output("crecimiento-ai-analysis-output", "children"),
    Output("crecimiento-analysis-spinner", "type"),
    Input("crecimiento-mes-filter", "value"),
)
def update_ai_analysis(mes_filtro):
    if mes_filtro == "all" or not mes_filtro:
        df = get_crecimiento_ventas_mensual_2024()
    else:
        df = get_crecimiento_ventas_mensual_2024(mes_filtro)

    df_filtered = df[pd.notnull(df["crecimiento_ventas"])].copy()

    if df_filtered.empty:
        return "No hay datos para generar análisis.", "border"

    meta_crecimiento = 0.02
    analisis = generar_analisis_crecimiento_ventas(df_filtered, meta_crecimiento)

    return analisis, "border"


@callback(
    Output("crecimiento-chat-history", "data", allow_duplicate=True),
    Output("crecimiento-chat-input", "value"),
    Output("crecimiento-typing-indicator", "style"),
    Output("crecimiento-ai-thinking", "data"),
    Output("crecimiento-chat-input", "disabled"),
    Output("crecimiento-chat-send", "disabled"),
    Input("crecimiento-chat-send", "n_clicks"),
    State("crecimiento-chat-input", "value"),
    State("crecimiento-chat-history", "data"),
    State("crecimiento-ai-thinking", "data"),
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
    Output("crecimiento-chat-history", "data"),
    Output("crecimiento-typing-indicator", "style", allow_duplicate=True),
    Output("crecimiento-ai-thinking", "data", allow_duplicate=True),
    Output("crecimiento-chat-input", "disabled", allow_duplicate=True),
    Output("crecimiento-chat-send", "disabled", allow_duplicate=True),
    Input("crecimiento-ai-thinking", "data"),
    State("crecimiento-chat-history", "data"),
    State("crecimiento-mes-filter", "value"),
    prevent_initial_call=True,
)
def generate_ai_response(is_thinking, history, mes_filtro):
    if not is_thinking or not history:
        return dash.no_update, {"display": "none"}, False, False, False

    user_messages = [msg for msg in history if msg.get("role") == "user"]
    if not user_messages:
        return dash.no_update, {"display": "none"}, False, False, False

    last_user_msg = user_messages[-1]["text"]

    if mes_filtro == "all" or not mes_filtro:
        df = get_crecimiento_ventas_mensual_2024()
    else:
        df = get_crecimiento_ventas_mensual_2024(mes_filtro)

    df_filtered = df[pd.notnull(df["crecimiento_ventas"])].copy()

    if df_filtered.empty:
        ai_response = "No hay datos disponibles para responder tu pregunta."
    else:
        contexto = f"""
        DATOS DE CRECIMIENTO DE VENTAS:
        
        Crecimiento promedio: {df_filtered['crecimiento_ventas'].mean()*100:.2f}%
        Meta de crecimiento: 2% mensual
        Meses que superaron la meta: {df_filtered['cumple_meta'].sum()} de {len(df_filtered)}
        Mejor mes de crecimiento: {df_filtered.loc[df_filtered['crecimiento_ventas'].idxmax()]['nombre_mes']} ({df_filtered['crecimiento_ventas'].max()*100:.2f}%)
        Peor mes de crecimiento: {df_filtered.loc[df_filtered['crecimiento_ventas'].idxmin()]['nombre_mes']} ({df_filtered['crecimiento_ventas'].min()*100:.2f}%)
        Crecimiento del último mes: {df_filtered['crecimiento_ventas'].iloc[-1]*100:.2f}% si hay datos suficientes
        Variabilidad del crecimiento: {df_filtered['crecimiento_ventas'].std()*100:.2f}%
        Promedio de ventas mensuales: ${df_filtered['ventas_mes'].mean():,.0f}
        Promedio de cantidad de ventas mensuales: {df_filtered['cantidad_ventas_mes'].mean():,.0f}
        """

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
    Output("crecimiento-chat-window", "children"),
    Input("crecimiento-chat-history", "data"),
)
def render_chat(history):
    bubbles = []

    if not history:
        welcome_msg = "¡Hola! Soy tu asistente de análisis de crecimiento de ventas. Puedes preguntarme sobre tendencias de crecimiento, comparaciones con la meta del 2%, análisis de meses con mejor/peor desempeño, recomendaciones para mejorar el crecimiento, o cualquier otra consulta relacionada con la evolución mensual de las ventas."
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
        id="crecimiento-chat-messages-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minHeight": "100px",
        },
    )
