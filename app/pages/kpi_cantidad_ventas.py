import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from app.services.data_service import get_cantidad_ventas_mensuales_2024
from app.components.graph_card import create_graph_card
from app.services.ai_service import (
    generar_analisis_cantidad_ventas,
    generar_respuesta_chat,
)
from app.services.ml_service import generar_contexto_ml_cantidad
import time

dash.register_page(__name__, path="/kpi-cantidad-ventas")

df_all = get_cantidad_ventas_mensuales_2024()
meses_disponibles = [
    {"label": row["nombre_mes"], "value": row["mes"]}
    for _, row in df_all.sort_values("mes").iterrows()
]

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            html.H1(
                                "Cantidad Total de Ventas",
                                className="page-title mb-2",
                            ),
                            html.P(
                                "Análisis del volumen de ventas mensuales",
                                className="page-subtitle text-muted",
                            ),
                        ],
                        className="page-header",
                    )
                )
            ],
            className="mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    create_graph_card(
                        graph=html.Div(
                            [
                                html.Label("Filtrar por mes:", className="form-label"),
                                dcc.Dropdown(
                                    id="cantidad-mes-filter",
                                    options=[
                                        {"label": "Todos los meses", "value": "all"}
                                    ]
                                    + meses_disponibles,
                                    value="all",
                                    placeholder="Selecciona un mes...",
                                    className="mb-3",
                                    style={"position": "relative", "zIndex": 1000},
                                ),
                                html.Small(
                                    "Nota: El año siempre será 2024 • Meta mensual: 2,400 ventas",
                                    className="text-muted",
                                ),
                            ],
                            style={"position": "relative", "zIndex": 100},
                        ),
                        title="Filtros",
                        description="Selecciona meses específicos para analizar",
                        height=200,
                    ),
                    lg=6,
                    md=12,
                    className="mb-4",
                )
            ]
        ),
        html.Div(
            [
                html.H4("Análisis Inteligente con IA", className="section-title mb-3"),
                dbc.Spinner(
                    html.Div(
                        id="cantidad-ai-analysis-output",
                        className="p-3 border rounded bg-light",
                        style={"whiteSpace": "pre-line"},
                    ),
                    id="cantidad-analysis-spinner",
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Asistente IA de Negocio", className="section-title mb-3"),
                dcc.Store(id="cantidad-chat-history", data=[]),
                dcc.Store(id="cantidad-ai-thinking", data=False),
                html.Div(
                    id="cantidad-chat-window",
                    className="chat-window mb-3",
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
                html.Div(
                    id="cantidad-typing-indicator",
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
                            id="cantidad-chat-input",
                            placeholder="Haz una pregunta sobre la cantidad de ventas...",
                            type="text",
                            disabled=False,
                        ),
                        dbc.Button(
                            "Enviar",
                            id="cantidad-chat-send",
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
                html.H4("Distribución de Ventas", className="section-title mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="cantidad-bar-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Cantidad de Ventas Mensuales",
                                description="Número de ventas por mes",
                                height=400,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="cantidad-pie-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Distribución de Ventas",
                                description="Participación porcentual por mes",
                                height=400,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                    ]
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
                                dbc.RadioItems(
                                    id="cantidad-view-toggle",
                                    className="btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                    labelCheckedClassName="active",
                                    options=[
                                        {
                                            "label": "Ventas Mensuales",
                                            "value": "mensual",
                                        },
                                        {
                                            "label": "Ventas Acumuladas",
                                            "value": "acumulado",
                                        },
                                    ],
                                    value="mensual",
                                ),
                                className="radio-group mb-4",
                            ),
                            html.Div(id="cantidad-dynamic-graph-container"),
                        ]
                    ),
                    title="Tendencia de Ventas",
                    description="Comparación contra la meta establecida",
                    height=450,
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.Hr(className="my-4"),
                html.Div(
                    [
                        html.Small(
                            "Dashboard de Cantidad de Ventas • Actualizado: Diciembre 2024",
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
        Output("cantidad-bar-graph", "figure"),
        Output("cantidad-pie-graph", "figure"),
    ],
    Input("cantidad-mes-filter", "value"),
)
def update_static_graphs(mes_filtro):
    """Actualiza los gráficos estáticos según el filtro de mes."""

    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_cantidad_ventas_mensuales_2024(mes_filtro_list)

    bar_fig = px.bar(
        df,
        x="nombre_mes",
        y="cantidad_ventas",
        title="",
        labels={
            "cantidad_ventas": "Cantidad de Ventas",
            "nombre_mes": "Mes",
        },
        color_discrete_sequence=["#3498db"],  # Color consistente
    )

    if mes_filtro_list and len(mes_filtro_list) == 1:
        mes_nombre = df.iloc[0]["nombre_mes"] if len(df) > 0 else ""
        title_text = f"Cantidad de Ventas - {mes_nombre} 2024"
    elif mes_filtro_list and len(mes_filtro_list) > 1:
        title_text = f"Cantidad de Ventas - {len(mes_filtro_list)} meses seleccionados"
    else:
        title_text = "Cantidad de Ventas Mensuales 2024"

    bar_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=30, b=50),
        title_text=title_text,
        title_font=dict(size=16, color="var(--primary-color)"),
        xaxis_title="Mes",
        yaxis_title="Cantidad de Ventas",
    )
    bar_fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Ventas: %{y:,.0f}<extra></extra>",
    )

    if len(df) > 0:
        pie_fig = px.pie(
            df,
            names="nombre_mes",
            values="cantidad_ventas",
            title="",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )

        pie_title = "Distribución por Mes"
        if mes_filtro_list and len(mes_filtro_list) == 1:
            pie_title = f"Ventas - {df.iloc[0]['nombre_mes']} 2024"

        pie_fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            font=dict(size=12),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(
                orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.2
            ),
            title_text=pie_title,
            title_font=dict(size=16, color="var(--primary-color)"),
        )
        pie_fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Ventas: %{value:,.0f}<extra></extra>",
        )
    else:
        pie_fig = go.Figure()
        pie_fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_text="No hay datos para el filtro seleccionado",
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

    return bar_fig, pie_fig


@callback(
    Output("cantidad-dynamic-graph-container", "children"),
    [
        Input("cantidad-view-toggle", "value"),
        Input("cantidad-mes-filter", "value"),
    ],
)
def update_dynamic_graph(view_type, mes_filtro):
    """Actualiza el gráfico dinámico según vista seleccionada."""

    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_cantidad_ventas_mensuales_2024(mes_filtro_list)

    if len(df) == 0:
        return html.Div(
            "No hay datos para el filtro seleccionado",
            className="text-center text-muted py-5",
        )

    if view_type == "mensual":
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["cantidad_ventas"],
                    mode="lines+markers",
                    name="Ventas Mensuales",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=8, color="#3498db"),
                ),
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["meta_mensual"],
                    mode="lines",
                    name="Meta Mensual",
                    line=dict(color="#e74c3c", width=2, dash="dash"),
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=30, b=50),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                title_text="Ventas Mensuales vs Meta 2024",
                title_font=dict(size=16, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Cantidad de Ventas",
                showlegend=True,
            ),
        )
    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["cantidad_acumulada"],
                    mode="lines+markers",
                    name="Ventas Acumuladas",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=8, color="#3498db"),
                ),
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["meta_acumulada"],
                    mode="lines",
                    name="Meta Acumulada",
                    line=dict(color="#e74c3c", width=2, dash="dash"),
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=30, b=50),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                title_text="Ventas Acumuladas vs Meta 2024",
                title_font=dict(size=16, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Ventas Acumuladas",
                showlegend=True,
            ),
        )

    return dcc.Graph(
        id="cantidad-line-graph",
        figure=fig,
        config={"displayModeBar": True},
        className="dynamic-graph",
    )


@callback(
    Output("cantidad-ai-analysis-output", "children"),
    Output("cantidad-analysis-spinner", "type"),
    Input("cantidad-mes-filter", "value"),
)
def update_ai_analysis(mes_filtro):
    """Genera análisis de IA para el resumen principal con indicador de carga."""

    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_cantidad_ventas_mensuales_2024(mes_filtro_list)

    if df.empty:
        return "No hay datos para generar análisis.", "border"

    meta_mensual = 2400

    analisis = generar_analisis_cantidad_ventas(df, meta_mensual)

    return analisis, "border"


@callback(
    Output("cantidad-chat-history", "data", allow_duplicate=True),
    Output("cantidad-chat-input", "value"),
    Output("cantidad-typing-indicator", "style"),
    Output("cantidad-ai-thinking", "data"),
    Output("cantidad-chat-input", "disabled"),
    Output("cantidad-chat-send", "disabled"),
    Input("cantidad-chat-send", "n_clicks"),
    State("cantidad-chat-input", "value"),
    State("cantidad-chat-history", "data"),
    State("cantidad-ai-thinking", "data"),
    prevent_initial_call=True,
)
def handle_user_message(n_clicks, user_msg, history, is_thinking):
    """Maneja el mensaje del usuario - se muestra inmediatamente."""

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
    Output("cantidad-chat-history", "data"),
    Output("cantidad-typing-indicator", "style", allow_duplicate=True),
    Output("cantidad-ai-thinking", "data", allow_duplicate=True),
    Output("cantidad-chat-input", "disabled", allow_duplicate=True),
    Output("cantidad-chat-send", "disabled", allow_duplicate=True),
    Input("cantidad-ai-thinking", "data"),
    State("cantidad-chat-history", "data"),
    State("cantidad-mes-filter", "value"),
    prevent_initial_call=True,
)
def generate_ai_response(is_thinking, history, mes_filtro):
    """Genera la respuesta de IA cuando está en modo 'pensando'."""

    if not is_thinking or not history:
        return dash.no_update, {"display": "none"}, False, False, False

    user_messages = [msg for msg in history if msg.get("role") == "user"]
    if not user_messages:
        return dash.no_update, {"display": "none"}, False, False, False

    last_user_msg = user_messages[-1]["text"]

    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_cantidad_ventas_mensuales_2024(mes_filtro_list)

    if df.empty:
        ai_response = "No hay datos disponibles para responder tu pregunta."
    else:
        meta_cantidad = 2400
        contexto = generar_contexto_ml_cantidad(df, meta_cantidad)
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
    Output("cantidad-chat-window", "children"),
    Input("cantidad-chat-history", "data"),
)
def render_chat(history):
    """Renderiza el chat."""

    bubbles = []

    if not history:
        welcome_msg = "¡Hola! Soy tu asistente de análisis de cantidad de ventas. Puedes preguntarme sobre volúmenes de transacciones, comparaciones con la meta (2,400 ventas/mes), patrones estacionales, recomendaciones para aumentar el tráfico de clientes, o cualquier otra consulta relacionada con el volumen de ventas."
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
        id="cantidad-chat-messages-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minHeight": "100px",
        },
    )
