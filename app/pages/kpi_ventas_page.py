import dash
from dash import html, dcc, Input, Output, callback, State, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from app.services.data_service import get_ventas_mensuales_2024
from app.components.graph_card import create_graph_card
from app.services.ai_service import generar_analisis_ia, generar_respuesta_chat
from app.services.ml_service import generar_recomendaciones_ml
import time

dash.register_page(__name__, path="/kpi-ventas")

df_all = get_ventas_mensuales_2024()
meses_disponibles = [
    {"label": row["nombre_mes"], "value": row["mes"]}
    for _, row in df_all.sort_values("mes").iterrows()
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
                                    "Análisis de Ventas", className="page-title mb-2"
                                ),
                                html.P(
                                    "Dashboard completo con KPIs, tendencias y métricas de ventas mensuales",
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
                                        id="mes-filter",
                                        options=[
                                            {"label": "Todos los meses", "value": "all"}
                                        ]
                                        + meses_disponibles,  # type: ignore
                                        value="all",
                                        placeholder="Selecciona un mes...",
                                        className="mb-3",
                                    ),
                                    html.Small(
                                        "Nota: El año siempre será 2024",
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
                        id="ai-analysis-output",
                        className="p-3 border rounded bg-light",
                        style={"whiteSpace": "pre-line"},
                    ),
                    id="analysis-spinner",
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Asistente IA de Negocio", className="section-title mb-3"),
                dcc.Store(id="chat-history", data=[]),
                dcc.Store(
                    id="ai-thinking", data=False
                ),  # Para controlar si la IA está pensando
                html.Div(
                    id="chat-window",
                    className="chat-window mb-3",
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
                html.Div(
                    id="typing-indicator",
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
                            id="chat-input",
                            placeholder="Haz una pregunta sobre las ventas...",
                            type="text",
                            disabled=False,
                        ),
                        dbc.Button(
                            "Enviar", id="chat-send", color="primary", disabled=False
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
                                    id="bar-graph", config={"displayModeBar": True}
                                ),
                                title="Ventas Mensuales 2024",
                                description="Distribución de ventas por mes del año actual",
                                height=400,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="pie-graph", config={"displayModeBar": True}
                                ),
                                title="Distribución de Ventas por Mes",
                                description="Porcentaje de ventas de cada mes respecto al total filtrado",
                                height=400,
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
                                        id="view-toggle",
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
                                ],
                                className="radio-group mb-4",
                            ),
                            html.Div(id="dynamic-graph-container"),
                        ]
                    ),
                    title="Análisis de Tendencias",
                    description="Intercambia entre vista mensual y acumulada para analizar tendencias",
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
                            "Dashboard de Ventas • Actualizado: Diciembre 2024",
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


# ============================================================================
# CALLBACKS PARA GRÁFICOS (MANTÉN LOS QUE YA TIENES)
# ============================================================================
@callback(
    [
        Output("bar-graph", "figure"),
        Output("pie-graph", "figure"),
    ],
    [Input("mes-filter", "value")],
)
def update_static_graphs(mes_filtro):
    """Actualiza los gráficos estáticos según el filtro de mes."""

    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_ventas_mensuales_2024(mes_filtro_list)

    bar_fig = px.bar(
        df,
        x="nombre_mes",
        y="ventas_mes",
        title="",
        labels={"ventas_mes": "Ventas ($)", "nombre_mes": "Mes"},
        color_discrete_sequence=["#3498db"],
    )

    if mes_filtro_list and len(mes_filtro_list) == 1:
        mes_nombre = df.iloc[0]["nombre_mes"] if len(df) > 0 else ""
        title_text = f"Ventas - {mes_nombre} 2024"
    elif mes_filtro_list and len(mes_filtro_list) > 1:
        title_text = f"Ventas - {len(mes_filtro_list)} meses seleccionados - 2024"
    else:
        title_text = "Ventas Mensuales 2024"

    bar_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=30, b=50),
        title_text=title_text,
        title_font=dict(size=16, color="var(--primary-color)"),
        xaxis_title="Mes",
        yaxis_title="Ventas ($)",
    )
    bar_fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Ventas: $%{y:,.0f}<extra></extra>",
    )

    if len(df) > 0:
        pie_fig = px.pie(
            df,
            names="nombre_mes",
            values="ventas_mes",
            title="",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )

        pie_title = "Distribución de Ventas"
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
            hovertemplate="<b>%{label}</b><br>Ventas: $%{value:,.0f}<br>%{percent}<extra></extra>",
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
    Output("dynamic-graph-container", "children"),
    [Input("view-toggle", "value"), Input("mes-filter", "value")],
)
def update_dynamic_graph(view_type, mes_filtro):
    """Actualiza el gráfico dinámico según la selección del usuario y filtro."""

    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_ventas_mensuales_2024(mes_filtro_list)

    if len(df) == 0:
        return html.Div(
            "No hay datos para el filtro seleccionado",
            className="text-center text-muted py-5",
        )

    if view_type == "mensual":
        promedio = df["ventas_mes"].mean() if len(df) > 0 else 0

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["ventas_mes"],
                    mode="lines+markers",
                    name="Ventas Mensuales",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=8, color="#3498db"),
                ),
                go.Scatter(
                    x=df["nombre_mes"],
                    y=[promedio] * len(df),
                    mode="lines",
                    name=f"Promedio: ${promedio:,.0f}",
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
                title_text="Ventas Mensuales vs Promedio 2024",
                title_font=dict(size=16, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Ventas ($)",
                showlegend=True,
            ),
        )

    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["ventas_acumuladas"],
                    mode="lines+markers",
                    name="Ventas Acumuladas",
                    line=dict(color="#2ecc71", width=3),
                    marker=dict(size=8, color="#2ecc71"),
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
                yaxis_title="Ventas Acumuladas ($)",
                showlegend=True,
            ),
        )

    return dcc.Graph(id="line-graph", figure=fig, config={"displayModeBar": True})


@callback(
    Output("ai-analysis-output", "children"),
    Output("analysis-spinner", "type"),
    Input("mes-filter", "value"),
)
def update_ai_analysis(mes_filtro):
    """Genera análisis de IA para el resumen principal con indicador de carga."""

    placeholder = html.Div(
        [
            html.P("Analizando datos de ventas...", className="text-muted"),
        ]
    )

    # Luego obtiene los datos
    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_ventas_mensuales_2024(mes_filtro_list)

    if df.empty:
        return "No hay datos para generar análisis.", "border"

    meta_mensual = df["ventas_mes"].mean() * 1.10

    analisis = generar_analisis_ia(df, meta_mensual)

    return analisis, "border"


@callback(
    Output("chat-history", "data", allow_duplicate=True),
    Output("chat-input", "value"),
    Output("typing-indicator", "style"),
    Output("ai-thinking", "data"),
    Output("chat-input", "disabled"),
    Output("chat-send", "disabled"),
    Input("chat-send", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    State("ai-thinking", "data"),
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
    Output("chat-history", "data"),
    Output("typing-indicator", "style", allow_duplicate=True),
    Output("ai-thinking", "data", allow_duplicate=True),
    Output("chat-input", "disabled", allow_duplicate=True),
    Output("chat-send", "disabled", allow_duplicate=True),
    Input("ai-thinking", "data"),
    State("chat-history", "data"),
    State("mes-filter", "value"),
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

    df = get_ventas_mensuales_2024(mes_filtro_list)

    if df.empty:
        ai_response = "No hay datos disponibles para responder tu pregunta."
    else:
        meta_mensual = df["ventas_mes"].mean() * 1.10
        resultado_ml = generar_recomendaciones_ml(df, meta_mensual)
        contexto = resultado_ml["contexto_ia"]

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
    Output("chat-window", "children"),
    Input("chat-history", "data"),
)
def render_chat(history):
    """Renderiza el chat."""

    bubbles = []

    if not history:
        welcome_msg = "¡Hola! Soy tu asistente de análisis de ventas. Puedes preguntarme sobre tendencias, comparaciones entre meses, recomendaciones para mejorar ventas, o cualquier otra consulta relacionada con los datos."
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
        id="chat-messages-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minHeight": "100px",
        },
    )
