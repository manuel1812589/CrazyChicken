import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from app.services.data_service import get_ticket_promedio_global_2024
from app.components.graph_card import create_graph_card
from app.services.ai_service import (
    generar_analisis_ticket_promedio,
    generar_respuesta_chat,
)
import time

dash.register_page(__name__, path="/kpi-ticket-promedio")

df_all = get_ticket_promedio_global_2024()
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
                                    "Ticket Promedio Global",
                                    className="page-title mb-2",
                                ),
                                html.P(
                                    "Análisis del ticket promedio por venta y comparación con la meta mensual",
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
                                        id="ticket-mes-filter",  # CAMBIADO
                                        options=[
                                            {"label": "Todos los meses", "value": "all"}
                                        ]
                                        + meses_disponibles,  # type: ignore
                                        value="all",
                                        placeholder="Selecciona un mes...",
                                        className="mb-3",
                                    ),
                                    html.Small(
                                        "Nota: El año siempre será 2024 • Meta: $70 por ticket",
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
        # Sección de Análisis IA
        html.Div(
            [
                html.H4("Análisis Inteligente con IA", className="section-title mb-3"),
                dbc.Spinner(
                    html.Div(
                        id="ticket-ai-analysis-output",  # CAMBIADO
                        className="p-3 border rounded bg-light",
                        style={"whiteSpace": "pre-line"},
                    ),
                    id="ticket-analysis-spinner",  # CAMBIADO
                ),
            ],
            className="mb-4",
        ),
        # Sección de Chat Asistente
        html.Div(
            [
                html.H4("Asistente IA de Negocio", className="section-title mb-3"),
                dcc.Store(id="ticket-chat-history", data=[]),  # CAMBIADO
                dcc.Store(id="ticket-ai-thinking", data=False),  # CAMBIADO
                html.Div(
                    id="ticket-chat-window",  # CAMBIADO
                    className="chat-window mb-3",
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
                html.Div(
                    id="ticket-typing-indicator",  # CAMBIADO
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
                            id="ticket-chat-input",  # CAMBIADO
                            placeholder="Haz una pregunta sobre el ticket promedio...",
                            type="text",
                            disabled=False,
                        ),
                        dbc.Button(
                            "Enviar",
                            id="ticket-chat-send",
                            color="primary",
                            disabled=False,  # CAMBIADO
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
                    "Distribución de Ticket Promedio", className="section-title mb-4"
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="bar-ticket-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Ticket Promedio por Mes",
                                description="Valor promedio por transacción en cada mes",
                                height=400,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="pie-ticket-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Distribución por Mes",
                                description="Porcentaje que representa cada mes en el total",
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
                                        id="ticket-view-toggle",  # CAMBIADO
                                        className="btn-group",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[
                                            {
                                                "label": "Ticket Promedio",
                                                "value": "ticket",
                                            },
                                            {
                                                "label": "Comparación con Meta",
                                                "value": "comparacion",
                                            },
                                        ],
                                        value="ticket",
                                    ),
                                ],
                                className="radio-group mb-4",
                            ),
                            html.Div(id="dynamic-ticket-graph-container"),
                        ]
                    ),
                    title="Análisis de Tendencias",
                    description="Intercambia entre vista de ticket promedio y comparación con meta",
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
                            "Dashboard de Ticket Promedio • Actualizado: Diciembre 2024",
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
        Output("bar-ticket-graph", "figure"),
        Output("pie-ticket-graph", "figure"),
    ],
    [Input("ticket-mes-filter", "value")],  # CAMBIADO
)
def update_static_graphs(mes_filtro):
    """Actualiza los gráficos estáticos según el filtro de mes."""

    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_ticket_promedio_global_2024(mes_filtro_list)

    bar_fig = px.bar(
        df,
        x="nombre_mes",
        y="ticket_promedio",
        title="",
        labels={"ticket_promedio": "Ticket Promedio ($)", "nombre_mes": "Mes"},
        color_discrete_sequence=["#3498db"],
    )

    if mes_filtro_list and len(mes_filtro_list) == 1:
        mes_nombre = df.iloc[0]["nombre_mes"] if len(df) > 0 else ""
        title_text = f"Ticket Promedio - {mes_nombre} 2024"
    elif mes_filtro_list and len(mes_filtro_list) > 1:
        title_text = f"Ticket Promedio - {len(mes_filtro_list)} meses seleccionados"
    else:
        title_text = "Ticket Promedio por Mes 2024"

    bar_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=12),
        margin=dict(l=50, r=50, t=30, b=50),
        title_text=title_text,
        title_font=dict(size=16, color="var(--primary-color)"),
        xaxis_title="Mes",
        yaxis_title="Ticket Promedio ($)",
    )
    bar_fig.update_traces(
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Ticket: $%{y:,.2f}<extra></extra>",
    )

    if len(df) > 0:
        pie_fig = px.pie(
            df,
            names="nombre_mes",
            values="ticket_promedio",
            title="",
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )

        pie_title = "Distribución por Mes"
        if mes_filtro_list and len(mes_filtro_list) == 1:
            pie_title = f"Ticket - {df.iloc[0]['nombre_mes']} 2024"

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
            hovertemplate="<b>%{label}</b><br>Ticket: $%{value:,.2f}<extra></extra>",
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
    Output("dynamic-ticket-graph-container", "children"),
    [
        Input("ticket-view-toggle", "value"),
        Input("ticket-mes-filter", "value"),
    ],  # CAMBIADO
)
def update_dynamic_graph(view_type, mes_filtro):
    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_ticket_promedio_global_2024(mes_filtro_list)

    if len(df) == 0:
        return html.Div(
            "No hay datos para el filtro seleccionado",
            className="text-center text-muted py-5",
        )

    if view_type == "ticket":
        promedio = df["ticket_promedio"].mean() if len(df) > 0 else 0

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["ticket_promedio"],
                    mode="lines+markers",
                    name="Ticket Promedio",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=8, color="#3498db"),
                ),
                go.Scatter(
                    x=df["nombre_mes"],
                    y=[promedio] * len(df),
                    mode="lines",
                    name=f"Promedio: ${promedio:,.2f}",
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
                title_text="Ticket Promedio Mensual 2024",
                title_font=dict(size=16, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Ticket Promedio ($)",
                showlegend=True,
            ),
        )

    else:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["ticket_promedio"],
                    mode="lines+markers",
                    name="Ticket Promedio",
                    line=dict(color="#3498db", width=3),
                    marker=dict(size=8, color="#3498db"),
                ),
                go.Scatter(
                    x=df["nombre_mes"],
                    y=df["meta_ticket"],
                    mode="lines",
                    name="Meta ($70)",
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
                title_text="Ticket Promedio vs Meta 2024",
                title_font=dict(size=16, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Ticket Promedio ($)",
                showlegend=True,
            ),
        )

    return dcc.Graph(
        id="ticket-line-graph", figure=fig, config={"displayModeBar": True}
    )  # CAMBIADO


# Callback para el análisis de IA
@callback(
    Output("ticket-ai-analysis-output", "children"),  # CAMBIADO
    Output("ticket-analysis-spinner", "type"),  # CAMBIADO
    Input("ticket-mes-filter", "value"),  # CAMBIADO
)
def update_ai_analysis(mes_filtro):
    """Genera análisis de IA para el resumen principal con indicador de carga."""

    # Obtiene los datos
    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_ticket_promedio_global_2024(mes_filtro_list)

    if df.empty:
        return "No hay datos para generar análisis.", "border"

    # Meta específica para ticket promedio
    meta_ticket = 70

    # Generar análisis específico para ticket promedio
    analisis = generar_analisis_ticket_promedio(df, meta_ticket)

    return analisis, "border"


# Callback para manejar mensajes del usuario en el chat
@callback(
    Output("ticket-chat-history", "data", allow_duplicate=True),  # CAMBIADO
    Output("ticket-chat-input", "value"),  # CAMBIADO
    Output("ticket-typing-indicator", "style"),  # CAMBIADO
    Output("ticket-ai-thinking", "data"),  # CAMBIADO
    Output("ticket-chat-input", "disabled"),  # CAMBIADO
    Output("ticket-chat-send", "disabled"),  # CAMBIADO
    Input("ticket-chat-send", "n_clicks"),  # CAMBIADO
    State("ticket-chat-input", "value"),  # CAMBIADO
    State("ticket-chat-history", "data"),  # CAMBIADO
    State("ticket-ai-thinking", "data"),  # CAMBIADO
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


# Callback para generar respuesta de IA
@callback(
    Output("ticket-chat-history", "data"),  # CAMBIADO
    Output("ticket-typing-indicator", "style", allow_duplicate=True),  # CAMBIADO
    Output("ticket-ai-thinking", "data", allow_duplicate=True),  # CAMBIADO
    Output("ticket-chat-input", "disabled", allow_duplicate=True),  # CAMBIADO
    Output("ticket-chat-send", "disabled", allow_duplicate=True),  # CAMBIADO
    Input("ticket-ai-thinking", "data"),  # CAMBIADO
    State("ticket-chat-history", "data"),  # CAMBIADO
    State("ticket-mes-filter", "value"),  # CAMBIADO
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

    # Obtener datos según filtro
    if mes_filtro == "all" or not mes_filtro:
        mes_filtro_list = None
    elif isinstance(mes_filtro, list):
        mes_filtro_list = mes_filtro
    else:
        mes_filtro_list = [mes_filtro]

    df = get_ticket_promedio_global_2024(mes_filtro_list)

    if df.empty:
        ai_response = "No hay datos disponibles para responder tu pregunta."
    else:
        # Crear contexto específico para ticket promedio
        contexto = f"""
        DATOS DE TICKET PROMEDIO:
        
        Ticket promedio general: ${df["ticket_promedio"].mean():,.2f}
        Mejor mes (mayor ticket): {df.loc[df["ticket_promedio"].idxmax()]["nombre_mes"]} (${df["ticket_promedio"].max():,.2f})
        Peor mes (menor ticket): {df.loc[df["ticket_promedio"].idxmin()]["nombre_mes"]} (${df["ticket_promedio"].min():,.2f})
        Cantidad de ventas totales: {df["cantidad_ventas"].sum():,.0f}
        Ventas totales: ${df["ventas_totales"].sum():,.0f}
        Meta de ticket: $70.00
        Meses que superan la meta: {(df["ticket_promedio"] >= 70).sum()}
        Meses por debajo de la meta: {(df["ticket_promedio"] < 70).sum()}
        Desviación estándar: ${df["ticket_promedio"].std():,.2f}
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


# Callback para renderizar el chat
@callback(
    Output("ticket-chat-window", "children"),  # CAMBIADO
    Input("ticket-chat-history", "data"),  # CAMBIADO
)
def render_chat(history):
    """Renderiza el chat."""

    bubbles = []

    if not history:
        welcome_msg = "¡Hola! Soy tu asistente de análisis de ticket promedio. Puedes preguntarme sobre tendencias, comparaciones con la meta ($70), recomendaciones para mejorar el ticket promedio, análisis de clientes, o cualquier otra consulta relacionada con el valor promedio por transacción."
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
        id="ticket-chat-messages-container",  # CAMBIADO
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minHeight": "100px",
        },
    )
