import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.services.data_service import get_ventas_promedio_por_vendedor_2024
from app.components.graph_card import create_graph_card
from app.services.ai_service import (
    generar_analisis_ventas_promedio_vendedor,
    generar_respuesta_chat,
)
import time

dash.register_page(__name__, path="/kpi-ventas-promedio-vendedor")

df_all = get_ventas_promedio_por_vendedor_2024()
meses_disponibles = [
    {"label": row["nombre_mes"], "value": row["mes"]}
    for _, row in df_all.sort_values("mes").drop_duplicates("mes").iterrows()
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
                                    "Ventas Promedio por Vendedor",
                                    className="page-title mb-2",
                                ),
                                html.P(
                                    "Análisis del desempeño promedio de vendedores y comparación con meta",
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
                                        id="vendedor-mes-filter",
                                        options=[
                                            {"label": "Todos los meses", "value": "all"}
                                        ]
                                        + meses_disponibles,
                                        value="all",
                                        placeholder="Selecciona un mes...",
                                        className="mb-3",
                                    ),
                                    html.Small(
                                        "Nota: El año siempre será 2024 • Meta: $20,000 por vendedor",
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
                        id="vendedor-ai-analysis-output",
                        className="p-3 border rounded bg-light",
                        style={"whiteSpace": "pre-line"},
                    ),
                    id="vendedor-analysis-spinner",
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Asistente IA de Negocio", className="section-title mb-3"),
                dcc.Store(id="vendedor-chat-history", data=[]),
                dcc.Store(id="vendedor-ai-thinking", data=False),
                html.Div(
                    id="vendedor-chat-window",
                    className="chat-window mb-3",
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
                html.Div(
                    id="vendedor-typing-indicator",
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
                            id="vendedor-chat-input",
                            placeholder="Haz una pregunta sobre ventas promedio por vendedor...",
                            type="text",
                            disabled=False,
                        ),
                        dbc.Button(
                            "Enviar",
                            id="vendedor-chat-send",
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
                html.H4("Desempeño por Vendedor", className="section-title mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="vendedor-bar-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Ventas por Vendedor",
                                description="Ventas totales por cada vendedor",
                                height=450,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="vendedor-pie-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Distribución de Ventas",
                                description="Participación porcentual de cada vendedor",
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
                html.H4("Análisis de Promedios", className="section-title mb-4"),
                create_graph_card(
                    graph=html.Div(
                        [
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        id="vendedor-view-toggle",
                                        className="btn-group",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[
                                            {
                                                "label": "Ventas Promedio",
                                                "value": "promedio",
                                            },
                                            {
                                                "label": "Comparación Individual",
                                                "value": "individual",
                                            },
                                            {
                                                "label": "Evolución Mensual",
                                                "value": "evolucion",
                                            },
                                        ],
                                        value="promedio",
                                    ),
                                ],
                                className="radio-group mb-4",
                            ),
                            html.Div(id="vendedor-dynamic-graph-container"),
                        ]
                    ),
                    title="Análisis Detallado",
                    description="Diferentes perspectivas del desempeño promedio",
                    height=500,
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("KPIs de Desempeño", className="section-title mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=html.Div(
                                    [
                                        html.Div(
                                            id="vendedor-kpi-cards",
                                            className="kpi-cards-grid",
                                        ),
                                    ]
                                ),
                                title="Métricas Clave",
                                description="Resumen del desempeño de vendedores",
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
                            "Dashboard de Ventas Promedio por Vendedor • Actualizado: Diciembre 2024",
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
        Output("vendedor-bar-graph", "figure"),
        Output("vendedor-pie-graph", "figure"),
    ],
    [Input("vendedor-mes-filter", "value")],
)
def update_static_graphs(mes_filtro):
    df = get_ventas_promedio_por_vendedor_2024(mes_filtro)

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(
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
        return empty_fig, empty_fig

    df_agrupado = df.groupby("vendedor", as_index=False).agg(
        {
            "ventas_totales": "sum",
            "cantidad_ventas": "sum",
            "clientes_atendidos": "sum",
            "productos_vendidos": "sum",
        }
    )

    colores = [
        "#2ecc71" if ventas >= 20000 else "#e74c3c"
        for ventas in df_agrupado["ventas_totales"]
    ]

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=df_agrupado["vendedor"],
                y=df_agrupado["ventas_totales"],
                marker_color=colores,
                text=df_agrupado["ventas_totales"].apply(lambda x: f"${x:,.0f}"),
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Ventas: $%{y:,.0f}<extra></extra>",
            ),
            go.Scatter(
                x=df_agrupado["vendedor"],
                y=[20000] * len(df_agrupado),
                mode="lines",
                name="Meta ($20,000)",
                line=dict(color="#3498db", width=2, dash="dash"),
                hovertemplate="Meta: $20,000<extra></extra>",
            ),
        ],
        layout=go.Layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=100),
            title_text="Ventas Totales por Vendedor vs Meta",
            title_font=dict(size=18, color="var(--primary-color)"),
            xaxis_title="Vendedor",
            yaxis_title="Ventas Totales ($)",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            xaxis=dict(tickangle=45),
        ),
    )

    pie_fig = px.pie(
        df_agrupado,
        names="vendedor",
        values="ventas_totales",
        title="",
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.Viridis,
    )

    pie_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=True,
        font=dict(size=12),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="right", x=1.2),
        title_text="Distribución de Ventas por Vendedor",
        title_font=dict(size=18, color="var(--primary-color)"),
    )
    pie_fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Ventas: $%{value:,.0f}<br>%{percent}<extra></extra>",
    )

    return bar_fig, pie_fig


@callback(
    Output("vendedor-dynamic-graph-container", "children"),
    [Input("vendedor-view-toggle", "value"), Input("vendedor-mes-filter", "value")],
)
def update_dynamic_graph(view_type, mes_filtro):
    df = get_ventas_promedio_por_vendedor_2024(mes_filtro)

    if df.empty:
        return html.Div(
            "No hay datos para el filtro seleccionado",
            className="text-center text-muted py-5",
        )

    if view_type == "promedio":
        df_mensual = (
            df.groupby(["nombre_mes", "mes"], as_index=False)
            .agg(
                {
                    "ventas_promedio_por_vendedor": "first",
                    "cumple_meta_promedio": "first",
                    "ventas_totales": "sum",
                    "vendedor": "nunique",
                }
            )
            .sort_values("mes")
        )

        colores = [
            "#2ecc71" if cumple else "#e74c3c"
            for cumple in df_mensual["cumple_meta_promedio"]
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=df_mensual["nombre_mes"],
                    y=df_mensual["ventas_promedio_por_vendedor"],
                    marker_color=colores,
                    text=df_mensual["ventas_promedio_por_vendedor"].apply(
                        lambda x: f"${x:,.0f}"
                    ),
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Promedio: $%{y:,.0f}<extra></extra>",
                ),
                go.Scatter(
                    x=df_mensual["nombre_mes"],
                    y=[20000] * len(df_mensual),
                    mode="lines",
                    name="Meta ($20,000)",
                    line=dict(color="#3498db", width=2, dash="dash"),
                    hovertemplate="Meta: $20,000<extra></extra>",
                ),
            ],
            layout=go.Layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Ventas Promedio por Vendedor por Mes vs Meta",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Ventas Promedio por Vendedor ($)",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                yaxis=dict(tickformat="$,.0f"),
            ),
        )

    elif view_type == "individual":
        if len(df["vendedor"].unique()) > 1:
            fig = px.box(
                df,
                x="vendedor",
                y="ventas_totales",
                title="",
                points="all",
                labels={
                    "ventas_totales": "Ventas Totales ($)",
                    "vendedor": "Vendedor",
                },
            )

            fig.add_hline(
                y=20000,
                line_dash="dash",
                line_color="red",
                annotation_text="Meta: $20,000",
                annotation_position="top right",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=100),
                title_text="Distribución de Ventas por Vendedor",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Vendedor",
                yaxis_title="Ventas Totales ($)",
                showlegend=False,
                xaxis=dict(tickangle=45),
                yaxis=dict(tickformat="$,.0f"),
            )

            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>Ventas: $%{y:,.0f}<extra></extra>",
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_text="Se necesita más de un vendedor para comparación",
                annotations=[
                    dict(
                        text="Agrega más meses para ver múltiples vendedores",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16),
                    )
                ],
            )

    else:
        if len(df["vendedor"].unique()) > 1:
            fig = px.line(
                df.sort_values(["mes", "vendedor"]),
                x="nombre_mes",
                y="ventas_totales",
                color="vendedor",
                title="",
                markers=True,
                labels={
                    "ventas_totales": "Ventas Totales ($)",
                    "nombre_mes": "Mes",
                    "vendedor": "Vendedor",
                },
            )

            fig.add_hline(
                y=20000,
                line_dash="dash",
                line_color="red",
                annotation_text="Meta: $20,000",
                annotation_position="top right",
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Evolución de Ventas por Vendedor",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Mes",
                yaxis_title="Ventas Totales ($)",
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
                ),
                yaxis=dict(tickformat="$,.0f"),
            )

            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{fullData.name}</b><br>Mes: %{x}<br>Ventas: $%{y:,.0f}<extra></extra>",
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_text="Se necesita más de un vendedor para ver evolución",
                annotations=[
                    dict(
                        text="Agrega más meses para ver múltiples vendedores",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16),
                    )
                ],
            )

    return dcc.Graph(
        id="vendedor-line-graph",
        figure=fig,
        config={"displayModeBar": True},
        className="dynamic-graph",
    )


@callback(
    Output("vendedor-kpi-cards", "children"),
    [Input("vendedor-mes-filter", "value")],
)
def update_kpi_cards(mes_filtro):
    df = get_ventas_promedio_por_vendedor_2024(mes_filtro)

    if df.empty:
        return html.Div(
            "No hay datos para el filtro seleccionado",
            className="text-center text-muted py-3",
        )

    total_ventas = df["ventas_totales"].sum()
    vendedores_unicos = df["vendedor"].nunique()
    promedio_general = total_ventas / vendedores_unicos if vendedores_unicos > 0 else 0

    vendedores_sobre_meta = df[df["cumple_meta"]].shape[0]
    total_vendedores_meses = df.shape[0]
    porcentaje_sobre_meta = (
        (vendedores_sobre_meta / total_vendedores_meses) * 100
        if total_vendedores_meses > 0
        else 0
    )

    mejor_vendedor = (
        df.loc[df["ventas_totales"].idxmax(), "vendedor"] if not df.empty else "N/A"
    )
    mejor_ventas = df["ventas_totales"].max() if not df.empty else 0

    clientes_promedio = df["clientes_atendidos"].sum() / vendedores_unicos
    ventas_promedio_por_cliente = (
        total_ventas / df["clientes_atendidos"].sum()
        if df["clientes_atendidos"].sum() > 0
        else 0
    )

    cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(f"${promedio_general:,.0f}", className="card-title"),
                        html.P(
                            "Promedio por Vendedor", className="card-text text-muted"
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
                            f"{vendedores_sobre_meta}/{total_vendedores_meses}",
                            className="card-title",
                        ),
                        html.P(
                            "Vendedores sobre Meta", className="card-text text-muted"
                        ),
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
                        html.H5(f"{vendedores_unicos}", className="card-title"),
                        html.P("Vendedores Únicos", className="card-text text-muted"),
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
                        html.H5(f"${mejor_ventas:,.0f}", className="card-title"),
                        html.P(
                            f"Mejor: {mejor_vendedor}", className="card-text text-muted"
                        ),
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
    Output("vendedor-ai-analysis-output", "children"),
    Output("vendedor-analysis-spinner", "type"),
    Input("vendedor-mes-filter", "value"),
)
def update_ai_analysis(mes_filtro):
    df = get_ventas_promedio_por_vendedor_2024(mes_filtro)

    if df.empty:
        return "No hay datos para generar análisis.", "border"

    meta_venta_vendedor = 20000
    analisis = generar_analisis_ventas_promedio_vendedor(df, meta_venta_vendedor)

    return analisis, "border"


@callback(
    Output("vendedor-chat-history", "data", allow_duplicate=True),
    Output("vendedor-chat-input", "value"),
    Output("vendedor-typing-indicator", "style"),
    Output("vendedor-ai-thinking", "data"),
    Output("vendedor-chat-input", "disabled"),
    Output("vendedor-chat-send", "disabled"),
    Input("vendedor-chat-send", "n_clicks"),
    State("vendedor-chat-input", "value"),
    State("vendedor-chat-history", "data"),
    State("vendedor-ai-thinking", "data"),
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
    Output("vendedor-chat-history", "data"),
    Output("vendedor-typing-indicator", "style", allow_duplicate=True),
    Output("vendedor-ai-thinking", "data", allow_duplicate=True),
    Output("vendedor-chat-input", "disabled", allow_duplicate=True),
    Output("vendedor-chat-send", "disabled", allow_duplicate=True),
    Input("vendedor-ai-thinking", "data"),
    State("vendedor-chat-history", "data"),
    State("vendedor-mes-filter", "value"),
    prevent_initial_call=True,
)
def generate_ai_response(is_thinking, history, mes_filtro):
    if not is_thinking or not history:
        return dash.no_update, {"display": "none"}, False, False, False

    user_messages = [msg for msg in history if msg.get("role") == "user"]
    if not user_messages:
        return dash.no_update, {"display": "none"}, False, False, False

    last_user_msg = user_messages[-1]["text"]

    df = get_ventas_promedio_por_vendedor_2024(mes_filtro)

    if df.empty:
        ai_response = "No hay datos disponibles para responder tu pregunta."
    else:
        contexto = f"""
        DATOS DE VENTAS PROMEDIO POR VENDEDOR:
        
        Ventas totales: ${df['ventas_totales'].sum():,.0f}
        Vendedores únicos: {df['vendedor'].nunique()}
        Ventas promedio por vendedor: ${df['ventas_totales'].sum()/df['vendedor'].nunique():,.0f}
        Meta por vendedor: $20,000
        
        Vendedores sobre meta: {df[df['cumple_meta']].shape[0]} de {df.shape[0]} ({df[df['cumple_meta']].shape[0]/df.shape[0]*100:.1f}%)
        Promedio mensual por vendedor: ${df['ventas_promedio_por_vendedor'].mean():,.0f}
        
        Mejor vendedor: {df.loc[df['ventas_totales'].idxmax(), 'vendedor']} (${df['ventas_totales'].max():,.0f})
        Promedio de ventas por transacción: ${df['promedio_por_venta'].mean():,.0f}
        Clientes atendidos por vendedor: {df['clientes_atendidos'].sum()/df['vendedor'].nunique():.0f}
        Productos vendidos por vendedor: {df['productos_vendidos'].sum()/df['vendedor'].nunique():.0f}
        
        Desempeño por mes:
        """

        df_mensual = (
            df.groupby(["nombre_mes", "mes"], as_index=False)
            .agg(
                {
                    "ventas_promedio_por_vendedor": "first",
                    "cumple_meta_promedio": "first",
                }
            )
            .sort_values("mes")
        )

        for _, row in df_mensual.iterrows():
            estado = "✅" if row["cumple_meta_promedio"] else "❌"
            contexto += f"\n- {row['nombre_mes']}: ${row['ventas_promedio_por_vendedor']:,.0f} {estado}"

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
    Output("vendedor-chat-window", "children"),
    Input("vendedor-chat-history", "data"),
)
def render_chat(history):
    bubbles = []

    if not history:
        welcome_msg = "¡Hola! Soy tu asistente de análisis de ventas promedio por vendedor. Puedes preguntarme sobre el desempeño individual de vendedores, comparaciones con la meta de $20,000, análisis de vendedores sobresalientes, recomendaciones para mejorar el desempeño del equipo, o cualquier otra consulta relacionada con las ventas promedio por vendedor."
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
        id="vendedor-chat-messages-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minHeight": "100px",
        },
    )
