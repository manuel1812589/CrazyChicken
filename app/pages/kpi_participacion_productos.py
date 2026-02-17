import dash
from dash import html, dcc, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.services.data_service import get_participacion_productos_2024
from app.components.graph_card import create_graph_card
from app.services.ai_service import (
    generar_analisis_participacion_productos,
    generar_respuesta_chat,
)
import time

dash.register_page(__name__, path="/kpi-participacion-productos")

df_all = get_participacion_productos_2024()
meses_disponibles = [
    {"label": "Enero", "value": 1},
    {"label": "Febrero", "value": 2},
    {"label": "Marzo", "value": 3},
    {"label": "Abril", "value": 4},
    {"label": "Mayo", "value": 5},
    {"label": "Junio", "value": 6},
    {"label": "Julio", "value": 7},
    {"label": "Agosto", "value": 8},
    {"label": "Septiembre", "value": 9},
    {"label": "Octubre", "value": 10},
    {"label": "Noviembre", "value": 11},
    {"label": "Diciembre", "value": 12},
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
                                    "Participación de Productos",
                                    className="page-title mb-2",
                                ),
                                html.P(
                                    "Análisis de la participación porcentual de productos en las ventas totales",
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
                                        id="participacion-mes-filter",
                                        options=[
                                            {"label": "Todos los meses", "value": "all"}
                                        ]
                                        + meses_disponibles,
                                        value="all",
                                        placeholder="Selecciona un mes...",
                                        className="mb-3",
                                    ),
                                    html.Small(
                                        "Nota: El año siempre será 2024 • Meta: 10% de participación por producto",
                                        className="text-muted",
                                    ),
                                ]
                            ),
                            title="Filtros",
                            description="Selecciona meses específicos para analizar",
                            height=230,
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
                        id="participacion-ai-analysis-output",
                        className="p-3 border rounded bg-light",
                        style={"whiteSpace": "pre-line"},
                    ),
                    id="participacion-analysis-spinner",
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("Asistente IA de Negocio", className="section-title mb-3"),
                dcc.Store(id="participacion-chat-history", data=[]),
                dcc.Store(id="participacion-ai-thinking", data=False),
                html.Div(
                    id="participacion-chat-window",
                    className="chat-window mb-3",
                    style={"maxHeight": "400px", "overflowY": "auto"},
                ),
                html.Div(
                    id="participacion-typing-indicator",
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
                            id="participacion-chat-input",
                            placeholder="Haz una pregunta sobre participación de productos...",
                            type="text",
                            disabled=False,
                        ),
                        dbc.Button(
                            "Enviar",
                            id="participacion-chat-send",
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
                html.H4("Participación por Producto", className="section-title mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="participacion-bar-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Top Productos por Participación",
                                description="Productos con mayor participación en las ventas",
                                height=450,
                            ),
                            lg=6,
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            create_graph_card(
                                graph=dcc.Graph(
                                    id="participacion-pareto-graph",
                                    config={"displayModeBar": True},
                                ),
                                title="Análisis Pareto",
                                description="Principio 80/20 de participación de productos",
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
                html.H4("Análisis de Concentración", className="section-title mb-4"),
                create_graph_card(
                    graph=html.Div(
                        [
                            html.Div(
                                [
                                    dbc.RadioItems(
                                        id="participacion-view-toggle",
                                        className="btn-group",
                                        inputClassName="btn-check",
                                        labelClassName="btn btn-outline-primary",
                                        labelCheckedClassName="active",
                                        options=[
                                            {
                                                "label": "Por Categoría",
                                                "value": "categoria",
                                            },
                                            {
                                                "label": "Comparación con Meta",
                                                "value": "meta",
                                            },
                                            {
                                                "label": "Mapa de Calor",
                                                "value": "heatmap",
                                            },
                                        ],
                                        value="categoria",
                                    ),
                                ],
                                className="radio-group mb-4",
                            ),
                            html.Div(id="participacion-dynamic-graph-container"),
                        ]
                    ),
                    title="Análisis Detallado",
                    description="Diferentes perspectivas de la participación de productos",
                    height=500,
                ),
            ],
            className="mb-4",
        ),
        html.Div(
            [
                html.H4("KPIs de Participación", className="section-title mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            create_graph_card(
                                graph=html.Div(
                                    [
                                        html.Div(
                                            id="participacion-kpi-cards",
                                            className="kpi-cards-grid",
                                        ),
                                    ]
                                ),
                                title="Métricas Clave",
                                description="Resumen de la participación de productos",
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
                            "Dashboard de Participación de Productos • Actualizado: Diciembre 2024",
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
        Output("participacion-bar-graph", "figure"),
        Output("participacion-pareto-graph", "figure"),
    ],
    [Input("participacion-mes-filter", "value")],
)
def update_static_graphs(mes_filtro):
    df = get_participacion_productos_2024(mes_filtro)

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

    df_top = df.nlargest(15, "participacion_porcentual").copy()

    colores = [
        "#2ecc71" if participacion >= 10 else "#e74c3c"
        for participacion in df_top["participacion_porcentual"]
    ]

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=df_top["producto"],
                y=df_top["participacion_porcentual"],
                marker_color=colores,
                text=df_top["participacion_formateada"],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Participación: %{text}<br>Ventas: $%{customdata:,.0f}<extra></extra>",
                customdata=df_top["ventas_totales"],
            ),
            go.Scatter(
                x=df_top["producto"],
                y=[10] * len(df_top),
                mode="lines",
                name="Meta (10%)",
                line=dict(color="#3498db", width=2, dash="dash"),
                hovertemplate="Meta: 10%<extra></extra>",
            ),
        ],
        layout=go.Layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=100),
            title_text="Top 15 Productos por Participación (%)",
            title_font=dict(size=18, color="var(--primary-color)"),
            xaxis_title="Producto",
            yaxis_title="Participación (%)",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            xaxis=dict(tickangle=45),
        ),
    )

    df_pareto = df.sort_values("participacion_porcentual", ascending=False).copy()
    df_pareto["cumulative_percentage"] = df_pareto["participacion_porcentual"].cumsum()

    pareto_fig = go.Figure()

    pareto_fig.add_trace(
        go.Bar(
            x=df_pareto["producto"],
            y=df_pareto["participacion_porcentual"],
            name="Participación",
            marker_color="#3498db",
            hovertemplate="<b>%{x}</b><br>Participación: %{y:.2f}%<extra></extra>",
        )
    )

    pareto_fig.add_trace(
        go.Scatter(
            x=df_pareto["producto"],
            y=df_pareto["cumulative_percentage"],
            name="Acumulado",
            yaxis="y2",
            line=dict(color="#e74c3c", width=3),
            hovertemplate="<b>%{x}</b><br>Acumulado: %{y:.2f}%<extra></extra>",
        )
    )

    pareto_fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
        margin=dict(l=50, r=50, t=50, b=100),
        title_text="Análisis Pareto de Productos",
        title_font=dict(size=18, color="var(--primary-color)"),
        xaxis_title="Producto",
        yaxis=dict(
            title="Participación (%)",
            titlefont=dict(color="#3498db"),
            tickfont=dict(color="#3498db"),
        ),
        yaxis2=dict(
            title="Acumulado (%)",
            titlefont=dict(color="#e74c3c"),
            tickfont=dict(color="#e74c3c"),
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis=dict(tickangle=45),
    )

    return bar_fig, pareto_fig


@callback(
    Output("participacion-dynamic-graph-container", "children"),
    [
        Input("participacion-view-toggle", "value"),
        Input("participacion-mes-filter", "value"),
    ],
)
def update_dynamic_graph(view_type, mes_filtro):
    df = get_participacion_productos_2024(mes_filtro)

    if df.empty:
        return html.Div(
            "No hay datos para el filtro seleccionado",
            className="text-center text-muted py-5",
        )

    if view_type == "categoria":
        df_categoria = df.groupby("categoria", as_index=False).agg(
            {
                "participacion_porcentual": "sum",
                "ventas_totales": "sum",
                "unidades_vendidas": "sum",
                "producto": "count",
            }
        )

        df_categoria = df_categoria.sort_values(
            "participacion_porcentual", ascending=False
        )

        fig = px.sunburst(
            df,
            path=["categoria", "producto"],
            values="ventas_totales",
            title="",
            color="participacion_porcentual",
            color_continuous_scale="RdBu",
            hover_data={"ventas_totales": "$,.0f", "participacion_porcentual": ":.2f%"},
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=50),
            title_text="Participación por Categoría y Producto",
            title_font=dict(size=18, color="var(--primary-color)"),
        )

        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Ventas: $%{customdata[0]:,.0f}<br>Participación: %{color:.2f}%<extra></extra>",
        )

    elif view_type == "meta":
        df_meta = df.copy()
        df_meta["estado"] = df_meta["cumple_meta"].apply(
            lambda x: "Sobre Meta" if x else "Bajo Meta"
        )

        fig = go.Figure()

        df_sobre_meta = df_meta[df_meta["cumple_meta"]]
        df_bajo_meta = df_meta[~df_meta["cumple_meta"]]

        if not df_sobre_meta.empty:
            fig.add_trace(
                go.Bar(
                    x=df_sobre_meta["producto"],
                    y=df_sobre_meta["participacion_porcentual"],
                    name="Sobre Meta",
                    marker_color="#2ecc71",
                    hovertemplate="<b>%{x}</b><br>Participación: %{y:.2f}%<br>Ventas: $%{customdata:,.0f}<extra></extra>",
                    customdata=df_sobre_meta["ventas_totales"],
                )
            )

        if not df_bajo_meta.empty:
            fig.add_trace(
                go.Bar(
                    x=df_bajo_meta["producto"],
                    y=df_bajo_meta["participacion_porcentual"],
                    name="Bajo Meta",
                    marker_color="#e74c3c",
                    hovertemplate="<b>%{x}</b><br>Participación: %{y:.2f}%<br>Ventas: $%{customdata:,.0f}<extra></extra>",
                    customdata=df_bajo_meta["ventas_totales"],
                )
            )

        fig.add_hline(
            y=10,
            line_dash="dash",
            line_color="#3498db",
            annotation_text="Meta: 10%",
            annotation_position="top right",
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(size=12),
            margin=dict(l=50, r=50, t=50, b=100),
            title_text="Participación vs Meta (10%)",
            title_font=dict(size=18, color="var(--primary-color)"),
            xaxis_title="Producto",
            yaxis_title="Participación (%)",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
            xaxis=dict(tickangle=45),
        )

    else:
        if df["categoria"].nunique() > 1:
            df_pivot = df.pivot_table(
                values="participacion_porcentual",
                index="producto",
                columns="categoria",
                aggfunc="first",
            ).fillna(0)

            fig = px.imshow(
                df_pivot,
                title="",
                color_continuous_scale="Viridis",
                labels={"color": "Participación (%)"},
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(size=12),
                margin=dict(l=50, r=50, t=50, b=50),
                title_text="Mapa de Calor: Participación por Producto y Categoría",
                title_font=dict(size=18, color="var(--primary-color)"),
                xaxis_title="Categoría",
                yaxis_title="Producto",
            )

            fig.update_traces(
                hovertemplate="<b>Producto: %{y}</b><br>Categoría: %{x}<br>Participación: %{z:.2f}%<extra></extra>",
            )
        else:
            fig = go.Figure()
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                title_text="Se necesita más de una categoría para el mapa de calor",
                annotations=[
                    dict(
                        text="Agrega más meses para ver múltiples categorías",
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                        font=dict(size=16),
                    )
                ],
            )

    return dcc.Graph(
        id="participacion-line-graph",
        figure=fig,
        config={"displayModeBar": True},
        className="dynamic-graph",
    )


@callback(
    Output("participacion-kpi-cards", "children"),
    [Input("participacion-mes-filter", "value")],
)
def update_kpi_cards(mes_filtro):
    df = get_participacion_productos_2024(mes_filtro)

    if df.empty:
        return html.Div(
            "No hay datos para el filtro seleccionado",
            className="text-center text-muted py-3",
        )

    total_productos = df.shape[0]
    productos_sobre_meta = df[df["cumple_meta"]].shape[0]
    porcentaje_sobre_meta = (
        (productos_sobre_meta / total_productos) * 100 if total_productos > 0 else 0
    )

    participacion_top5 = df.nlargest(5, "participacion_porcentual")[
        "participacion_porcentual"
    ].sum()

    producto_lider = df.loc[df["participacion_porcentual"].idxmax(), "producto"]
    participacion_lider = df["participacion_porcentual"].max()

    concentracion_indice = df.nlargest(3, "participacion_porcentual")[
        "participacion_porcentual"
    ].sum()

    cards = [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H5(
                            f"{productos_sobre_meta}/{total_productos}",
                            className="card-title",
                        ),
                        html.P(
                            "Productos sobre Meta", className="card-text text-muted"
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
                        html.H5(f"{participacion_top5:.1f}%", className="card-title"),
                        html.P("Top 5 Productos", className="card-text text-muted"),
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
                        html.H5(f"{participacion_lider:.1f}%", className="card-title"),
                        html.P(
                            f"Líder: {producto_lider[:15]}...",
                            className="card-text text-muted",
                        ),
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
                        html.H5(f"{concentracion_indice:.1f}%", className="card-title"),
                        html.P("Concentración Top 3", className="card-text text-muted"),
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
    Output("participacion-ai-analysis-output", "children"),
    Output("participacion-analysis-spinner", "type"),
    Input("participacion-mes-filter", "value"),
)
def update_ai_analysis(mes_filtro):
    df = get_participacion_productos_2024(mes_filtro)

    if df.empty:
        return "No hay datos para generar análisis.", "border"

    meta_participacion = 0.10
    analisis = generar_analisis_participacion_productos(df, meta_participacion)

    return analisis, "border"


@callback(
    Output("participacion-chat-history", "data", allow_duplicate=True),
    Output("participacion-chat-input", "value"),
    Output("participacion-typing-indicator", "style"),
    Output("participacion-ai-thinking", "data"),
    Output("participacion-chat-input", "disabled"),
    Output("participacion-chat-send", "disabled"),
    Input("participacion-chat-send", "n_clicks"),
    State("participacion-chat-input", "value"),
    State("participacion-chat-history", "data"),
    State("participacion-ai-thinking", "data"),
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
    Output("participacion-chat-history", "data"),
    Output("participacion-typing-indicator", "style", allow_duplicate=True),
    Output("participacion-ai-thinking", "data", allow_duplicate=True),
    Output("participacion-chat-input", "disabled", allow_duplicate=True),
    Output("participacion-chat-send", "disabled", allow_duplicate=True),
    Input("participacion-ai-thinking", "data"),
    State("participacion-chat-history", "data"),
    State("participacion-mes-filter", "value"),
    prevent_initial_call=True,
)
def generate_ai_response(is_thinking, history, mes_filtro):
    if not is_thinking or not history:
        return dash.no_update, {"display": "none"}, False, False, False

    user_messages = [msg for msg in history if msg.get("role") == "user"]
    if not user_messages:
        return dash.no_update, {"display": "none"}, False, False, False

    last_user_msg = user_messages[-1]["text"]

    df = get_participacion_productos_2024(mes_filtro)

    if df.empty:
        ai_response = "No hay datos disponibles para responder tu pregunta."
    else:
        contexto = f"""
        DATOS DE PARTICIPACIÓN DE PRODUCTOS:
        
        Total productos analizados: {df.shape[0]}
        Productos sobre meta (10%): {df[df['cumple_meta']].shape[0]} de {df.shape[0]}
        Ventas totales: ${df['ventas_totales'].sum():,.0f}
        
        Productos líderes (top 5 por participación):
        """

        df_top5 = df.nlargest(5, "participacion_porcentual")
        for idx, row in df_top5.iterrows():
            contexto += f"\n- {row['producto']}: {row['participacion_formateada']} (${row['ventas_totales']:,.0f})"

        contexto += f"""
        
        Concentración de mercado:
        - Top 3 productos: {df.nlargest(3, 'participacion_porcentual')['participacion_porcentual'].sum():.1f}%
        - Top 5 productos: {df_top5['participacion_porcentual'].sum():.1f}%
        - Top 10 productos: {df.nlargest(10, 'participacion_porcentual')['participacion_porcentual'].sum():.1f}%
        
        Producto líder: {df.loc[df['participacion_porcentual'].idxmax(), 'producto']} ({df['participacion_porcentual'].max():.2f}%)
        Participación promedio: {df['participacion_porcentual'].mean():.2f}%
        Desviación estándar: {df['participacion_porcentual'].std():.2f}%
        
        Distribución por categorías:
        """

        df_categoria = df.groupby("categoria", as_index=False).agg(
            {"participacion_porcentual": "sum", "producto": "count"}
        )

        for _, row in df_categoria.iterrows():
            contexto += f"\n- {row['categoria']}: {row['participacion_porcentual']:.1f}% ({row['producto']} productos)"

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
    Output("participacion-chat-window", "children"),
    Input("participacion-chat-history", "data"),
)
def render_chat(history):
    bubbles = []

    if not history:
        welcome_msg = "¡Hola! Soy tu asistente de análisis de participación de productos. Puedes preguntarme sobre productos líderes en participación, análisis del principio 80/20, comparaciones con la meta del 10%, concentración de mercado por producto, recomendaciones para equilibrar el portafolio, o cualquier otra consulta relacionada con la participación porcentual de productos en las ventas."
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
        id="participacion-chat-messages-container",
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "10px",
            "minHeight": "100px",
        },
    )
