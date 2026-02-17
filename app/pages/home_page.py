import dash
from dash import html
import dash_bootstrap_components as dbc
from app.components.module_card import create_module_card
from app.components.module_data import get_home_modules

dash.register_page(__name__, path="/")

modules_data = get_home_modules()

module_cards = []
for module in modules_data:
    card = dbc.Col(
        create_module_card(
            title=module["title"],
            description=module["description"],
            icon_class=module["icon"],
            color=module["color"],
            badges=module["badges"],
            href=module["href"],
            button_text=module["button_text"],
        ),
        lg=4,
        md=6,
        className="mb-4",
    )
    module_cards.append(card)

layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Dashboard de Análisis", className="page-title mb-3"),
                        html.P(
                            "Bienvenido al sistema de análisis comercial. Selecciona un módulo para comenzar a explorar los datos.",
                            className="page-subtitle text-muted",
                        ),
                    ],
                    className="mb-2",
                )
            ],
            className="page-header",
        ),
        html.Div(
            [
                html.H3("Módulos Disponibles", className="section-title mb-4"),
                dbc.Row(module_cards),
            ]
        ),
        html.Div(
            [
                html.Hr(className="my-5"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H5(
                                    "Dashboard Pollería",
                                    className="mb-2",
                                    style={"color": "var(--primary-color)"},
                                ),
                                html.P(
                                    "Sistema de análisis comercial para la gestión inteligente de tu pollería.",
                                    className="text-muted mb-0 small",
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.Div(
                            [
                                html.Small(
                                    "© 2024 Pollería Dashboard • Versión 2.0",
                                    className="text-muted",
                                ),
                            ],
                            className="text-end",
                        ),
                    ],
                    className="d-flex justify-content-between align-items-center",
                ),
            ],
            className="app-footer",
        ),
    ],
    className="home-container",
)
