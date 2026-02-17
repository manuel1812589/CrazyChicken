import dash_bootstrap_components as dbc
from dash import html, dcc


def navbar(title="Dashboard Pollería"):
    return dbc.Navbar(
        dbc.Container(
            [
                html.Div(
                    [
                        html.H4(title, className="fw-bold text-white mb-0"),
                        html.Small(
                            "Sistema de Análisis Comercial", className="text-light"
                        ),
                    ],
                    className="ms-0",
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem("Perfil"),
                        dbc.DropdownMenuItem("Configuración"),
                        dbc.DropdownMenuItem(divider=True),
                        dbc.DropdownMenuItem(
                            "Cerrar Sesión", href="/logout", external_link=True
                        ),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Usuario",
                    align_end=True,
                ),
            ],
            fluid=True,
            className="d-flex align-items-center justify-content-between",
        ),
        color="primary",
        dark=True,
        fixed="top",
        className="shadow-sm navbar-primary",
        style={"z-index": 1030},
    )
