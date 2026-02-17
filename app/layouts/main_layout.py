from dash import html, dcc, page_container
import dash_bootstrap_components as dbc
from app.components.navbar import navbar
from app.components.sidebar import sidebar


def main_layout():
    return html.Div(
        [
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="active-module"),
            html.Div(id="auth-content"),
            html.Div(
                id="layout-wrapper",
                children=[
                    navbar("Dashboard Pollería"),
                    html.Div(
                        [
                            html.Div(
                                sidebar(),
                                id="sidebar",
                                className="sidebar-container",
                            ),
                            html.Div(
                                page_container,
                                className="content-container p-4",
                                id="main-content",
                            ),
                        ],
                        className="app-container d-flex",
                        id="app-container",
                    ),
                ],
            ),
        ]
    )
