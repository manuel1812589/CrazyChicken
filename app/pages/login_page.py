import dash
from dash import html, dcc, Input, Output, State, callback
import dash_bootstrap_components as dbc
from flask_login import login_user
from app.auth.session import User
from app.services.auth_service import authenticate_user

dash.register_page(__name__, path="/login")

layout = html.Div(
    className="login-page",
    children=[
        html.Div(className="login-background"),
        dbc.Container(
            fluid=True,
            children=[
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            className="login-card",
                            children=[
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Dashboard Pollería",
                                                    className="mb-2",
                                                ),
                                                html.P(
                                                    "Sistema de Análisis Comercial",
                                                    className="text-muted",
                                                ),
                                            ],
                                            className="login-header",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Label(
                                                    "Usuario",
                                                    className="form-label",
                                                ),
                                                dbc.Input(
                                                    id="username",
                                                    type="text",
                                                    placeholder="Ingrese su usuario",
                                                    className="mb-3",
                                                    autoComplete="username",
                                                ),
                                                dbc.Label(
                                                    "Contraseña",
                                                    className="form-label",
                                                ),
                                                dbc.Input(
                                                    id="password",
                                                    type="password",
                                                    placeholder="Ingrese su contraseña",
                                                    className="mb-4",
                                                    autoComplete="current-password",
                                                ),
                                                dbc.Button(
                                                    "Iniciar Sesión",
                                                    id="login-btn",
                                                    color="primary",
                                                    size="lg",
                                                    className="login-button mb-3",
                                                    n_clicks=0,
                                                ),
                                                html.Div(
                                                    id="login-msg",
                                                    className="text-center mt-3",
                                                ),
                                                dcc.Location(
                                                    id="redirect", refresh=True
                                                ),
                                            ],
                                            className="login-form",
                                        ),
                                        html.Div(
                                            [
                                                html.Hr(className="my-4"),
                                                html.Small(
                                                    "© 2024 Dashboard Pollería - Sistema de Análisis Comercial",
                                                    className="text-muted text-center d-block",
                                                ),
                                            ],
                                            className="login-footer",
                                        ),
                                    ]
                                )
                            ],
                        ),
                        xs=12,
                        sm=10,
                        md=8,
                        lg=6,
                        xl=4,
                    ),
                    className="justify-content-center align-items-center",
                    style={"minHeight": "100vh"},
                ),
            ],
        ),
    ],
)


@callback(
    [
        Output("login-msg", "children"),
        Output("login-msg", "className"),
        Output("redirect", "pathname"),
    ],
    Input("login-btn", "n_clicks"),
    [State("username", "value"), State("password", "value")],
    prevent_initial_call=True,
)
def login(n_clicks, username, password):
    if not username or not password:
        return (
            "⚠️ Por favor complete todos los campos",
            "text-danger text-center mt-3",
            dash.no_update,
        )

    user = authenticate_user(username, password)

    if user:
        login_user(User(user.id, user.nombre))
        return (
            "✅ Inicio de sesión exitoso",
            "text-success text-center mt-3",
            "/",
        )

    return (
        "❌ Usuario o contraseña incorrectos",
        "text-danger text-center mt-3",
        dash.no_update,
    )
