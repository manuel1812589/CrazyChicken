from dash import Dash, Input, Output, html, dcc, page_container, State
from dash.exceptions import PreventUpdate
from functools import wraps
from dash import ctx
import dash_bootstrap_components as dbc
from app.auth.session import login_manager
from app.layouts.main_layout import main_layout
from app.components.module_data import get_module_by_href
from config.settings import DEBUG
import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_PATH = os.path.join(BASE_PATH, "assets")

external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    dbc.icons.FONT_AWESOME,
    "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
]

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    prevent_initial_callbacks=True,
    external_stylesheets=external_stylesheets,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder=ASSETS_PATH,
)
app.css.config.serve_locally = True


def safe_callback_for(module_href):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            active_module = ctx.states.get("active-module.data")
            if active_module != module_href:
                raise PreventUpdate
            return func(*args, **kwargs)

        return wrapper

    return decorator


@app.server.route("/test-css")
def test_css():
    css_path = os.path.join(ASSETS_PATH)
    if os.path.exists(css_path):
        return f"CSS encontrado en: {css_path}"
    else:
        return f"CSS NO encontrado. Buscando en: {css_path}"


@app.callback(
    [
        Output("layout-wrapper", "style"),
        Output("auth-content", "children"),
    ],
    Input("url", "pathname"),
)
def route_layout(pathname):
    if pathname == "/login":
        return (
            {"display": "none"},
            page_container,
        )

    return (
        {"display": "block"},
        None,
    )


@app.callback(
    Output("active-module", "data"),
    Input("url", "pathname"),
)
def set_active_module(pathname):
    module = get_module_by_href(pathname)
    if module:
        return module["href"]
    return "home"


@app.callback(
    Output("url", "pathname"), Input("url", "pathname"), prevent_initial_call=True
)
def _dummy(_):
    raise PreventUpdate


server = app.server
login_manager.init_app(server)

server.config.update(
    SESSION_COOKIE_SECURE=not DEBUG,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SECRET_KEY=os.getenv("SECRET_KEY", "fallback-solo-para-desarrollo"),
)

app.title = "Dashboard Pollería"
app.layout = main_layout()

if __name__ == "__main__":
    app.run_server(debug=True, port=8050, host="0.0.0.0")
