from dash import html, page_container


def auth_layout():
    """Layout para páginas de autenticación (login, logout) sin navbar/sidebar"""
    return html.Div(
        [page_container],
        className="auth-layout-container",
    )
