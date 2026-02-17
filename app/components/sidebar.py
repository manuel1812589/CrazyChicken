import dash_bootstrap_components as dbc
from dash import html
from app.components.module_data import get_sidebar_modules


def sidebar():
    modules = get_sidebar_modules()

    sidebar_items = []
    for module in modules:
        sidebar_items.append(
            dbc.NavLink(
                [
                    html.I(className=f"{module['sidebar_icon']} me-3"),
                    module["sidebar_title"],
                ],
                href=module["href"],
                className="nav-link-custom d-flex align-items-center",
                active="exact",
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    dbc.Nav(
                        sidebar_items,
                        vertical=True,
                        pills=True,
                        className="sidebar-nav",
                    ),
                    html.Div(
                        [
                            html.Hr(className="sidebar-divider my-4"),
                            html.Div(
                                [
                                    html.Small(
                                        "© 2024 Pollería Dashboard",
                                        className="text-muted d-block text-center",
                                    ),
                                    html.Small(
                                        "Versión 2.0",
                                        className="text-muted d-block text-center",
                                    ),
                                ],
                                className="sidebar-footer",
                            ),
                        ],
                        className="mt-auto",
                    ),
                ],
                className="d-flex flex-column h-100",
            ),
        ],
        className="sidebar-container",
    )
