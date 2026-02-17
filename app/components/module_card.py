import dash_bootstrap_components as dbc
from dash import html


def create_module_card(
    title="Módulo",
    description="Descripción del módulo",
    icon_class="fas fa-chart-line",
    color="primary",
    badges=None,
    href="#",
    button_text="Acceder",
):
    color_gradients = {
        "primary": "linear-gradient(135deg, #3498db, #2980b9)",
        "danger": "linear-gradient(135deg, #e74c3c, #c0392b)",
        "success": "linear-gradient(135deg, #2ecc71, #27ae60)",
        "info": "linear-gradient(135deg, #9b59b6, #8e44ad)",
        "warning": "linear-gradient(135deg, #f39c12, #d35400)",
    }

    if badges is None:
        badges = ["Característica 1", "Característica 2", "Característica 3"]

    return dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.I(
                                                        className=icon_class,
                                                        style={
                                                            "fontSize": "1.8rem",
                                                            "color": "white",
                                                        },
                                                    )
                                                ],
                                                className="card-icon",
                                                style={
                                                    "background": color_gradients.get(
                                                        color,
                                                        color_gradients["primary"],
                                                    ),
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                },
                                            )
                                        ],
                                        className="text-center mb-3",
                                    ),
                                    html.H4(
                                        title,
                                        className="card-title text-center mb-3",
                                    ),
                                ],
                                className="card-top-section",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P(
                                                description,
                                                className="card-text",
                                            ),
                                        ],
                                        className="card-description mb-3",
                                        style={
                                            "minHeight": "80px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Small(
                                                "Principales funcionalidades:",
                                                className="text-muted d-block mb-2 fw-semibold",
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Badge(
                                                        badge,
                                                        color=color,
                                                        className="me-2 mb-2",
                                                    )
                                                    for badge in badges[:4]
                                                ],
                                                className="card-badges",
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                ],
                                className="card-middle-section flex-grow-1",
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        [
                                            html.I(className="fas fa-arrow-right me-2"),
                                            button_text,
                                        ],
                                        href=href,
                                        color=color,
                                        className="w-100 py-3 fw-semibold",
                                        size="lg",
                                    ),
                                ],
                                className="card-bottom-section mt-auto",
                            ),
                        ],
                        className="d-flex flex-column h-100",
                    )
                ]
            )
        ],
        className="dashboard-card h-100",
    )
