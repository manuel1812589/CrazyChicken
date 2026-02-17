import dash_bootstrap_components as dbc
from dash import html


def create_kpi_card(
    title="KPI",
    value="0",
    unit="",
    change_value="0%",
    change_positive=True,
    icon_class="fas fa-chart-line",
    color="primary",
    description="",
    period="vs mes anterior",
):
    color_classes = {
        "primary": "kpi-primary",
        "success": "kpi-success",
        "danger": "kpi-danger",
        "info": "kpi-info",
        "warning": "kpi-warning",
    }

    color_class = color_classes.get(color, "kpi-primary")

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
                                            html.I(
                                                className=icon_class,
                                                style={
                                                    "fontSize": "1.5rem",
                                                    "color": "white",
                                                },
                                            )
                                        ],
                                        className="kpi-icon",
                                        style={
                                            "background": f"var(--{color}-color, var(--secondary-color))",
                                        },
                                    ),
                                    html.H6(
                                        title,
                                        className="kpi-card-title mb-0",
                                    ),
                                ],
                                className="d-flex align-items-center mb-3",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        value,
                                        className="kpi-value",
                                    ),
                                    html.Span(
                                        unit,
                                        className="kpi-unit",
                                    ),
                                ],
                                className="kpi-main-value mb-2",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.I(
                                                className=f"fas {'fa-arrow-up' if change_positive else 'fa-arrow-down'} me-1",
                                            ),
                                            html.Span(
                                                change_value,
                                                className=f"kpi-change {'positive' if change_positive else 'negative'}",
                                            ),
                                            html.Span(
                                                f" {period}",
                                                className="kpi-period text-muted",
                                            ),
                                        ],
                                        className="kpi-change-section mb-2",
                                    ),
                                    html.P(
                                        description,
                                        className="kpi-description text-muted mb-0 small",
                                    ),
                                ],
                                className="kpi-meta",
                            ),
                        ]
                    )
                ]
            )
        ],
        className=f"kpi-card {color_class}",
    )
