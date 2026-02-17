import dash_bootstrap_components as dbc
from dash import html, dcc


def create_graph_card(
    graph=None,
    title="Gráfico",
    description="",
    height=400,
    show_filters=False,
    filters=None,
    full_width=False,
    card_id=None,
):
    children = []

    children.append(
        html.Div(
            [
                html.H4(title, className="graph-title"),
                (
                    html.P(description, className="graph-description text-muted mb-0")
                    if description
                    else None
                ),
            ],
            className="graph-header mb-4",
        )
    )

    if show_filters and filters:
        children.append(
            html.Div(
                filters,
                className="graph-filters mb-4",
            )
        )

    children.append(
        html.Div(
            graph,
            className="graph-content",
            style={"height": f"{height}px"} if height else {},
        )
    )

    card_props = {
        "children": [dbc.CardBody(children)],
        "className": "graph-container" + (" full-width" if full_width else ""),
    }

    if card_id:
        card_props["id"] = card_id

    return dbc.Card(**card_props)
