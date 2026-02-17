ICON_MAPPING = {
    "home": "fas fa-home",
    "ventas": "fas fa-chart-line",
    "ticket": "fas fa-receipt",
    "ventas-cantidad": "fas fa-box",
    "crecimiento": "fas fa-arrow-trend-up",
    "vendedores": "fas fa-user-tie",
    "productos": "fas fa-drumstick-bite",
    "participacion_productos": "fas fa-pie-chart",
}

MODULES = [
    {
        "title": "Inicio",
        "sidebar_title": "Inicio",
        "description": "Página principal del dashboard con acceso a todos los módulos.",
        "icon": ICON_MAPPING["home"],
        "sidebar_icon": ICON_MAPPING["home"],
        "color": "primary",
        "badges": ["Resumen", "Acceso Rápido", "Módulos", "Dashboard"],
        "href": "/",
        "button_text": "Ir al Inicio",
        "sidebar_show": True,
        "home_show": False,
    },
    {
        "title": "Análisis de Ventas Totales",
        "sidebar_title": "Ventas",
        "description": "Análisis de Ventas Totales. Analiza el desempeño comercial de tu pollería.",
        "icon": ICON_MAPPING["ventas"],
        "sidebar_icon": ICON_MAPPING["ventas"],
        "color": "primary",
        "badges": ["KPIs de Ventas", "Tendencias", "Comparativas", "Gráficos"],
        "href": "/kpi-ventas",
        "button_text": "Acceder al Dashboard",
        "sidebar_show": True,
        "home_show": True,
    },
    {
        "title": "Ticket Promedio",
        "sidebar_title": "Ticket Promedio",
        "description": "Análisis del ticket promedio por venta y comparación con metas. Evalúa el valor promedio por transacción.",
        "icon": ICON_MAPPING["ticket"],
        "sidebar_icon": ICON_MAPPING["ticket"],
        "color": "success",
        "badges": ["KPIs", "Comparativas", "Metas", "Tendencias"],
        "href": "/kpi-ticket-promedio",
        "button_text": "Acceder al Dashboard",
        "sidebar_show": True,
        "home_show": True,
    },
    {
        "title": "Total de Ventas (Cantidad)",
        "sidebar_title": "Total de Ventas (Cantidad)",
        "description": "Análisis del Total de Ventas por Cantidad. Analiza el los pollos vendidos por Mes",
        "icon": ICON_MAPPING["ventas-cantidad"],
        "sidebar_icon": ICON_MAPPING["ventas-cantidad"],
        "color": "warning",
        "badges": ["KPIs", "Comparativas", "Metas", "Tendencias", "Cantidad"],
        "href": "/kpi-cantidad-ventas",
        "button_text": "Acceder al Dashboard",
        "sidebar_show": True,
        "home_show": True,
    },
    {
        "title": "Crecimiento de Ventas",
        "sidebar_title": "Crecimiento",
        "description": "Análisis del crecimiento porcentual mensual de las ventas, comparación con la meta del 2% y tendencias de desempeño.",
        "icon": ICON_MAPPING["crecimiento"],
        "sidebar_icon": ICON_MAPPING["crecimiento"],
        "color": "info",
        "badges": [
            "Crecimiento %",
            "Meta 2%",
            "Tendencias",
            "IA",
            "KPIs",
        ],
        "href": "/kpi-crecimiento-ventas",
        "button_text": "Ver Crecimiento de Ventas",
        "sidebar_show": True,
        "home_show": True,
    },
    {
        "title": "Ventas Promedio por Vendedor",
        "sidebar_title": "Vendedores",
        "description": "Análisis del desempeño promedio de vendedores, comparación con la meta de $20,000 y evaluación individual del equipo.",
        "icon": ICON_MAPPING["vendedores"],
        "sidebar_icon": ICON_MAPPING["vendedores"],
        "color": "secondary",
        "badges": [
            "Promedios",
            "Meta $20k",
            "Equipo de Ventas",
            "Rendimiento",
            "IA",
        ],
        "href": "/kpi-ventas-promedio-vendedor",
        "button_text": "Ver Rendimiento de Vendedores",
        "sidebar_show": True,
        "home_show": True,
    },
    {
        "title": "Productos Vendidos por Tipo",
        "sidebar_title": "Productos",
        "description": "Análisis de productos vendidos por tipo de plato, desempeño del mix de productos y comparación con metas de volumen.",
        "icon": ICON_MAPPING["productos"],
        "sidebar_icon": ICON_MAPPING["productos"],
        "color": "danger",
        "badges": [
            "Mix de Productos",
            "Categorías",
            "Volumen",
            "Metas",
            "IA",
        ],
        "href": "/kpi-productos-vendidos",
        "button_text": "Ver Productos Vendidos",
        "sidebar_show": True,
        "home_show": True,
    },
    {
        "title": "Participación de Productos",
        "sidebar_title": "Participación",
        "description": "Análisis de la participación porcentual de productos en las ventas totales, incluyendo top productos, concentración de mercado y asistencia de IA.",
        "icon": ICON_MAPPING["participacion_productos"],
        "sidebar_icon": ICON_MAPPING["participacion_productos"],
        "color": "danger",
        "badges": ["Participación %", "KPIs", "IA", "Análisis", "Pareto"],
        "href": "/kpi-participacion-productos",
        "button_text": "Ver Dashboard de Participación",
        "sidebar_show": True,
        "home_show": True,
    },
]


def get_home_modules():
    return [module for module in MODULES if module["home_show"]]


def get_sidebar_modules():
    return [module for module in MODULES if module["sidebar_show"]]


def get_module_by_href(href):
    for module in MODULES:
        if module["href"] == href:
            return module
    return None
