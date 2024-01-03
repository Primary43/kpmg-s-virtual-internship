import dash_bootstrap_components as dbc
from dash import html

# Dropdown Component-----------------------------------------------
ml_models_dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("DecisionTree", href="https://app-db1-5d66c8de929e.herokuapp.com/db1"),
        dbc.DropdownMenuItem("KNeighborsClassifier", href="https://app-db1-5d66c8de929e.herokuapp.com/db2"),
        dbc.DropdownMenuItem("EnsembleVoting", href="https://app-db1-5d66c8de929e.herokuapp.com/db3"),
        dbc.DropdownMenuItem("DeepLearning",
                             href="https://app-db1-5d66c8de929e.herokuapp.com/deep-learning",
                             target="_blank"),
    ],
    nav=True,
    in_navbar=True,
    label="XAI Models",
)

# Navbar Component------------------------------------------------
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Model Development", href="/")),
        dbc.NavItem(dbc.NavLink("Prediction", href="https://app-predict-ed883b6bdb3f.herokuapp.com")),
        dbc.NavItem(dbc.NavLink("Dashboard",
                                href="/dashboard",
                                target="_blank")),
        ml_models_dropdown,
    ],
    brand="ModelHub",
    brand_href="/",
    color="primary",
    dark=True,
)
# Modify layout with Navbar ------------------------------------------------
def add_navbar_to_dashboard(dashboard, navbar):
    if hasattr(dashboard, 'app') and hasattr(dashboard.app, 'layout'):
        original_layout = dashboard.app.layout
        dashboard.app.layout = html.Div([navbar, original_layout])

# Slideshow Component------------------------------------------------
carousel = dbc.Carousel(
    items=[
        {"key": "2", "src": "/assets/demo2.png", "href":"https://app.powerbi.com/view?r=eyJrIjoiZjUwZTM2YTYtOGVjMi00ZTJlLThjYTQtNDhkZTg1ZmQ0Y2VhIiwidCI6ImZkYzBiMDZlLWJiZGYtNDkyNS1iZDBhLTg2ZDg0OTNiOGFmMSJ9"},
        {"key": "3", "src": "/assets/demo3.png", "href":"https://app.powerbi.com/view?r=eyJrIjoiZjUwZTM2YTYtOGVjMi00ZTJlLThjYTQtNDhkZTg1ZmQ0Y2VhIiwidCI6ImZkYzBiMDZlLWJiZGYtNDkyNS1iZDBhLTg2ZDg0OTNiOGFmMSJ9"},
        {"key": "1", "src": "/assets/demo1.png", "href":"https://app.powerbi.com/view?r=eyJrIjoiZjUwZTM2YTYtOGVjMi00ZTJlLThjYTQtNDhkZTg1ZmQ0Y2VhIiwidCI6ImZkYzBiMDZlLWJiZGYtNDkyNS1iZDBhLTg2ZDg0OTNiOGFmMSJ9"},
    ],
    controls=True,
    indicators=True,
    interval=3000,
    ride="carousel",
    variant="dark",
    className="carousel-fade",
    id='carousel',  # Assign an ID to the carousel for the callback
)

carousel2 = dbc.Carousel(
    items=[
        {"key": "2", "src": "/assets/deploy2.png", "href":"https://app-predict-ed883b6bdb3f.herokuapp.com"},
        {"key": "1", "src": "/assets/deploy1.png", "href":"https://app-predict-ed883b6bdb3f.herokuapp.com"},
    ],
    controls=True,
    indicators=True,
    interval=2750,
    ride="carousel",
    variant="dark",
    className="carousel-fade",
    id='carousel',  # Assign an ID to the carousel for the callback
)
