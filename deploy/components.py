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
        dbc.NavItem(dbc.NavLink("Model Development", href="https://app-db1-5d66c8de929e.herokuapp.com")),
        dbc.NavItem(dbc.NavLink("Prediction", href="/")),
        dbc.NavItem(dbc.NavLink("Dashboard",
                                href="https://app-db1-5d66c8de929e.herokuapp.com/dashboard",
                                target="_blank")),
        ml_models_dropdown,
    ],
    brand="ModelHub",
    brand_href="https://app-db1-5d66c8de929e.herokuapp.com",
    color="primary",
    dark=True,
)
# Modify layout with Navbar ------------------------------------------------
def add_navbar_to_dashboard(dashboard, navbar):
    if hasattr(dashboard, 'app') and hasattr(dashboard.app, 'layout'):
        original_layout = dashboard.app.layout
        dashboard.app.layout = html.Div([navbar, original_layout])