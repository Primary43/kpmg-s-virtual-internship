from flask import Flask
from joblib import load
from components import navbar, add_navbar_to_dashboard
from main_roc_auc_component import MainRocAucComponent
import os
from explainerdashboard import ExplainerDashboard
from explainerdashboard.custom import *

import dash
from dash import html, Dash, Input, Output, dcc, callback
import dash_bootstrap_components as dbc


# Initialize explainer components
explainer_tree = load("explainer/explainer_tree.joblib")
explainer_knc = load("explainer/explainer_knc.dill")
explainer_vote = load("explainer/explainer_vote.dill")

server = Flask(__name__)
# Dash app
app = Dash(__name__, server=server, url_base_pathname="/", suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create the custom tab, Dash app and layout pages------------------------------------------------
RocAucComponent_tab = MainRocAucComponent(explainer_tree, explainer_knc, explainer_vote)

db1 = ExplainerDashboard.from_config("yaml/dashboard1.yaml",server=server, url_base_pathname="/db1/")
db2 = ExplainerDashboard.from_config("yaml/dashboard2.yaml",server=server, url_base_pathname="/db2/")
db3 = ExplainerDashboard.from_config("yaml/dashboard3.yaml",server=server, url_base_pathname="/db3/")

add_navbar_to_dashboard(db1, navbar)
add_navbar_to_dashboard(db2, navbar)
add_navbar_to_dashboard(db3, navbar)

# Main Page
main_layout = html.Div([RocAucComponent_tab.layout()])

# Dalex-Arena (DeepExplain) Page
deep_layout = html.Div([
    html.Iframe(id='external-site',
                src='https://arena.drwhy.ai/?session=https://gist.githubusercontent.com/Primary43/c5e05c45d0a0a3261dd0b9d9e178e289/raw/868aa59313bb42befe22100bed952eb18c21c27f/session.json',
                style={'height': '1000px',
                       'width': '100%'})
])

# Dashboard Page
dashboard_layout = html.Div([
    html.Iframe(id='external-site',
                src='https://app.powerbi.com/view?r=eyJrIjoiZjUwZTM2YTYtOGVjMi00ZTJlLThjYTQtNDhkZTg1ZmQ0Y2VhIiwidCI6ImZkYzBiMDZlLWJiZGYtNDkyNS1iZDBhLTg2ZDg0OTNiOGFmMSJ9',
                style={'height': '1000px',
                       'width': '100%'})
])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# Serve the layout based on the URL pathname
def serve_layout(pathname):
    if pathname == '/deep-learning':
        return deep_layout
    elif pathname == '/dashboard':
        return dashboard_layout
    elif pathname == '/db1':
        return db1.server.index()
    elif pathname == '/db2':
        return db2.server.index()
    elif pathname == '/db3':
        return db3.server.index()
    else:
        return main_layout

# Callback to switch between layouts
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    return serve_layout(pathname)

# call component
RocAucComponent_tab.register_callbacks(app)

# Run Servers ------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5100))  # Default port is 5100 if not on Heroku
    app.run_server(debug=True, host='0.0.0.0', port=port)






