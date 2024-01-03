import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
from prediction import process_prediction
from components import navbar
import os
from flask import Flask


server = Flask(__name__)
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, server=server, url_base_pathname="/", external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)


# Main layout Tabs ---------------------------------------
app.layout = html.Div([
    navbar,
    html.H1("Customer Segment Prediction"),
    dcc.Tabs(id="input_type", value='new_input', children=[
        dcc.Tab(label='Existing Customer', value='existing_data'),
        dcc.Tab(label='New Customer', value='new_input'),
    ]),
    html.Div(id='tabs-content')
])

# ComponentTab 1 ---------------------------------------
form1 = html.Div([
    dbc.Row([
            dbc.Col(dbc.Label("Enter Customer ID:")),
        ], className="g-0 ",
    ),
    dbc.Row([
            dbc.Col(dbc.Input(id='input_index', type="text", placeholder="Enter Customer ID")),
            dbc.Col(dbc.Button("Submit", id='submit-button1', n_clicks=0, color="primary")),
        ], className="g-0 ",
    ),
])
# Content for tab 1
tab1_content = html.Div([
    form1,
    html.Div(id='output-prediction1')
])

# ComponentTab 2 ---------------------------------------

Gender_Radio = html.Div([
            dbc.Label("Gender:"),
            dbc.RadioItems(
                options=[
                    {"label": "Female", "value": "Female"},
                    {"label": "Male", "value": "Male"}
                ],
                id="input-gender",
                value="Female",
                inline=True,
            ),
])

State_Radio = html.Div([
            dbc.Label("State:"),
            dbc.RadioItems(
                options=[
                    {"label": "NSW", "value": "NSW"},
                    {"label": "QLD", "value": "QLD"},
                    {"label": "VIC", "value": "VIC"},
                ],
                id="input-state",
                value="VIC",
                inline=True,
            ),
])

Wealth_Radio = html.Div([
            dbc.Label("Wealth Segment:"),
            dbc.RadioItems(
                options=[
                  {"label": "Affluent Customer", "value": "Affluent Customer"},
                  {"label": "High Net Worth", "value": "High Net Worth"},
                  {"label": "Mass Customer", "value": "Mass Customer"}
                ],
                id="input-wealth-segment",
                value="Affluent Customer",  # Default value
                inline=True,
            ),
])

OwnCars_Radio = html.Div([
            dbc.Label("Owns Car:"),
            dbc.RadioItems(
                options=[
                  {"label": "True", "value": "True"},
                  {"label": "False", "value": "False"}
                ],
                id="input-owns-car",
                value="True",  # Default value
                inline=True,
            ),
])

Age_Slider = html.Div([
            dbc.Label("Age:"),
            dcc.Slider(
                id='input-age',
                min=18,
                max=99,
                step=1,
                value=30,  # Default value
                marks={i: str(i) for i in range(0, 100, 10)},  # Marks on the slider
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ])

Property_Slider = html.Div([
            dbc.Label("Property Value:"),
            dcc.Slider(
                id='input-property',
                min=0,
                max=12,
                step=1,
                value=10,  # Default value
                marks={i: str(i) for i in range(0, 12, 2)},  # Marks on the slider
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ])

Tenure_Slider = html.Div([
            dbc.Label("Tenure:"),
            dcc.Slider(
                id='input-tenure',
                min=0,
                max=25,
                step=1,
                value=10,  # Default value
                marks={i: str(i) for i in range(0, 25, 2)},  # Marks on the slider
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ])


PastBike_Slider = html.Div([
            dbc.Label("Past 3 years bike related purchases:"),
            dcc.Slider(
                id='input-pastbike',
                min=0,
                max=100,
                step=1,
                value=10,  # Default value
                marks={i: str(i) for i in range(0, 100, 10)},  # Marks on the slider
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ])

job_industry_dropdown = html.Div([
            dbc.Label("Job Industry:"),
            dcc.Dropdown(
                id='input-job-industry',
                options=[
                    {'label': 'Agriculture', 'value': 'Agriculture'},
                    {'label': 'Entertainment', 'value': 'Entertainment'},
                    {'label': 'Financial Services', 'value': 'Financial Services'},
                    {'label': 'Health', 'value': 'Health'},
                    {'label': 'IT', 'value': 'IT'},
                    {'label': 'Manufacturing', 'value': 'Manufacturing'},
                    {'label': 'Property', 'value': 'Property'},
                    {'label': 'Retail', 'value': 'Retail'},
                    {'label': 'Telecommunications', 'value': 'Telecommunications'}
                ],
                value='Agriculture',
                clearable=False,  # Optional: to prevent user from leaving the dropdown empty
                searchable=True,  # Optional: to allow user to search options in the dropdown
            )
])

# Extract unique job titles
job_cluster_df = pd.read_csv('data/job_cluster_df.csv', index_col=0)
unique_job_titles = job_cluster_df['job_title'].sort_values().unique()
job_title_options = [{'label': title, 'value': title} for title in unique_job_titles]

job_title_dropdown = html.Div([
            dbc.Label("Job Title:"),
            dcc.Dropdown(
                id='input-job-title',
                options=job_title_options,
                value=unique_job_titles[0],
                clearable=False,
                searchable=True,
            )
])

# Content for tab 2
tab2_content = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                # First column of inputs
                dbc.Col([
                    Gender_Radio,
                    Age_Slider,
                    State_Radio,
                    Wealth_Radio,
                    OwnCars_Radio
                ], width='6'),  # Adjust the width as needed for your layout

                # Second column of inputs
                dbc.Col([
                    job_industry_dropdown,
                    job_title_dropdown,
                    Property_Slider,
                    Tenure_Slider,
                    PastBike_Slider
                ], width='6'),  # Adjust the width as needed for your layout
            ]),

            # Submit button centered below the columns
            dbc.Row([
                dbc.Button("Submit", id="submit-button2", color="primary")
            ], className="col-6 mx-auto"),
        ])
    ),

    # Div for displaying the output
    html.Div(id='output-prediction2')
], className="mt-3")  # Add some top margin to the entire card


# Callback display tab ---------------------------------------
@callback(Output('tabs-content', 'children'),
          Input('input_type', 'value'))
# Display each tab
def render_content(tab):
    if tab == 'existing_data':
        return tab1_content
    elif tab == 'new_input':
        return tab2_content


# Callback output tab1 ---------------------------------------
@callback(Output('output-prediction1', 'children'),
          [Input('submit-button1', 'n_clicks')],
          [State('input_index', 'value')],
          prevent_initial_call=True)

def display_prediction_results(n_clicks, input_index):
    if n_clicks and n_clicks > 0:  # Check if the button was clicked
        if input_index is None or input_index == '':
            # If the input is blank, return an error message
            return html.Div("Please enter a Customer ID.")
        try:
            # converting it to an integer
            input_index = int(input_index)
        except ValueError:
            # If the input is not a valid integer, return an error message
            return html.Div("Invalid Customer ID. Please enter a valid number.")
        # Continue with processing if input_index is valid
        return process_prediction(input_index)
    # If n_clicks is None or 0, which means the button has not been clicked, return an empty div
    return html.Div()

# Callback output tab2
@callback(Output('output-prediction2', 'children'),
          [Input('submit-button2', 'n_clicks'),
           Input('input-gender', 'value'),
           Input('input-wealth-segment', 'value'),
           Input('input-state', 'value'),
           Input('input-job-title', 'value'),
           Input('input-job-industry', 'value'),
           Input('input-owns-car', 'value'),
           Input('input-pastbike', 'value'),
           Input('input-tenure', 'value'),
           Input('input-property', 'value'),
           Input('input-age', 'value')
    ]
)
def update_output(n_clicks, gender, wealth_segment, state, job_title, job_industry_category, owns_car, past_3_years_bike_related_purchases, tenure, property_valuation, age):
    if n_clicks is None:
        return '' # No clicks yet, do nothing
    else:
        # Create a dictionary
        data = {
            'gender': gender,
            'wealth_segment': wealth_segment,
            'state': state,
            'job_title': job_title,
            'job_industry_category': job_industry_category,
            'owns_car': owns_car,
            'past_3_years_bike_related_purchases': past_3_years_bike_related_purchases,
            'tenure': tenure,
            'property_valuation': property_valuation,
            'age': age,
        }
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame([data])
        return process_prediction(input_type='new_input', input_df=df)
    return html.Div()

# Run Servers ------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4200))  # Default port is 5100 if not on Heroku
    app.run_server(debug=True, host='0.0.0.0', port=port)
