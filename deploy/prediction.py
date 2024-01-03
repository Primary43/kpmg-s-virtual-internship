import pickle
from tensorflow.keras.models import load_model
from model_pipeline import DataPreprocessor
import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, State
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# load/ apply preprocessing / make a prediction
class ModelPredictor:
    def __init__(self, model1_path='model/model1.h5', scaler1_path='model/scaler1.pkl',
                 model2_path='model/model2.h5', scaler2_path='model/scaler2.pkl'):
        self.model1 = load_model(model1_path)
        self.scaler1 = pickle.load(open(scaler1_path, 'rb'))
        self.model2 = load_model(model2_path)
        self.scaler2 = pickle.load(open(scaler2_path, 'rb'))

    def predict(self, df):
        # Apply preprocessing using DataPreprocessor instance
        preprocessor = DataPreprocessor()  # You may need to pass any required arguments to the constructor
        df_processed = preprocessor.process(df)

        # Scale the data for model 1
        df_scaled = self.scaler1.transform(df_processed)

        # Make predictions with the first model
        prob_predictions1 = self.model1.predict(df_scaled)
        labels1 = self.post_process_predictions(prob_predictions1)

        # Store prediction in df
        pred_df = df.copy()
        final_predictions = labels1.copy()
        pred_df['label'] = final_predictions

        #DataFrame for model 1 probabilities
        prob_df_model1 = pd.DataFrame({
            'prob_non_active': 1 - prob_predictions1[:, 0],
            'prob_active': prob_predictions1[:, 0]
        }, index=df.index)

        # Select instances predicted as 'active' to predict with model 2
        active_indices = labels1.flatten() == 1
        active_instances = df_processed[active_indices]

        prob_df_model2 = pd.DataFrame(columns=['prob_active_low', 'prob_active_high'])

        # If there are any active instances, predict their class with model 2
        if not active_instances.empty:
            # Scale the data for model 2
            active_instances_scaled = self.scaler2.transform(active_instances)

            # Make predictions with the second model
            prob_predictions2 = self.model2.predict(active_instances_scaled)
            prob_df_model2 = pd.DataFrame({
                'prob_active_low': 1 - prob_predictions2[:, 0],
                'prob_active_high': prob_predictions2[:, 0]
            }, index=active_instances.index)
            labels2 = self.post_process_predictions(prob_predictions2)

            # Combine predictions from both models
            final_predictions[
                labels1.flatten() == 1] = labels2 + 1
            pred_df = df.copy()
            pred_df['label'] = final_predictions

        return prob_df_model1, prob_df_model2, pred_df

    def post_process_predictions(self, predictions):
        # Apply threshold to convert probabilities to binary labels
        return (predictions > 0.5).astype(int)

# Display prediction result with probabilities percentage
def display_prediction_results(probabilities_model1, probabilities_model2=None):
    # Format probabilities as percentages for model 1
    probabilities_model1_percent = probabilities_model1 * 100
    print("Probabilities from Model 1:")
    print(probabilities_model1_percent.round(2).astype(str) + '%')
    # Check if the probabilities_model2 DataFrame is empty
    if probabilities_model2 is not None:
        # If it's not empty, format probabilities as percentages for model 2
        probabilities_model2_percent = probabilities_model2 * 100
        print("Probabilities from Model 2:")
        print(probabilities_model2_percent.round(2).astype(str) + '%')
    else:
        # If probabilities_model2 is empty, just display a message
        print("No additional probabilities from Model 2 to display.")

# process request for existing data or new input
def process_prediction(input_index=None, input_type='existing_data', input_df=None):
    if input_type == 'existing_data':
        # Load existing data
        df = pd.read_csv('data/NewCustomer_df_input.csv', index_col=0)
        if input_index not in df.index:
            print(f"Customer ID: {input_index} not found in existing data.")
            results = [html.H5(f"Customer ID: {input_index} not found in existing data.")]
            return results

        data_row = df.loc[[input_index]]
        data_row['owns_car'] = data_row['owns_car'].map({True:'True', False:'False'})
        # Create the DataTable
        table = create_table_from_prediction(data_row)
        # Predict
        predictor = ModelPredictor()
        prob1_df, prob2_df, pred_df = predictor.predict(data_row)
        # Display the prediction results
        print(f"Prediction for Customer ID: {input_index}")
        # Check if prob2_df is not empty and the input_index exists in prob2_df
        if not prob2_df.empty and input_index in prob2_df.index:
            display_prediction_results(prob1_df.loc[input_index], prob2_df.loc[input_index])
        else:
            # If prob2_df is empty or the index does not exist, handle accordingly
            display_prediction_results(prob1_df.loc[input_index], None)

        # Initialize the Div that will contain the results
        results = []
        results.append(html.H5(f"Prediction for Customer ID: {input_index}"))
        # Assume 'table' is defined somewhere above as a dash_table.DataTable
        results.append(table)
        plot_probability_results(results, prob1_df, prob2_df)
        build_probability_results(results, prob1_df, prob2_df)
        return html.Div(results, style={'textAlign': 'center', 'width': '100%'})

    elif input_type == 'new_input':
        table = create_table_from_prediction(input_df)
        input_df['owns_car'] = input_df['owns_car'].map({'True': True, 'False': False})
        # initial results
        results = []
        results = [html.H5(f"Prediction for New Customer:")]
        results.append(table)
        predictor = ModelPredictor()
        prob1_df, prob2_df, pred_df = predictor.predict(input_df)
        print(prob1_df, prob2_df, pred_df)
        plot_probability_results(results, prob1_df, prob2_df)
        build_probability_results(results, prob1_df, prob2_df)
        return html.Div(results, style={'textAlign': 'center', 'width': '100%'})


def process_prediction_(input_index=None, input_type='existing_data', input_df=None):
    if input_type == 'existing_data':
        # Load existing data
        df = pd.read_csv('data/NewCustomer_df_input.csv', index_col=0)
        if input_index not in df.index:
            print(f"Customer ID: {input_index} not found in existing data.")
            results = [html.H5(f"Customer ID: {input_index} not found in existing data.")]
            return results

        data_row = df.loc[[input_index]]
        # Create the DataTable
        table = create_table_from_prediction(data_row)
        # Predict
        predictor = ModelPredictor()
        prob1_df, prob2_df, pred_df = predictor.predict(data_row)
        # Display the prediction results
        print(f"Prediction for Customer ID: {input_index}")
        # Check if prob2_df is not empty and the input_index exists in prob2_df
        if not prob2_df.empty and input_index in prob2_df.index:
            display_prediction_results(prob1_df.loc[input_index], prob2_df.loc[input_index])
        else:
            # If prob2_df is empty or the index does not exist, handle accordingly
            display_prediction_results(prob1_df.loc[input_index], None)

        # Initialize the Div that will contain the results
        results = [html.H5(f"Prediction for Customer ID: {input_index}")]
        # Assume 'table' is defined somewhere above as a dash_table.DataTable
        results.append(table)
        plot_probability_results(results, prob1_df, prob2_df)
        build_probability_results(results, prob1_df, prob2_df)
        return html.Div(results)

    elif input_type == 'new_input':
        table = create_table_from_prediction(input_df)
        input_df['owns_car'] = input_df['owns_car'].map({'True': True, 'False': False})
        # initial results
        results = [html.H5(f"Prediction for New Customer:")]
        results.append(table)
        predictor = ModelPredictor()
        prob1_df, prob2_df, pred_df = predictor.predict(input_df)
        print(prob1_df, prob2_df, pred_df)
        build_probability_results(results, prob1_df, prob2_df)
        return html.Div(results)

def build_probability_results(results, prob1_df, prob2_df=None):
    # Add probabilities from Model 1
    results.append(html.H6("Probabilities from Model 1 (Active Customer Identification):"))
    prob_non_active_percentage = prob1_df['prob_non_active'].iloc[0] * 100
    prob_active_percentage = prob1_df['prob_active'].iloc[0] * 100
    results.append(html.P(f"Non-Active Probability: {prob_non_active_percentage:.2f}%"))
    results.append(html.P(f"Active Probability: {prob_active_percentage:.2f}%"))

    # Check if prob2_df is provided and add its contents to the results
    if prob2_df is not None and not prob2_df.empty:
        results.append(html.H6("Probabilities from Model 2 (High-Value Customer Identification):"))
        prob_active_low_percentage = prob2_df['prob_active_low'].iloc[0] * 100
        prob_active_high_percentage = prob2_df['prob_active_high'].iloc[0] * 100
        results.append(html.P(f"Low Value Probability: {prob_active_low_percentage:.2f}%"))
        results.append(html.P(f"High Value Probability: {prob_active_high_percentage:.2f}%"))
    else:
        results.append(html.H6("No additional probabilities from Model 2 to display."))
    return results

def plot_probability_results(results, prob1_df, prob2_df=None):
    # Add text probabilities from Model 1
    prob_non_active_percentage = prob1_df['prob_non_active'].iloc[0] * 100
    prob_active_percentage = prob1_df['prob_active'].iloc[0] * 100

    # Prepare data for the stacked bar chart
    bar_data = [
        go.Bar(name='Non-Active', x=['Model 1: Active Customer Identification'], y=[prob_non_active_percentage], marker_color='rgb(255,193,7)'),
        go.Bar(name='Active', x=['Model 1: Active Customer Identification'], y=[prob_active_percentage], marker_color='rgb(13,110,253)')
    ]

    # Check if prob2_df is provided and add its contents to the bar chart
    if prob2_df is not None and not prob2_df.empty:
        prob_active_low_percentage = prob2_df['prob_active_low'].iloc[0] * 100
        prob_active_high_percentage = prob2_df['prob_active_high'].iloc[0] * 100

        bar_data.extend([
            go.Bar(name='Active Low-Value', x=['Model 2: High-Value Customer Identification'], y=[prob_active_low_percentage], marker_color='rgb(97,161,254)'),
            go.Bar(name='Active High-Value', x=['Model 2: High-Value Customer Identification'], y=[prob_active_high_percentage], marker_color='rgb(0,0,135)')
        ])

    # Define the layout for the bar chart
    layout = go.Layout(
        barmode='stack',
        title='Probabilities from Models:',
        yaxis=dict(title='Probability (%)', ticksuffix='%')
    )

    # Create the figure with data and layout
    fig = go.Figure(data=bar_data, layout=layout)

    # Add the figure to the results list as a Dash Graph component
    graph = dcc.Graph(
        id='probability_chart',
        figure=fig
    )
    results.append(graph)
    return results

def create_table_from_prediction(pred_df):
    # Convert prediction DataFrame to Dash DataTable or HTML table
    return dbc.Table.from_dataframe(
        pred_df, index=True,
        bordered=True, color="secondary", hover=True, responsive=True, striped=True,
    )

