from explainerdashboard.custom import *
from components import carousel, carousel2

# MainDashboard Component ------------------------------------------------
class MainRocAucComponent(ExplainerComponent):
    def __init__(self, explainer1, explainer2, explainer3):
        super().__init__(explainer1, explainer2, explainer3)  # Pass both explainers to the superclass
        self.confmat1 = RocAucComponent(explainer1, cutoff=0.5,
                                        hide_selector=True, hide_popout=True, hide_cutoff=True, hide_footer=True,
                                        hide_title=True, hide_poweredby=True, )
        self.confmat2 = RocAucComponent(explainer2, cutoff=0.5,
                                        hide_selector=True, hide_popout=True, hide_cutoff=True, hide_footer=True,
                                        hide_title=True, hide_poweredby=True, )
        self.confmat3 = RocAucComponent(explainer3, cutoff=0.5,
                                        hide_selector=True, hide_popout=True, hide_cutoff=True, hide_footer=True,
                                        hide_title=True, hide_poweredby=True, )
    def layout(self):
        return dbc.Container([
            self.create_model_section(),
            self.create_dashboard_section(),
            self.create_xai_section(),
            self.create_deploymodel_section()
        ])

    def create_model_section(self):
        return html.Div([
            dbc.Row(html.H1("Model Development", className="display-3 text-primary")),
            dbc.Row(html.P(
                "For the purpose of understanding customer value among a group of 1000 new customers lacking transaction history, a supervised classification model was developed. This model is trained on an engineered target—specifically, a segmentation derived from the RFM (Recency, Frequency, Monetary value) model analyzing the behavior of existing customers with transaction records."),
                    className="lead"),
            html.H1("Engineered Target: Clustered RFM", className="display-20"),
            dbc.Row(html.A(
                children=[
                    html.Img(
                        src="/assets/rfm.png",
                        style={
                            'height': 'auto',  # Maintain aspect ratio
                            'width': '100%',  # Use '100%' to make it responsive to the container width
                            'max-width': '100%',  # Set a max-width to ensure it's not too large on bigger screens
                            'border': 'none',  # Remove border if any
                        }
                    )
                ],
                className="text-center"
            )),
            dbc.Row(html.P(
                "The result reveals low correlations between RFM clusters and other features, suggesting non-linear relationships. Facing challenges in identifying the underlying characteristics of the target RFM cluster, we implemented supervised learning algorithms to discern patterns and classify the target clusters."),
                    className="lead"),
            html.H1("Two-Step Classification Approach", className="display-20"),
            dbc.Row(html.P(
                "Given the presence of class imbalance and insufficient information in the dataset, the two-step binary classification model has proven to be more effective than the multi-classification approach, which has encountered challenges in accurately predicting each individual class."),
                    className="lead"),
            dbc.Row(html.P(
                "The resulted active customer from the first model are fed into the second training model to distinguish a level of customer value. After filtering out non-active customer, these models were trained to identify high/mid-value customer (class 1) vs low-value customer (class 0)"),
                    className="lead"),
            dbc.Row(html.A(
                children=[
                    html.Img(
                        src="/assets/two-step.png",
                        style={
                            'height': 'auto',
                            'width': '100%',
                            'max-width': '100%',
                            'border': 'none',
                        }
                    )
                ],
                className="text-center"
            )),

            dbc.Row(html.Hr(className="my-2")),
            dbc.Row(html.Br()),
        ])
    def create_deploymodel_section(self):
        return html.Div([
            dbc.Row(dbc.Col(html.H1("Model Prediction ", className="display-3 text-primary"))),
            html.P("Implementing a model deployment capable of real-time prediction for both new input or existing customers based on demographic data.", className="lead"),
            dbc.Row(dbc.Container([
                carousel2  # interval  # Include the interval component in your layout
            ])),
            dbc.Row(dbc.Col(html.Hr(className="my-2"))),
            dbc.Row(dbc.Col(html.Br())),
        ])
    def create_dashboard_section(self):
        return html.Div([
            dbc.Row(dbc.Col(html.H1("Customer Segmentation Dashboard ", className="display-3 text-primary"))),
            html.P("A Distribution of Customer Segmentation and its Characteristic", className="lead"),
            dbc.Row(dbc.Container([
                carousel  # interval  # Include the interval component in your layout
            ])),

            dbc.Row(dbc.Col(html.Hr(className="my-2"))),
            dbc.Row(dbc.Col(html.Br())),
        ])

    def create_xai_section(self):
        return html.Div([
            dbc.Row(dbc.Col(html.H1("Explainable Artificial Intelligence (XAI) ", className="display-3 text-primary"))),
            dbc.Row(dbc.Col(html.P(
                "The other trained machine learning model dashboards explain transparant inner workings with the selected samples of thier predictions.The dashboard for machine learning and deep learning models provides a clear view of their decision-making process by presenting prediction samples and employing SHAP (SHapley Additive exPlanations) from game theory. This method quantifies each feature's influence on a prediction, indicating whether it raises or lowers the likelihood of a particular outcome. Essentially, SHAP values clarify how each feature sways the model's predictions."),
                            className="lead")),
            dbc.Row(dbc.Col(html.H1("DeepLearning Model Explainer"), className="text-center")),
            dbc.Row(dbc.Col(
                html.H4("Model 1: NonActive Customer Identification / Model 2:  High-Value Customer Identification"),
                className="text-center")),
            dbc.Row(dbc.Col(html.P(""), className="text-center")),
            dbc.Row(dbc.Col(html.A(
                href="https://app-db1-5d66c8de929e.herokuapp.com/deep-learning",
                target="_blank",
                children=[
                    html.Img(
                        src="/assets/arena.png",
                        style={
                            'height': 'auto',
                            'width': 'auto',
                            'max-width': '100%',
                            'border': 'none',
                        }
                    )
                ],
                className="text-center"
            ))),

            dbc.Row(html.Br()),
            dbc.Row(dbc.Col(html.H1("MachineLearning Model Explainer"), className="text-center")),
            dbc.Row(dbc.Col(html.H4("Model 2:  High-Value Customer Identification"), className="text-center")),
            dbc.Row([
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Button("DecisionTree", href="https://app-db1-5d66c8de929e.herokuapp.com/db1", color="primary"),
                            self.confmat1.layout()
                        ])
                    ]),
                    md=4
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Button("KNeighborsClassifier", href="https://app-db1-5d66c8de929e.herokuapp.com/db2",
                                       color="primary"),
                            self.confmat2.layout()
                        ])
                    ]),
                    md=4
                ),
                dbc.Col(
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Button("EnsembleVoting", href="https://app-db1-5d66c8de929e.herokuapp.com/db3", color="primary"),
                            self.confmat3.layout()
                        ])
                    ]),
                    md=4
                )
            ])
        ])