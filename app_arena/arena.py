import pandas as pd
import numpy as np
import os
import sys

import dalex as dx
import shap
from ModelClassification import preprocess_trainingdata, set_seed, predict_proba

# Re-Utilise the model
import pickle
from tensorflow.python import keras
from keras.models import load_model
from joblib import dump, load
from flask import Flask, jsonify

# Set up relative paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model1_path = os.path.join(current_dir, 'models', 'model1.h5')
data1_path = os.path.join(current_dir, 'data', 'df1.csv')
model2_path = os.path.join(current_dir, 'models', 'model2.h5')
data2_path = os.path.join(current_dir, 'data', 'df2.csv')

# Load model
model1 = load_model(model1_path)
model2 = load_model(model2_path)

# Load data
df1 = pd.read_csv(data1_path, index_col=0)
df2 = pd.read_csv(data2_path, index_col=0)

# Run model 2 -------------------------------------
X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_features, n_classes, label_encoder , scaler = preprocess_trainingdata(df2, label_column = 'label', test_size=0.2, random_state=42)
# Convert X_train_scaled and X_test_scaled to DataFrames for visualising
original_feature_names = df2.drop(columns=['label']).columns
X_test_df = pd.DataFrame(X_test_scaled, columns=original_feature_names)
y_test_df = pd.DataFrame(y_test_encoded, columns=["target"])

exp_deep2 = dx.Explainer(model2, X_test_df, y_test_encoded,
                         predict_function=lambda model, X: predict_proba(model, X),
                         label='model2_deep',
                         model_type='classification')

# Run model 1 -------------------------------------
X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_features, n_classes, label_encoder , scaler = preprocess_trainingdata(df1, label_column = 'label', test_size=0.2, random_state=42)
# Convert X_train_scaled and X_test_scaled to DataFrames for visualising
original_feature_names = df1.drop(columns=['label']).columns
X_test_df = pd.DataFrame(X_test_scaled, columns=original_feature_names)
y_test_df = pd.DataFrame(y_test_encoded, columns=["target"])
exp_deep1 = dx.Explainer(model1, X_test_df, y_test_encoded,
                         predict_function=lambda model, X: predict_proba(model, X),
                         label='model1_deep',
                         model_type='classification')

#print(exp_deep1.model_performance())
sys.stdout.flush()
# Run Arena
arena=dx.Arena()
arena.push_model(exp_deep2)

arena.push_model(exp_deep1)
arena.push_observations(X_test_df)

if __name__ == '__main__':
    # Heroku assigns the port dynamically so we need to respect that
    port = int(os.environ.get('PORT', 5555))  # Default port is 5000 if not on Heroku
    arena.run_server(host='0.0.0.0', port=port)

