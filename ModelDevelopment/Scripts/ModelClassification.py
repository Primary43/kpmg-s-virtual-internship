import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import random
# Preprocessing
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Model and Evaluation
from sklearn import metrics
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
# SMOTE for Imbalance
from yellowbrick.classifier import ConfusionMatrix, ClassPredictionError
from yellowbrick.target import ClassBalance
from imblearn.over_sampling import SMOTE
# Visualisation
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import scikitplot as skplt


# prep_classification
def set_seed(seed=42):
    """Set seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Preprocessing training dataset -------------------------------------
def preprocess_trainingdata(df, label_column, test_size=0.2, random_state=42):
    """
    Preprocess the data: split, balance class, scale, and encode the labels.

    Parameters:
    df (DataFrame): The dataframe containing the features and labels.
    label_column (str): The name of the column containing the labels.
    test_size (float): The size of the test set. Default is 0.2.
    random_state (int): The seed used by the random number generator.

    Returns:
    X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_features, n_classes
    """

    # Splitting the data
    X = df.drop([label_column], axis=1)
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Visualizing class balance
    visualizer = ClassBalance()
    visualizer.fit(y_train, y_test)
    visualizer.show()

    # Balancing the classes with SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=random_state)

    # Visualizing class balance after SMOTE
    visualizer = ClassBalance()
    visualizer.fit(y_train, y_test)
    visualizer.show()

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Number of features
    n_features = X_train_scaled.shape[1]

    # Number of classes
    n_classes = len(np.unique(y_train))

    # Encoding the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, n_features, n_classes, label_encoder


# Evaluate model training -------------------------------------
def evaluate_and_plot_model(model, X_test, y_test, history):
    """
    Evaluate the given model using the test data and plot the training
    and validation loss and accuracy from training's history
    """
    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', score[1])

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Evaluate classification report -------------------------------------
def display_classification_results(y_test_encoded, y_pred, label_encoder):
    """
    Displays the classification report, confusion matrix,
    class prediction error.
    """

    # Classification Report
    class_rep_dl = classification_report(y_test_encoded, y_pred)
    print("Deep Learning Classification Report:\n", class_rep_dl)

    # Confusion Matrix
    class_names = label_encoder.classes_
    confusion = confusion_matrix(y_test_encoded, y_pred)

    # Create a new matplotlib figure-------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Axes[0]: Plot the confusion matrix in the first subplot
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=class_names)
    disp.plot(ax=axes[0], cmap=plt.cm.Blues)
    disp.im_.set_clim(0, 250)
    axes[0].set_title("Confusion Matrix")

    # Class Prediction Error
    df_error = pd.DataFrame({'Actual': y_test_encoded, 'Predicted': y_pred})
    df_error = df_error.sort_values('Actual')  # Sort the DataFrame for better visualization

    # Axes[1]: Generate and plot a crosstab
    crosstab_result = pd.crosstab(df_error['Actual'], df_error['Predicted'])
    crosstab_result.plot(kind='bar', stacked=True, ax=axes[1], width=0.8)
    axes[1].set_title("Class Prediction Error")
    axes[1].set_xlabel('Actual Classes')
    axes[1].set_ylabel('Count')
    axes[1].legend(title='Predicted Classes', loc='upper left', bbox_to_anchor=(1, 1))

    # Show the plots
    plt.tight_layout()
    plt.show()


def plot_multiclass_roc(y_test_encoded, y_pred_probs, title='ROC curves for each class'):
    n_classes = y_test_encoded.shape[1]
    roc_auc = []

    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_encoded[:, i], y_pred_probs[:, i])
        roc_auc.append(roc_auc_score(y_test_encoded[:, i], y_pred_probs[:, i]))
        plt.plot(fpr, tpr, label=f'ROC curve of Class {i} (area = {roc_auc[i]:0.4f})')

    # Compute micro-average ROC curve and ROC area
    fpr_micro, tpr_micro, _ = roc_curve(y_test_encoded.ravel(), y_pred_probs.ravel())
    micro_auc = roc_auc_score(y_test_encoded, y_pred_probs, average="micro")
    plt.plot(fpr_micro, tpr_micro, label=f'Micro-average ROC curve (area = {micro_auc:.2f})',
             linestyle=':', linewidth=4)
    plt.plot(np.linspace(0, 1, 100),
         np.linspace(0, 1, 100),
         label='baseline',
         linestyle='--', color='black')

    # Plotting
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def plot_binaryclass_roc(y_test_encoded, y_pred_probs, title='ROC curves'):
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_probs)
    auc = roc_auc_score(y_test_encoded, y_pred_probs)
    #create ROC curve
    plt.plot(np.linspace(0, 1, 100),
         np.linspace(0, 1, 100),
         label='baseline',
         linestyle='--', color='black')
    plt.plot(fpr,tpr,label=f'AUC= {auc:.2f}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.legend(loc=4)
    plt.show()

def plot_multiple_roc_curves(y_test_encoded, model_pred_probs, title='ROC Curves Comparison'):
    """
    Plot ROC curves for multiple models.
    model_pred_probs (dict): Dictionary where keys are model names and values are predicted probabilities.
    """
    plt.plot(np.linspace(0, 1, 100),
             np.linspace(0, 1, 100),
             label='baseline',
             linestyle='--', color='black')

    for model_name, y_pred_probs in model_pred_probs.items():
        fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_probs)
        auc = roc_auc_score(y_test_encoded, y_pred_probs)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC= {auc:.2f})')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title)
    plt.legend(loc=4)
    plt.show()

