# Customer-Segmentaion-Prediction-@kpmg-s-virtual-internship
To understand customer value among a group of new customers lacking transaction history, different supervised classification ML and DL models were developed and assessed.
<div align="center">
    <img src="https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/asset/kpmg.png", width=1000>
</div>

## Final WebApp on Heroku: https://app-db1-5d66c8de929e.herokuapp.com/

### 1. Model Development: 
https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/ModelDevelopment/Module_2_kpmg_slide.pdf
- 1.1 Initially, an engineered target was obtained— a segmentation derived from the cluster of RFM (Recency, Frequency, Monetary value) model analyzing the behavior of existing customers with transaction records.
<div align="center">
    <img src="https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/asset/cluster.png", width=750>
</div>
- 1.2 Among different supervised classification ML and DL models, Two-Step Deep learning Classification Approach has proven to be more effective than the multi-classification approach, which has encountered challenges in accurately predicting each class in this given problem, compared to the model baseline (multi-classification deep learning model), Decision Tree, KNeighborsClassifier, EnsembleVoting model.
<div align="center">
    <img src="https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/asset/two-step.png", width=750>
</div>
- 1.3 Explainable AI (XAI) was applied to provide a clear view of each feature's influence on a prediction, indicating whether it raises or lowers the likelihood of a particular outcome. This includes the utilization of ExplainableDashboard and Dalex libraries
<div align="center">
    <img src="https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/asset/explainer.png", width=800>
</div>
<div align="center">
    <img src="https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/asset/arena.png", width=800>
</div>

### 2. Dashboard and Report: 
were presented on PowerBI  providing interactive visualizations and business intelligence capabilities. 
<div align="center">
    <img src="https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/asset/demo1.png", width=800>
</div>

### 3. Web application and model prediction:
were deployed using Dash/ Flask on Heroku for the capability of real-time prediction for both new input and existing customers based on demographic data.
<div align="center">
    <img src="https://github.com/Primary43/kpmg-s-virtual-internship/blob/main/asset/prediction.png", width=800>
</div>

