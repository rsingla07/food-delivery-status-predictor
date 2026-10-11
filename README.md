# Food Delivery Status Predictor

Predicts whether a food delivery will be Fast or Delayed (split at median
delivery time) using Logistic Regression with GridSearchCV tuning.

## Workflow
Data cleaning → EDA → Encoding & scaling → Logistic Regression →
Hyperparameter tuning → Evaluation → Streamlit deployment

## Results
Test accuracy ≈ 0.48, ROC AUC ≈ 0.39. The model performs at baseline level.
Correlation analysis shows all features have near-zero correlation with
Delivery_Time, so the dataset appears to contain no learnable signal.
L1 regularization also zeroed out 10 of the 11 feature coefficients.

## Run the app locally
pip install -r requirements.txt
python -m streamlit run app.py

## Files
- project_code.ipynb: full analysis notebook (Google Colab)
- app.py: Streamlit app
- *.pkl: saved model, scaler, encoders and mappings