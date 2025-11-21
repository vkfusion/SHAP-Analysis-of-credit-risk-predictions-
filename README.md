# SHAP Analysis of Credit Risk Predictions (An Explainable AI Approach with XGBoost)
End-to-end machine learning project for predicting credit default using XGBoost and interpreting model predictions using SHAP values.

## Badges
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-red)
![Status](https://img.shields.io/badge/Project-Active-brightgreen)

## Project Overview
This project aims to develop a robust machine learning model for predicting credit default risk in a financial institution. Using a widely recognized credit card client dataset, an **XGBoost** classifier was trained to identify customers at high risk of default. A key focus of this project was to provide model interpretability using **SHAP (SHapley Additive exPlanations)**, allowing us to understand the reasoning behind the model's predictions.

## Key Methodology & Technologies

* **Data Analysis & Preprocessing:** Handling of categorical variables with one-hot encoding, and preparing the dataset for model training.
* **Model Development:** The dataset was split into training and testing sets using `stratify=y` to handle class imbalance.
* **Algorithm:** **XGBoost Classifier**
* **Evaluation Metrics:** The model's performance was evaluated using `Accuracy`, `Precision`, `Recall`, `F1-Score`, and `ROC AUC Score`.
* **Model Explainability:** **SHAP** library was used to analyze both global and local feature importance.

## Project Outcomes & Insights

* **Reliable Performance:** The model achieved a strong **81.6% accuracy** and a **0.7712 ROC AUC score**, demonstrating its effectiveness in predicting credit risk.
* **Most Influential Features:** The SHAP analysis revealed that a client's recent payment status (`PAY_0`, `PAY_2`) is the most significant factor in predicting default risk, confirming that past behavior is a powerful predictor.
* **Explainable Decisions:** With SHAP's **Force Plot**, we can now explain the specific factors that drive each individual prediction, offering a transparent and data-driven approach to credit decisions.

## Repository Structure
- Default_Risk_Prediction_using_XGBoost_and_SHAP.ipynb
- default_of_credit_card_clients.xls
- Outputs/
- readme.md



## Dataset Description
See UCI Credit Card Default Dataset.

## Installation
```bash
pip install xgboost shap pandas numpy scikit-learn matplotlib seaborn
```

## Run Notebook
```bash
jupyter notebook Default_Risk_Prediction_using_XGBoost_and_SHAP.ipynb
```

## How to Run the Project

This project was developed using Google Colab. To run the code:

1.  Clone this repository to your local machine.
2.  Open the `.ipynb` file in a Google Colab notebook.
3.  Install the required libraries: `!pip install pandas scikit-learn xgboost shap matplotlib seaborn`
4.  Run the notebook cells sequentially to reproduce the analysis and results.
## License
MIT License.
