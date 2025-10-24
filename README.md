# Behavior-Based Prediction of Gamer Engagement

This project is an end-to-end data science workflow to predict a gamer's engagement level (Low, Medium, or High) based on their in-game behavior. The final optimized model is deployed in a real-time, interactive web application built with Gradio.

This repository contains the full project report and the commented Jupyter Notebook with all data preprocessing, exploratory analysis, model training, and deployment steps.

## 🚀 Overview

The goal of this project is to build a classification model that can accurately predict player engagement. This is valuable for game developers, allowing them to:
* Identify players at risk of churning (Low engagement).
* Understand the habits of highly engaged players (High engagement).
* Develop strategies to improve user experience, increase retention, and optimize monetization.

The project uses the `online_gaming_behavior_dataset.csv` and achieves a **final accuracy of 91.67%** using an XGBoost classifier optimized with Optuna.

## ✨ Key Features

* **Data Cleaning:** The dataset was checked for missing values and duplicates.
* **Exploratory Data Analysis (EDA):** In-depth analysis of feature distributions and their correlation with the target variable (`engagement_level`).
* **Baseline Modeling:** Compared four different models (K-Nearest Neighbors, Logistic Regression, Random Forest, and Gradient Boosting) to establish a performance benchmark.
* **Advanced Optimization:** Used the `Optuna` library for Bayesian hyperparameter tuning to find the best possible parameters for the top-performing model (XGBoost).
* **Interactive Web App:** Deployed the final model using `Gradio` to create a simple web interface for real-time predictions.
* **Data Logging:** The Gradio app includes a function to log all user inputs and predictions to a `user_input.csv` file, allowing for future model retraining.

## 💡 Key Insights

1.  **Behavior is More Predictive Than Demographics:** The most significant finding from the analysis is that behavioral metrics are far more important than demographic data.
    * **Top Predictors:** `sessions_week` (correlation: 0.61) and `avg_sesion_duration` (correlation: 0.48) were the strongest indicators of engagement.
    * **Weak Predictors:** `age`, `gender`, and `location` had almost zero correlation (near 0.0) with the engagement level.

2.  **Model's "Blind Spot" is the "Medium" Player:** The final confusion matrix shows the model is excellent at identifying "Low" (95% accuracy) and "High" (90% accuracy) engagement. However, it struggles most with the "Medium" category (85% accuracy), frequently misclassifying these players as either Low or High. This suggests "Medium" engagement is an ambiguous state with behaviors that overlap both extremes.

## 📈 Final Model Performance

After comparing four baseline models, XGBoost was selected for optimization. A 50-trial hyperparameter search was conducted using Optuna.

* **Model:** Tuned `XGBoost` Classifier
* **Final Test Accuracy:** `91.67%`
* **Key Hyperparameters (found by Optuna):**
    * `n_estimators`: 1343
    * `learning_rate`: 0.016
    * `max_depth`: 8
    * `subsample`: 0.76

## 💻 Technology Stack

* **Data Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Hyperparameter Tuning:** Optuna
* **Web App Deployment:** Gradio

---

## 🚀 How to Run This Project

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repository-Name].git
cd [Your-Repository-Name]

