# RealEstateAI-Solutions---Real-Estate-Market-Forecast-Model
Implementation of advanced regularization techniques in linear regression models

This Jupyter Notebook provides a comprehensive analysis and predictive modeling for real estate prices. The notebook includes data preprocessing, exploratory data analysis (EDA), outlier detection, multicollinearity analysis, feature engineering, and machine learning model training and evaluation.

## Features
- **Exploratory Data Analysis (EDA):**
    - Univariate and multivariate analysis
    - Staistics test
    - Outlier detection using IQR and Z-score methods.
    - Multicollinearity analysis using correlation matrices and Variance Inflation Factor (VIF).
- **Preprocessing:**
    - Feature scaling, encoding, and optional transformations (e.g., log transformation, polynomial features).
    - Custom preprocessing pipelines using `ColumnTransformer`.
- **Machine Learning Models:**
    - Linear Regression, Ridge, Lasso, and ElasticNet.
    - Hyperparameter tuning using Optuna.
    - Cross-validation for model evaluation.
- **Diagnostics:**
    - Residual analysis, Q-Q plots, and scale-location plots.
    - Feature importance analysis and coefficient interpretation.

## Structure
1. **Data Preprocessing:** Handles missing values, scaling, encoding, and feature transformations.
2. **Exploratory Analysis:** Identifies outliers, multicollinearity, and key predictors.
3. **Model Training:** Trains and evaluates multiple regression models with and without log transformation.
4. **Hyperparameter Tuning:** Optimizes model parameters using Optuna.
5. **Diagnostics:** Analyzes residuals, feature importance, and multicollinearity.

## Results
- The best-performing model is **Linear Regression with log-transformed target**, achieving:
    - **RMSE (Test):** ~1.31 million
    - **R² (Test):** ~0.66
- Key predictors include `airconditioning`, `hotwaterheating`, and `prefarea`.

## How to Run
1. Install the required dependencies using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
2. Open the Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
3. Run the cells sequentially to reproduce the analysis and results.

## Dependencies
See `requirements.txt` for the list of required Python packages.

## License
This project is licensed under the MIT License.
