# 📊 Bank Customer Retention Modeling

An end-to-end, production-grade Machine Learning pipeline built in Python to predict bank customer churn. This project leverages advanced tree-based ensembles to identify high-risk customers, allowing businesses to optimize retention strategies.

## 🚀 Key Features & Implementation
* **Data Synthesis:** Engineered a realistic 10,000-row banking dataset matching Kaggle's schema, incorporating a realistic 80:20 class imbalance.
* **Feature Engineering:** Automated pipeline cleaning (removing uninformative IDs), demographic One-Hot Encoding, and numerical normalization via `StandardScaler`.
* **Model Benchmarking:** Trained and evaluated **Logistic Regression**, **Random Forest**, and **XGBoost** side-by-side using robust classification metrics.
* **Hyperparameter Tuning:** Fine-tuned the dominant XGBoost model using `GridSearchCV` to maximize the **F1-Score**, neutralizing minority class bias.
* **Production Deployment:** Exported the final model and scaling transformations into production-ready serialized objects (`.pkl` files) via `Joblib`.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Libraries:** Scikit-Learn, XGBoost, Pandas, NumPy, Seaborn, Joblib
* **Environment:** Google Colab
