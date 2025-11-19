# Petrotrade-ETL-payment-status-prediction
Built an end-to-end automated pipeline for Petrotrade payment prediction. SQL CTE sampling and feature selection in PostgreSQL, Python preprocessing and Random Forest model, Streamlit app for single/batch predictions, Airflow DAG for automated runs, and a live Power BI dashboard connected to PostgreSQL.
Project Description – Petrotrade ETL & Payment Status Prediction (End-to-End Automated Pipeline)

In this project, I worked with a real dataset from Petrotrade and built a fully automated end-to-end pipeline that combines ETL processing, machine learning prediction, data warehousing, and dashboard reporting.

1. Data Extraction & Sampling (PostgreSQL)

Loaded the original Petrotrade dataset into PostgreSQL.

Used CTE queries to extract balanced monthly samples to ensure fair representation across all periods.

Performed SQL-based feature selection to reduce noise and keep only impactful variables.

2. Data Processing & Model Training (Python)

Pulled the sampled data from PostgreSQL into Python.

Applied complete data cleaning, formatting correction (especially inconsistent dates), and preprocessing.

Trained a Random Forest classifier to predict customer payment status (on-time vs. delayed).

Evaluated, tuned, and validated the model to ensure strong performance.

3. Streamlit Prediction Interface

Built a Streamlit web app with two modes:

Single record prediction

Bulk CSV prediction with downloadable output

Integrated the full preprocessing pipeline inside the app to ensure prediction consistency.

4. Airflow DAG Automation

Created an Apache Airflow DAG that:

Automatically pulls the monthly SQL samples

Runs preprocessing + prediction using the trained ML model

Saves the final predicted output into a new PostgreSQL table

This ensures the entire workflow runs fully automated without manual intervention.

5. Power BI Dashboard

Built a dynamic Power BI dashboard that connects directly to the PostgreSQL prediction table.

Visualized payment trends, predicted delays, customer risk segments, and overall financial insights.

Because the data is updated via Airflow, the dashboard is always live and up to date.

Final Outcome

The whole system—from raw Petrotrade data → SQL sampling → Python model → automated predictions → dashboard—is now fully automated, scalable, and ready for real business use.
