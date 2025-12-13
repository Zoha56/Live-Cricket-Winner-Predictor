# 🏏 Live T20 Cricket Match Winner Predictor

## 📊 Project Overview

This project is a machine learning application designed to predict the live winning probability of the chasing team during the second innings of a T20 (Twenty20) cricket match. The application uses a **Logistic Regression** model, trained on ball-by-ball data, and is deployed as an interactive web dashboard using **Streamlit**.

### Key Features

  * **Real-time Prediction:** Input current match metrics (score, wickets, overs, venue) to get an immediate winning probability for the batting team.
  * **Model Performance:** Displays key metrics (Accuracy, F1-Score, AUC) achieved by the trained model.
  * **Exploratory Data Analysis (EDA):** Provides insights into the dataset used for training, including venue statistics and correlation analysis.
  * **User-Friendly Interface:** The Streamlit application is broken down into four clear navigation sections.

## ⚙️ Technologies Used

| Category | Tools/Libraries |
| :--- | :--- |
| **Language** | Python (3.8+) |
| **Core ML** | Scikit-learn (Logistic Regression, ColumnTransformer) |
| **Data Handling** | Pandas, NumPy |
| **Web App** | Streamlit |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Serialization** | Pickle, Joblib |

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1\. Prerequisites

Ensure you have Python installed on your system.

### 2\. Clone the Repository

```bash
git clone https://github.com/Zoha56/Live-Cricket-Winner-Predictor
cd live-cricket-winner-predictor
```

### 3\. Install Dependencies

Install all required Python packages using the provided `requirements.txt` (or a similar method).

```bash
pip install -r requirements.txt
```

### 4\. Project Assets (Model & Data)

For the Streamlit app to run, you must have the following files generated from your training script (`ml.py`):

  * `cricket_predictor_model.pkl` (The trained Logistic Regression model)
  * `model_preprocessor.pkl` (The Scikit-learn `ColumnTransformer`)
  * `model_metrics.pkl` (Model performance metrics)
  * **EDA Images:** The application also references several static image files for the EDA dashboard, such as `eda_chase_outcome.png`, `eda_correlation_heatmap.png`, etc.

**If these files are missing, run your ML training script (`python ml.py`) first.**

### 5\. Run the Streamlit Application

Start the interactive dashboard from your terminal:

```bash
streamlit run app.py
```

The application will automatically open in your web browser.

## 🧠 Machine Learning Model Details

The core of the prediction engine is a **Logistic Regression** model  This model was chosen for its speed, interpretability, and its ability to output direct probability scores.

### Data Structure & Feature Pipeline

The model input is derived from the live state of the **2nd innings** of a T20 match. A `ColumnTransformer`  is used to manage the feature pipeline before prediction:

| Feature Name | Type | Processing | Role |
| :--- | :--- | :--- | :--- |
| `Venue`, `Bat First`, `Bat Second` | Categorical | One-Hot Encoding | Contextual factors (pitch, team experience). |
| `Runs to Get`, `Balls Remaining`, `Innings Wickets`| Numerical | Standard Scaling | Core resources and risk metrics. |
| `Current Run Rate (CRR)`| Engineered | Standard Scaling | Run rate achieved so far. |
| `Required Run Rate (RRR)`| Engineered | Standard Scaling | Run rate needed to win. |

### Performance Metrics

The model was evaluated on a held-out test set, demonstrating robust performance for a binary classification task:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | ${ml_metrics['Accuracy']:.4f}$ |
| **Precision** | ${ml_metrics['Precision']:.4f}$ |
| **F1-Score** | ${ml_metrics['F1-Score']:.4f}$ |
| **ROC-AUC** | ${ml_metrics['ROC-AUC']:.4f}$ |

## 💻 Streamlit App Navigation

The application is structured into four main pages, accessible via the sidebar for clear user experience:

1.  **📖 Introduction:** Project background, data source, and objectives.
2.  **🔮 Live Predictor:** The user-friendly interface for match simulation and real-time probability prediction.
3.  **📈 EDA Dashboard:** Detailed data structure, feature analysis, and data visualizations.
4.  **📘 Conclusion:** Model architecture explanation and performance summary.

## 🤝 Contribution

This project was developed as part of a university data science course. Feedback and suggestions are welcome.

-----

*Developed for: **Data Science - PUCIT***
