# KFD (Kyasanur Forest Disease) Prediction

## Project Overview
This project focuses on developing and evaluating machine learning models to predict the presence of Kyasanur Forest Disease (KFD) based on patient symptoms and demographic information. The primary goal is to accurately classify 'Confirmed' KFD cases, addressing the challenge of class imbalance inherent in medical datasets.

## Table of Contents
1.  [Project Goal](#project-goal)
2.  [Dataset](#dataset)
3.  [Methodology](#methodology)
4.  [Models Evaluated](#models-evaluated)
5.  [Key Results and Findings](#key-results-and-findings)
6.  [Feature Analysis](#feature-analysis)
7.  [Setup and Usage](#setup-and-usage)
8.  [Dependencies](#dependencies)

## Project Goal
The main objective is to build a robust binary classification model to predict whether a patient has a 'Confirmed' case of KFD ('C'). The project frames this as a classification task where 'C' (Confirmed) is the positive class (target=1), and all other categories ('S' - Suspected, 'N' - Negative, 'PR' - Probable) are treated as the negative class (target=0).

## Dataset
*   **Source**: The project uses the `KFD_399 (1).csv` dataset.
*   **Size**: Contains 399 samples and 25 features.
*   **Target Variable**: `KFD` (Kyasanur Forest Disease status).
*   **Class Distribution**: Initial analysis revealed a significant class imbalance for the 'Confirmed' class:
    *   `PR`: 151
    *   `C`: 104 (Minority Class)
    *   `S`: 72
    *   `N`: 72

## Methodology
The project followed a structured machine learning pipeline:
1.  **Data Loading & Initial Exploration**: Loaded data from CSV, inspected its structure, columns, and initial target variable distribution.
2.  **Data Preprocessing**: 
    *   Cleaned string columns by stripping whitespace.
    *   Binarized the `KFD` target variable (`C` -> 1, others -> 0).
    *   Performed an 80/20 stratified train-test split to maintain class proportions.
    *   Created a preprocessing pipeline using `OneHotEncoder` for categorical features and `StandardScaler` for numerical scaling.
    *   Calculated `scale_pos_weight` for XGBoost to handle class imbalance.
3.  **Model Initialization**: Implemented three popular classification algorithms:
    *   Logistic Regression
    *   Support Vector Classifier (SVC)
    *   XGBoost Classifier
    All models were configured with strategies (e.g., `class_weight='balanced'`, `scale_pos_weight`) to mitigate the effects of class imbalance.
4.  **Model Evaluation**: A custom evaluation function was used to calculate and report: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and Classification Report.
5.  **Addressing Imbalance with SMOTEN**: To further combat class imbalance, SMOTEN (Synthetic Minority Over-sampling Technique for Nominal and Continuous) was integrated into the pipeline using `imblearn`.
6.  **Feature Analysis**: 
    *   **Correlation Heatmap**: Visualized feature relationships after numerical encoding.
    *   **Chi-squared Test**: Identified statistically significant features related to the target.
    *   **XGBoost Feature Importance**: Determined the most influential features according to the XGBoost model.

## Models Evaluated
*   **Logistic Regression** (with `class_weight='balanced'`) & **LR + SMOTEN**
*   **Support Vector Classifier (SVC)** (with `class_weight='balanced'`) & **SVC + SMOTEN**
*   **XGBoost Classifier** (with `scale_pos_weight`) & **XGB + SMOTEN**

## Key Results and Findings
| Model              | Accuracy | Precision | Recall | F1       | ROC_AUC  |
|:-------------------|:---------|:----------|:-------|:---------|:---------|
| LogisticRegression | 0.9625   | 0.875     | 1.0    | 0.933333 | 0.995964 |
| SVC (RBF)          | 0.9625   | 0.875     | 1.0    | 0.933333 | 1.0      |
| XGBoost            | 0.9625   | 0.875     | 1.0    | 0.933333 | 0.995157 |
| **XGB + SMOTEN**   | **0.9750** | **0.913043**| **1.0** | **0.954545** | **0.990315** |
| SVC + SMOTEN       | 0.9625   | 0.875000  | 1.0    | 0.933333 | 0.987086 |
| LR + SMOTEN        | 0.9500   | 0.840000  | 1.0    | 0.913043 | 0.994350 |

### Highlights:
*   **Consistent Perfect Recall**: All models, both with intrinsic class weighting and SMOTEN, achieved a perfect Recall of 1.0 for the positive class. This means they successfully identified every 'Confirmed' KFD case in the test set, which is crucial for medical diagnosis applications.
*   **XGBoost + SMOTEN as Best Performer**: The XGBoost Classifier combined with SMOTEN yielded the highest F1-score (0.9545) and accuracy (0.975) while maintaining perfect recall. It also had the lowest number of false positives (2), indicating superior overall balance between precision and recall.
*   **Effective Imbalance Handling**: The results demonstrate that techniques like `class_weight='balanced'`, `scale_pos_weight`, and `SMOTEN` are highly effective in handling class imbalance in this dataset, enabling models to learn the characteristics of the minority class well.

## Feature Analysis
*   **Chi-squared Top Features**: `unconsciousness_Y`, `low bp_Y`, `severe myalgia_Y`, `neck stiffness_Y`, `GPS_loc_N`, `headache_Y`, `cough_Y`, `unconsciousness_N`, `fever_N`, `bloody gum_Y`.
*   **XGBoost Feature Importance Top Features**: `unconsciousness_Y`, `GPS_loc_N`, `unconsciousness_N`, `GPS_loc_P`, `Occupation_P`, `Occupation_N`, `severe myalgia_Y`, `low bp_Y`, `joint pain_Y`, `severe myalgia_N`.

Both methods consistently highlighted features related to **unconsciousness, low blood pressure, severe myalgia, and geographical location (GPS_loc)** as highly important predictors for KFD.

## Setup and Usage
To run this project, you'll need a Python environment with the specified dependencies. 

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install Dependencies**:
    It is recommended to use a virtual environment. Install the necessary libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    # Or manually install:
    # pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
    ```

3.  **Data**: Ensure your `KFD_399 (1).csv` dataset is in the correct path or uploaded as per the notebook instructions.

4.  **Run the Notebook**: Open and run the provided Jupyter Notebook (e.g., `kfd_prediction_analysis.ipynb`) to reproduce the analysis and model evaluations.

## Dependencies
*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `xgboost`
*   `imbalanced-learn`
*   `matplotlib`
*   `seaborn`
