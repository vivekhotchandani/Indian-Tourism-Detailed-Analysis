# **Indian Tourism Detailed Analysis**

## **Overview**
This project focuses on analyzing Indian tourism trends using machine learning techniques. The dataset includes various features related to tourism, and the analysis involves preprocessing, feature engineering, model training, and evaluation.

## **Dataset Collection**
The dataset consists of various attributes related to Indian tourism, including:
- Number of tourists (domestic and international)
- Revenue from tourism
- State-wise tourism statistics
- Seasonal trends in tourist visits
- Other socio-economic factors affecting tourism

The dataset has undergone preprocessing techniques such as handling missing values, normalization, and data balancing (SMOTE technique has been used to balance classes).

## **Preprocessing and Feature Engineering**
- **Handling Missing Data**: Null values were replaced using mean/mode imputation.
- **Feature Selection**: Key attributes affecting tourism were selected based on statistical correlation.
- **Normalization**: Applied Min-Max scaling to bring all numerical features to a common scale.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Used to balance class distribution in the dataset [`y_train_smote.pkl`]

## **Model Training**
The repository contains multiple Python scripts for training machine learning models:
- **Classification Models**: Logistic Regression, Random Forest, XGBoost, and Support Vector Machines (SVM).
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score.
- **Hyperparameter Tuning**: Grid Search and Randomized Search were used to optimize models.

## **Requirements**
To run the project, install dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
The file includes essential libraries such as:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## **How to Run**
1. Clone the repository:
   ```bash
   git clone https://github.com/vivekhotchandani/Indian-Tourism-Detailed-Analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the preprocessing script to clean the dataset.
4. Execute the model training scripts to train and evaluate models.

## **Results & Insights**
- **Feature Importance Analysis**: Identified key factors affecting tourism demand.
- **Predictive Modeling**: Built a model to forecast tourism trends with high accuracy.
- **Data Visualizations**: Graphical representations of tourism trends were generated.

## **Future Scope**
- Integrating deep learning models for better predictions.
- Adding external data sources such as weather, festivals, and economic indicators.
- Deploying the model using a web interface for real-time predictions.
