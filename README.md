# Credit Card Anomaly Detection
This project is an interactive Streamlit dashboard for analyzing and detecting fraudulent transactions in credit card datasets. It combines exploratory data analysis (EDA), model training, model comparison, and a fraud prediction playground into one user-friendly app.

## Project Overview
Credit card fraud is a major challenge for financial institutions and consumers. Detecting fraudulent transactions accurately is crucial for minimizing economic losses.

This dashboard provides:

- An overview of the dataset and class imbalance (fraud vs non-fraud).
- EDA tools to explore distributions and correlations between features.
- Multiple machine learning models to detect fraud:
    Random Forest
    Logistic Regression
    Gaussian Mixture Model (GMM)
- Model comparison with precision, recall, F1, and ROC AUC metrics.
- An interactive prediction playground, where users can input feature values and test fraud detection models in real-time.
- Dynamic dataset upload support – users can upload their own credit card transaction datasets.
  
## Dataset
The app supports dynamic CSV upload. By default, the dataset used in this project is the Credit Card Fraud Detection dataset, which includes the following features:

- Time: The time elapsed between this transaction and the first transaction in the dataset.
- V1 to V28: The result of a PCA transformation to protect sensitive information about the customers.
- Amount: The transaction amount.
- Class: The label for the transaction, where 0 indicates a legitimate transaction and 1 indicates a fraudulent transaction.

You can download this dataset from here
- https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud 

## Project Structure
- Overview (Tab 1): Dataset statistics and class distribution (donut chart).
- EDA (Tab 2): Feature histograms (fraud vs non-fraud) and correlation heatmap.
- Model Training (Tab 3): Train a Random Forest baseline model with adjustable hyperparameters.
- Model Comparison (Tab 4): Compare Random Forest, Logistic Regression, and GMM using test set metrics.
- Prediction (Tab 5): Input feature values manually and predict fraud probability with chosen model.

## Models Used 
- Random Forest (baseline supervised model)
- Logistic Regression (linear supervised model)
- Gaussian Mixture Model (GMM) (unsupervised anomaly detection model)

## Dependencies
- Python 3.8+
- streamlit
- numpy
- pandas
- scikit-learn
- seaborn
- matplotlib

Install them via:

        pip install -r requirements.txt

## Running the app
- Clone the repository
- Install dependencies
- Run the app
- Upload your dataset or use the default dataset to explore fraud detection

  
## Conclusion
This project demonstrates how machine learning can be applied to real-world fraud detection. By combining EDA, multiple models, and interactive prediction tools, the dashboard provides a practical way to:

- Understand dataset characteristics
- Compare different ML approaches
- Experiment with fraud detection in real-time
