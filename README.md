# ML Model - Titanic Survival Prediction  
### Author:  
Arnau Muñoz Barrera  

Machine Learning project focused on predicting passenger survival on the Titanic using classification models and data preprocessing techniques.

---

## 📄 Project Report

This project explores supervised machine learning techniques to solve a binary classification problem: predicting whether a passenger survived the Titanic disaster.

The dataset includes demographic and socio-economic features such as age, gender, ticket class, fare, and family relations. The objective is to build and evaluate predictive models while applying proper data preprocessing, feature engineering, and performance validation techniques.

---

## Prerequisites  

- Python 3.9+  
- An IDE such as Visual Studio Code, IntelliJ IDEA, or Jupyter Notebook  
- pip (Python package manager)  

### Necessary libraries:
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

---

# Dataset  

Titanic dataset (commonly used in machine learning competitions such as Kaggle’s Titanic challenge).

Main features include:

- `PassengerId`  
- `Pclass`  
- `Name`  
- `Sex`  
- `Age`  
- `SibSp`  
- `Parch`  
- `Ticket`  
- `Fare`  
- `Cabin`  
- `Embarked`  
- `Survived` (target variable)  

---

# Objectives  

In this project, I aimed to:

- Understand the complete machine learning workflow from raw data to model evaluation  
- Perform exploratory data analysis (EDA)  
- Apply data preprocessing and feature engineering techniques  
- Train and compare classification models  
- Evaluate model performance using appropriate metrics  
- Improve predictive accuracy through tuning and validation strategies  

---

# Features Implemented  

### • Exploratory Data Analysis (EDA)  
Analyzed distributions, correlations, and survival patterns based on gender, passenger class, and age using visualizations and statistical summaries.

### • Data Cleaning and Preprocessing  
Handled missing values (e.g., `Age`, `Cabin`, `Embarked`), encoded categorical variables (`Sex`, `Embarked`), and removed irrelevant features such as `Name` and `Ticket` when appropriate.

### • Feature Engineering  
Created new features such as family size and passenger title extraction to improve predictive performance.

### • Categorical Encoding  
Applied label encoding and/or one-hot encoding to convert categorical variables into numerical representations.

### • Feature Scaling  
Standardized or normalized numerical features where necessary to improve model convergence.

### • Model Training  
Implemented and compared multiple classification algorithms such as:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine (optional)  

### • Model Evaluation  
Evaluated models using:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

### • Cross-Validation  
Applied k-fold cross-validation to ensure robustness and reduce overfitting.

### • Hyperparameter Tuning  
Optimized model parameters using techniques such as `GridSearchCV`.

### • Performance Comparison  
Compared multiple models to select the best-performing one based on validation metrics.

### • Focus on Generalization  
Ensured that the final model generalizes well to unseen data rather than overfitting the training set.

### • Code Modularity and Clarity  
Organized preprocessing, modeling, and evaluation steps into clean and reusable code blocks for readability and maintainability.

---

# Model Architecture  

The final selected model is based on a supervised binary classification approach, trained to predict the probability of survival:

- `0` → Did not survive  
- `1` → Survived  

Pipeline stages:

1. Data cleaning  
2. Feature engineering  
3. Encoding & scaling  
4. Model training  
5. Evaluation & validation  

---

# Results  

The best-performing model achieved competitive accuracy on the validation dataset, demonstrating the importance of:

- Proper preprocessing  
- Feature selection  
- Balanced bias-variance tradeoff  
- Cross-validation  

---

# Setup  

## Clone the repository  

```sh
git clone https://github.com/arnaumunozbarrera/ML-Model-Titanic.git
cd ML-Model-Titanic
```

## Install dependencies  

```sh
pip install -r requirements.txt
```

If `requirements.txt` is not included, manually install:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Run the project  

If using Jupyter Notebook:

```sh
jupyter notebook
```

Open the main notebook file and execute the cells sequentially.

If using a Python script:

```sh
python main.py
```
