# QSAR Biodegradation Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop a machine learning classification system to predict whether a chemical compound is **Ready Biodegradable (RB)** or **Not Ready Biodegradable (NRB)** based on its molecular descriptor values.

Five classification algorithms are implemented and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbours (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

---

## 2. Dataset Description

The **QSAR Biodegradation** dataset was obtained from the UCI Machine Learning Repository (Dataset ID: 254).

The dataset contains molecular descriptor values for **1,055 chemical compounds**. It contains **41 molecular descriptor features** and one binary target variable called `experimental class`.

The two target classes are:

* **RB** – Ready Biodegradable
* **NRB** – Not Ready Biodegradable

The original dataset contains 356 RB compounds and 699 NRB compounds. It has no missing values. The molecular descriptors consist of integer and real-valued features.

### Data preprocessing

The following preprocessing steps were performed:

* The dataset was loaded from the original semicolon-separated CSV file.
* Official feature names from the UCI dataset were assigned.
* Exact duplicate rows were identified and removed.
* The target classes were encoded as:

  * RB → 1
  * NRB → 0
* The data was divided into training and testing sets using an **80:20 stratified split** with `random_state = 2026`.
* The final dataset contained **1,052 rows** after duplicate removal.
* Feature scaling using `StandardScaler` was applied to Logistic Regression, KNN, Naive Bayes and SVM.
* Decision Tree was trained without feature scaling.
* All preprocessing steps that require fitting were performed using the training data only.

### Dataset Source

UCI Machine Learning Repository:

QSAR biodegradation — Dataset ID 254

https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation

**Citation:**

Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., & Consonni, V. (2013). *QSAR biodegradation*. UCI Machine Learning Repository. https://doi.org/10.24432/C5H60M.

---

## 3. GitHub Repository

**Repository:**
https://github.com/anjalipn001/qsar-biodegradation-ml

The repository contains:

* Source code
* Saved trained models
* Test dataset
* Model evaluation results
* Streamlit application
* `requirements.txt`
* Project documentation

---

## 4. Models Used

### 4.1 Logistic Regression

Logistic Regression is used as a linear classification model. Standardization was applied to the input features before training.

### 4.2 Decision Tree Classifier

A Decision Tree classifies the compounds by recursively splitting the feature space based on decision rules. Feature scaling is not required for this model.

### 4.3 K-Nearest Neighbours (KNN)

KNN predicts the class of a compound based on the classes of its nearest training samples. Standardization was applied because KNN is distance-based.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes assumes that the continuous features follow a Gaussian distribution within each class. It provides a probabilistic classification approach.

### 4.5 Support Vector Machine (SVM)

An SVM with an RBF kernel was used to identify a suitable decision boundary between the two classes. Feature standardization was applied before classification.

---

## 5. Model Comparison

The models were evaluated on the same held-out test set using six performance metrics.

| Model               | Accuracy |    AUC | Precision | Recall | F1-score |    MCC |
| ------------------- | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression |   0.8720 | 0.9231 |    0.8056 | 0.8169 |   0.8112 | 0.7145 |
| Decision Tree       |   0.7962 | 0.7666 |    0.7059 | 0.6761 |   0.6906 | 0.5391 |
| KNN                 |   0.8294 | 0.8788 |    0.7397 | 0.7606 |   0.7500 | 0.6207 |
| Naive Bayes         |   0.6588 | 0.8549 |    0.4964 | 0.9577 |   0.6538 | 0.4603 |
| SVM                 |   0.8673 | 0.9176 |    0.8116 | 0.7887 |   0.8000 | 0.7009 |

### Observations

**Logistic Regression:**
Logistic Regression achieved the highest Accuracy, AUC, F1-score and MCC among the five models. It provided a good balance between precision and recall and performed consistently across the evaluation metrics.

**Decision Tree:**
The Decision Tree obtained the lowest AUC and relatively lower Accuracy and F1-score compared with Logistic Regression, KNN and SVM. Its performance was therefore weaker on this dataset.

**KNN:**
KNN achieved better performance than the Decision Tree and Naive Bayes in most metrics. Its Accuracy and F1-score were reasonable, although it did not perform as well as Logistic Regression and SVM.

**Naive Bayes:**
Naive Bayes achieved the highest Recall of **0.9577**, meaning that it correctly identified most of the RB compounds. However, its Precision was only **0.4964**, resulting in a lower Accuracy, F1-score and MCC. This indicates that the model produced a relatively high number of false positive predictions.

**SVM:**
SVM performed close to Logistic Regression and achieved the second-highest Accuracy, AUC and MCC. Its Precision was slightly higher than Logistic Regression, but its Recall and F1-score were slightly lower.

---

## 6. Overall Best Model

Based on the evaluation results, **Logistic Regression** is the best-performing model for this dataset.

It achieved:

* **Accuracy:** 87.20%
* **AUC:** 92.31%
* **Precision:** 80.56%
* **Recall:** 81.69%
* **F1-score:** 81.12%
* **MCC:** 71.45%

Logistic Regression provided the best overall balance across the evaluation metrics and performed slightly better than SVM on the held-out test set.

---

## 7. Streamlit Application

The project includes a Streamlit-based application for evaluating the saved classification models.

The application provides:

* CSV dataset upload
* Model selection
* Prediction using the selected saved model
* Accuracy, AUC, Precision, Recall, F1-score and MCC
* Confusion matrix
* Classification report

The uploaded test dataset is evaluated using the selected **pre-trained model**. The application does not retrain the model using the uploaded test data.

**Streamlit App:**
*To be added after deployment.*

---

## 8. Project Structure

```text
qsar-biodegradation-ml/
│
├── data/
│   └── biodeg.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
│
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── model_pipeline.py
│   ├── train_models.py
│   └── verify_saved_models.py
│
├── app.py
├── test_data.csv
├── model_results.csv
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 9. How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Train and evaluate the models:

```bash
python src/train_models.py
```

Verify the saved models:

```bash
python src/verify_saved_models.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```
