# Beginner ML Projects 🚀

A collection of beginner-friendly machine learning projects to build core skills in data preprocessing, model training, evaluation, and basic deep learning — before moving on to more advanced work.

Each project folder includes its own dataset link, notebook/script, and a mini README with results.

---

## 📂 Projects

| # | Project | Domain | Key Skills |
|---|---------|--------|------------|
| 1 | [House Price Prediction](./01-house-price-prediction) | Regression | Feature engineering, sklearn pipelines |
| 2 | [Titanic Survival Classifier](./02-titanic-survival) | Classification | EDA, logistic regression, storytelling |
| 3 | [Iris / Wine Classification](./03-iris-wine-classification) | Classification | Clean sklearn pipeline basics |
| 4 | [Customer Churn Prediction](./04-customer-churn) | Classification | Business framing, imbalanced classes |
| 5 | [Spam Email Classifier](./05-spam-classifier) | NLP | Bag-of-words, TF-IDF, Naive Bayes |
| 6 | [MNIST Digit Recognizer](./06-mnist-digit-recognizer) | Computer Vision | First neural net, CNN intro |

---

## 🎯 Goal of This Repo

These projects are intentionally simple in scope but cover the fundamentals every ML practitioner needs:

- Loading and cleaning real-world data
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing pipelines
- Training and evaluating classic ML models (regression, classification)
- A first taste of neural networks and CNNs
- Communicating results clearly (metrics + visuals)

---

## 🛠️ Tech Stack

- Python 3.x
- `pandas`, `numpy` — data handling
- `scikit-learn` — classic ML models & pipelines
- `matplotlib`, `seaborn` — visualization
- `tensorflow` / `keras` or `pytorch` — MNIST CNN
- Jupyter Notebooks for exploration + writeups

---

## 📦 Setup

```bash
git clone https://github.com/<your-username>/beginner-ml-projects.git
cd beginner-ml-projects
pip install -r requirements.txt
```

Each project can also be run independently — see the README inside each project folder for dataset links and run instructions.

---

## 📁 Repo Structure

```
ML/
Beginner/
├── 01-house-price-prediction/
├── 02-titanic-survival/
├── 03-iris-wine-classification/
├── 04-customer-churn/
├── 05-spam-classifier/
├── 06-mnist-digit-recognizer/
├── requirements.txt
└── README.md   ← you are here
```

---

## 📊 What Each Project Covers

### 1. House Price Prediction
Predict housing prices using the Kaggle Ames/Boston housing dataset. Covers handling missing values, encoding categorical features, and comparing linear regression vs. regularized models (Ridge/Lasso).

### 2. Titanic Survival Classifier
Classic binary classification problem. Focus on EDA storytelling — understanding *why* certain features (class, sex, age) matter — before modeling with logistic regression / decision trees.

### 3. Iris / Wine Classification
A clean, minimal multi-class classification example. Great for practicing sklearn `Pipeline` and `ColumnTransformer` without messy real-world data.

### 4. Customer Churn Prediction
Business-flavored classification task. Introduces class imbalance handling (SMOTE, class weights) and evaluation beyond accuracy (precision, recall, F1, ROC-AUC).

### 5. Spam Email Classifier
Intro to NLP: text cleaning, tokenization, TF-IDF vectorization, and a Naive Bayes or logistic regression classifier.

### 6. MNIST Digit Recognizer
First neural network project. Start with a simple dense network, then build a CNN and compare performance.

---

## ✅ Results Snapshot

*(Update this table as you complete each project)*

| Project | Metric | Score |
|---------|--------|-------|
| House Price Prediction | RMSE | TBD |
| Titanic Survival | Accuracy | TBD |
| Iris/Wine Classification | Accuracy | TBD |
| Customer Churn | F1-score | TBD |
| Spam Classifier | Precision/Recall | TBD |
| MNIST CNN | Test Accuracy | TBD |

---

## 📌 Notes

- Datasets are not committed to the repo (see each project's README for download links) — add `data/` to `.gitignore`.
- Notebooks include markdown commentary explaining *why* a decision was made, not just the code.
- Contributions/suggestions welcome via issues or PRs.

---

## 📄 License

MIT License — free to use, modify, and learn from.