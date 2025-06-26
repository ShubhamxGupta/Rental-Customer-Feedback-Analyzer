# Customer Review Sentiment Analysis

## 📌 Project Overview

This project performs **sentiment analysis** and **feature extraction** on customer reviews using **NLTK** and **Transformers**. The goal is to analyze customer feedback from `train_data.csv` and `test_data.csv`, identify sentiment categories (positive, negative, neutral), extract key themes or issues using TF-IDF, and summarize the findings in a structured tabular format.

---

## 🗂️ Files

- `train_data.csv` - Training dataset with customer reviews.
- `test_data.csv` - Testing dataset with customer reviews.
- `sentiment_analysis.ipynb` - Jupyter notebook performing the entire analysis.
- `README.md` - Project documentation.

---

## 🧠 Key Steps

### 1. **Data Loading**

- Loaded the training and test datasets using **pandas**.

### 2. **Data Exploration**

- Displayed the first few rows of both datasets.
- Checked for missing values and structural info.
- Found missing values in the `'Action'` column (42 in training and 16 in testing).

### 3. **Text Preprocessing**

- Cleaned the review text by:
  - Converting to lowercase
  - Removing punctuation and special characters
  - Removing stopwords (using NLTK)

### 4. **Sentiment Analysis**

- Used **VADER** sentiment analysis from **NLTK**.
- Classified reviews as:
  - **Positive**: Compound score ≥ 0.05
  - **Negative**: Compound score ≤ -0.05
  - **Neutral**: Otherwise

### 5. **Feature Extraction**

- Combined cleaned reviews from both datasets.
- Applied **TF-IDF Vectorization** with unigrams and bigrams.
- Extracted the **Top 20** most important terms across all reviews.

### 6. **Summary Reports**

- Calculated sentiment distributions:
  - **Training**: 36% neutral, 35.5% positive, 28.5% negative
  - **Testing**: 42% positive, 38% neutral, 20% negative
- Displayed sample reviews per sentiment.
- Highlighted frequent keywords (e.g., _"long lineup"_, _"lower prices"_, _"delivery disappointed"_).

### 7. **Detailed Tabular Output**

- Merged all data and added:
  - Original review
  - Cleaned review
  - Sentiment label
  - Presence of top TF-IDF terms (as boolean columns)

---

## 📊 Results Snapshot

| Customer Review | Cleaned Review | Sentiment | has_long_lineup | has_lower_prices |
|-----------------|----------------|-----------|------------------|------------------|
| "Very long lineup but clean." | long lineup clean | negative | ✅ | ❌ |

---

## 📈 Insights & Next Steps

- **Missing Values**: Mostly in `'Action'`, which might need imputation or removal if used in future modeling.
- **Theme Insights**: Top terms reveal concerns around:
  - Wait times (`long lineup`, `slow service`)
  - Cleanliness (`clean upon`)
  - Pricing (`lower prices`)
  - Delivery experience (`delivery disappointed`)
- **Next**:
  - Use this as a foundation for **training classifiers** (e.g., Logistic Regression, BERT) on labeled sentiments.
  - Extend to **multi-class classification** or **topic modeling** for deeper insights.

---

## 🛠️ Dependencies

Make sure to install the following:

```bash
pip install pandas numpy scikit-learn nltk

