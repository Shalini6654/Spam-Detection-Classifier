# 📧 Spam Detection Classifier

<img width="1919" height="882" alt="Screenshot 2026-01-29 014304" src="https://github.com/user-attachments/assets/fa3e8a20-8da6-4e3a-8612-81d0953e7d8d" />

You can try the projects:- https://spam-detection-classifier-byshalini.streamlit.app/
This project is an **end-to-end Machine Learning application** that classifies email/SMS messages as **Spam** or **Ham (Not Spam)**. The project follows a clean, structured pipeline from problem understanding to model deployment.

---

## 📌 Problem Statement

Spam messages cause security risks and poor user experience. The goal of this project is to build a **reliable spam detection system** that can automatically classify incoming messages using Machine Learning.

---

## 🧭 Project Workflow (Step-by-Step)

### **Step 1: Data Collection**

* A labeled dataset containing spam and ham messages is used.
* Each message is associated with a target label:

  * `0` → Ham (Not Spam)
  * `1` → Spam

---

### **Step 2: Data Cleaning & Preprocessing**

* Convert text to lowercase
* Remove punctuation and special characters
* Remove stopwords
* Normalize text to improve model learning

This ensures consistency between training and prediction phases.

---

### **Step 3: Exploratory Data Analysis (EDA)**

* Analyze class distribution (Spam vs Ham)
* Identify data imbalance
* Understand word frequency patterns

⚠️ Dataset is **imbalanced**, so accuracy alone is not a reliable metric.

---

### **Step 4: Feature Extraction**

Text data is converted into numerical form using:

* **TF-IDF Vectorization**

  * Captures word importance
  * Uses unigrams and bigrams
  * Reduces noise with `min_df` and `max_df`

---

### **Step 5: Model Training**

* **Logistic Regression** is used for classification
* Handles imbalance using:

  class_weight='balanced'
* Model is trained on vectorized text features

---

### **Step 6: Model Evaluation**

Model performance is evaluated using:

* Precision
* Recall
* F1-score

Special focus is given to **Spam Recall**, since missing spam is more harmful than misclassifying ham.

---

### **Step 7: Threshold Tuning**

Instead of default prediction:


model.predict()

A custom probability threshold is used:

if spam_probability > 0.25:
    SPAM
else:
    HAM

This improves spam detection in real-world scenarios.

---

### **Step 8: Model Serialization**

Trained artifacts are saved using **pickle**:

* `NB_Spam_model.pkl`
* `vectorizer.pkl`

These are reused during deployment without retraining.

---

### **Step 9: Deployment (Streamlit App)**

* User enters a message
* Message is preprocessed and vectorized
* Model predicts whether it is **Spam or Not Spam**
* Output is shown instantly on UI

---

## 🆘 Help & Support

If you face any issues while running or understanding this project:

* Ensure the same preprocessing is used during training and prediction
* Verify that `NB_Spam_model.pkl` and `vectorizer.pkl` are loaded from the same training run
* Check class imbalance before trusting accuracy
* Tune the probability threshold if spam messages are misclassified

For further help, feel free to raise an issue or contact the author.

---

## ⚠️ Challenges Faced

* **Imbalanced Dataset**: Spam messages were significantly fewer than ham messages, making accuracy misleading
* **Model Selection**: Naive Bayes underperformed on recall, requiring a switch to Logistic Regression
* **Threshold Selection**: Default prediction threshold caused spam misclassification
* **Consistency Issues**: Ensuring identical preprocessing during training and deployment
* **Model Serialization**: Handling pickle loading correctly in Streamlit

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Streamlit
* Pickle

---

## 📂 Project Structure

```
Spam-Detection-Classifier/
│
├── app.py
├── NB_Spam_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── spam.csv(data)
```

---

## 🚀 How to Run the Project

1. Clone the repository

   git clone https://github.com/Shalini6654/Spam-Detection-Classifier
2. Install dependencies

   pip install -r requirements.txt
3. Run the Streamlit app

   streamlit run app.py

---

## ✅ Key Learnings

* Accuracy is misleading for imbalanced datasets
* Recall is critical in spam detection
* Proper preprocessing consistency is mandatory
* Logistic Regression outperforms Naive Bayes for this task

---

## 📌 Future Improvements

* Add logging for false positives/negatives
* Improve UI
* Use advanced NLP models
* Add email header-based features

---

## 👩‍💻 Author

Shalini

---

⭐ If you find this project useful, feel free to star the repository!
