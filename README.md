# Spam Mail Prediction using Machine Learning

This project is a machine learning model built to classify emails as either "spam" or "ham" (not spam). It uses Natural Language Processing (NLP) techniques to process the email text and a **Logistic Regression** algorithm to make the predictions.

---

## Dataset

* **Source:** The model is trained on the `mail_data.csv` dataset.
* **Content:** This dataset contains 5,572 email messages.
* **Columns:**
    * `Category`: The label for the email, either 'spam' or 'ham'.
    * `Message`: The raw text content of the email.

---

## ⚙️ Project Workflow

1.  **Import Libraries:** Essential libraries such as `numpy`, `pandas`, and `scikit-learn` are imported.
2.  **Load Data:** The `mail_data.csv` file is loaded into a pandas DataFrame.
3.  **Pre-processing:** Any null values in the dataset are replaced with an empty string.
4.  **Label Encoding:** The categorical labels in the 'Category' column are converted to numerical values for the model:
    * `spam` is mapped to `0`
    * `ham` is mapped to `1`
5.  **Feature/Label Split:** The data is separated into features (`X` - the 'Message' column) and labels (`Y` - the 'Category' column).
6.  **Train-Test Split:** The dataset is divided into a training set (80% of the data) and a testing set (20%).
7.  **Feature Extraction:** The raw text data (`X_train` and `X_test`) is transformed into numerical feature vectors using `TfidfVectorizer`. This process converts the text into a matrix of TF-IDF features, removes English stop words, and converts all text to lowercase.
8.  **Model Training:** A `LogisticRegression` model is initialized and trained on the training data (`X_train_features` and `Y_train`).
9.  **Evaluation:** The model's performance is evaluated using the `accuracy_score` metric.

---

## 📊 Performance

The model demonstrates high accuracy and good generalization, as the performance on the training and test data is very similar.

* **Training Data Accuracy:** 96.70%
* **Test Data Accuracy:** 96.59%

---

## 🚀 How to Use the Predictive System

The notebook includes a section to build a predictive system that can classify a new, unseen email.

To test it:
1.  Open the notebook and run all the cells to train the model and the feature extractor.
2.  Go to the cell labeled "Building a Predictive System".
3.  Modify the `input_mail` variable and paste the email text you want to classify.

    ```python
    input_mail = ["Congratulations! You've won a $1000 prize. Click here to claim."]
    ```
4.  Run the cell. The script will:
    * Transform the input text using the fitted `feature_extraction` (TF-IDF vectorizer).
    * Use the trained `model` to make a prediction.
    * Print the result, which will be either **'Ham mail'** or **'Spam mail'**.

---

## 🛠️ Dependencies

The project relies on the following Python libraries:

* `numpy`
* `pandas`
* `scikit-learn`
    * `model_selection.train_test_split`
    * `feature_extraction.text.TfidfVectorizer`
    * `linear_model.LogisticRegression`
    * `metrics.accuracy_score`
