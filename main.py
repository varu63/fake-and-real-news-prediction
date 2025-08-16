import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1️ Load your datasets
fake_df = pd.read_csv("Fake.csv")   # must have a 'text' column
real_df = pd.read_csv("True.csv")   # must have a 'text' column

# 2️ Add labels (0 = fake, 1 = real)
fake_df["label"] = 0
real_df["label"] = 1

# 3️ Combine into one dataframe
data = pd.concat([fake_df, real_df], ignore_index=True)

# 4️ Split into features and labels
X = data["text"]  # raw text
y = data["label"]

# 5️ Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️ Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 7️ Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 8️ Predictions
y_pred = model.predict(X_test_tfidf)

# 9️ Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

 # Assume 'model' and 'vectorizer' are already trained from the earlier code

#  New text input (can be single string or list of strings)
new_text = ["This is a breaking news article about the government."]

#  Convert text to TF-IDF vector (must use the SAME vectorizer used during training)
new_text_tfidf = vectorizer.transform(new_text)

#  Predict (0 = fake, 1 = real)
prediction = model.predict(new_text_tfidf)

#  Show result
# if prediction[0] == 0:
#     print("Prediction: FAKE")
# else:
#     print("Prediction: REAL")

# Save model
joblib.dump(model, "fake_real_model.pkl")

# Save vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")


