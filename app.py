from flask import Flask, request, render_template
import joblib

# Load trained model and vectorizer
model = joblib.load("fake_real_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""
    if request.method == "POST":
        # Get text from form
        input_text = request.form["news_text"]
        
        # Transform using vectorizer
        input_tfidf = vectorizer.transform([input_text])
        
        # Predict
        pred = model.predict(input_tfidf)[0]
        prediction = "REAL" if pred == 1 else "FAKE"
    
    return render_template("home.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
