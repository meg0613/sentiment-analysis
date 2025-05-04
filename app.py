from flask import Flask, render_template, request
import pickle
import os
import sqlite3

# Load model and vectorizer
model_path = 'model/sentiment_model.pkl'
vectorizer_path = 'model/tfidf_vectorizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None

    if request.method == 'POST':
        review = request.form['review']
        vec_review = vectorizer.transform([review])
        pred = model.predict(vec_review)[0]
        prob = model.predict_proba(vec_review)[0].max()

        prediction = "Positive" if pred == 1 else "Negative"
        confidence = round(prob * 100, 2)
        log_prediction(review, prediction, confidence)


    return render_template('index.html', prediction=prediction, confidence=confidence)

def log_prediction(text, prediction, confidence):
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (text, prediction, confidence)
        VALUES (?, ?, ?)
    ''', (text, prediction, confidence))
    conn.commit()
    conn.close()

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    if not review.strip():
        return render_template("index.html", prediction="Please enter some text.")
    
    data = vectorizer.transform([review])
    prediction = model.predict(data)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return render_template("index.html", prediction=sentiment)
    
@app.route('/history')
def history():
    conn = sqlite3.connect('logs.db')
    cursor = conn.cursor()
    cursor.execute('SELECT text, prediction, confidence, timestamp FROM predictions ORDER BY id DESC LIMIT 10')
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', rows=rows)


if __name__ == '__main__':
    app.run(debug=True)
