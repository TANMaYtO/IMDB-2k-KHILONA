from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('are_bisi_model_bann_gaya.pkl')
vectorizer = joblib.load('are_bisi_tokens.pkl')

@app.route('/predict', methods= ['GET','POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    x = vectorizer.transform([text])
    pred = model.predict(x)[0]
    return jsonify({'SENTIMENT': pred})

if __name__ == '__main__':
    app.run(debug=True)