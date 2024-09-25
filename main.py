# app.py
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return "API Flask rodando no Heroku!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simule uma predição (substitua pela lógica do seu modelo de IA)
    resultado = {'prediction': 'simulação de resultado'}
    return jsonify(resultado)

if __name__ == '__main__':
    app.run(debug=True)
