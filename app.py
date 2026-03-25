from pyexpat import model

from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import scale
import LinearRegression

app =Flask(__name__)

@app.route('/')
def home():
    return "Hello Flask"

@app.route('/FirstPage')
def firstPage():
    return render_template('index.html')

@app.route('/LinearRegression/',methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    if request.method == "POST":
        hours = float (request.form["hours"])
        calculateResult = LinearRegression.calculateGrade(hours)
    return render_template("LinearRegressionGrades.html", result = calculateResult)

@app.route('/LogisticRegression/', methods=["GET", "POST"])
def predict_purchase():
    result = None
    probability = None

    if request.method == "POST":
        if model is None or scale is None:
            result = "Error: El modelo no está cargado"
        else:
            try:
                # Obtener datos del formulario
                edad = float(request.form['edad'])
                ingreso = float(request.form['ingreso_mensual'])
                visitas = float(request.form['visitas_web_mes'])
                tiempo = float(request.form['tiempo_sitio_min'])
                compras = float(request.form['compras_previas'])
                descuento = float(request.form['descuento_usado'])

                # Preparar datos para el modelo
                features = np.array([[edad, ingreso, visitas, tiempo, compras, descuento]])
                features_scaled = scale.transform(features)

                # Predicción
                pred = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0][1] * 100

                result = "SÍ comprará" if pred == 1 else "NO comprará"
                probability = round(prob, 1)

            except Exception as e:
                result = f"Error en los datos: {str(e)}"

    return render_template('index.html', result=result, probability=probability)


if __name__ == '__main__':
    app.run(debug=True)