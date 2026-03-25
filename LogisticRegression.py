from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


df = pd.read_csv("logistic_regresion/dataset_regresion_logistica.csv")

# Variables
X = df[["edad", "ingreso_mensual", "visitas_web_mes", 
        "tiempo_sitio_min", "compras_previas", "descuento_usado"]]

y = df["target"]

# Crear modelo
model = LogisticRegression(max_iter=1000)

# Entrenar modelo
model.fit(X, y)

# Función de predicción
def predecir_compra(edad, ingreso, visitas, tiempo, compras, descuento):
    datos = [[edad, ingreso, visitas, tiempo, compras, descuento]]
    pred = model.predict(datos)[0]

    if pred == 1:
        return "El cliente probablemente COMPRA"
    else:
        return "El cliente probablemente NO COMPRA"


@app.route('/')
def home():
    return "Regresión Logística funcionando"

@app.route('/logisticRegression', methods=["GET", "POST"])
def logistic():
    result = None

    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso = float(request.form["ingreso"])
        visitas = float(request.form["visitas"])
        tiempo = float(request.form["tiempo"])
        compras = float(request.form["compras"])
        descuento = float(request.form["descuento"])

        result = predecir_compra(edad, ingreso, visitas, tiempo, compras, descuento)

    return render_template("logistic.html", result=result)


if __name__ == "_main_":
    app.run(debug=True)