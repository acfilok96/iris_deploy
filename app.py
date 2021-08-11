from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result_page", methods=["POST"])
def about():
    if request.method == "POST":
        req_data = []

        sepal_length_ = float(request.form['sepal_length'])
        req_data.append(sepal_length_)
        sepal_width_ = float(request.form['sepal_width'])
        req_data.append(sepal_width_)
        petal_length_ = float(request.form['petal_length'])
        req_data.append(petal_length_)
        petal_width_ = float(request.form['petal_width'])
        req_data.append(petal_width_)

        
        pred_value = model.predict([np.array(req_data)])[0]
        
        flower_set = ['setosa','versicolor','virginica']

        return render_template("index.html", prediction = flower_set[pred_value])

if __name__ == "__main__":
    app.run(debug=True)