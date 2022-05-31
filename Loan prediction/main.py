import os
import json
from flask import Flask, render_template, session, request, flash, redirect
import flask
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import StandardScaler
import sklearn.externals
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, session

################################################################################################################################3

with open("config.json", "r") as c:
    params = json.load(c)["params"]

app = Flask(
    __name__, template_folder="template", static_folder="static"
)  # instance is being created
app.secret_key = "super-secret-key"  # secre

Model = pickle.load(open(r"model.sav", "rb"))

X = pd.read_csv("X.csv")
X.drop(columns=["Unnamed: 0"], inplace=True)

print(X.shape)

######################################################################################################################################
def predict_price(
    Income,
    Age,
    Experience,
    marraige,
    House_Ownership,
    Car_Ownership,
    CURRENT_JOB_YRS,
    CURRENT_HOUSE_YRS,
    CITY,
    Profession,
    model,
):

    prof_index = np.where(X.columns == Profession)[0][0]
    loc_index = np.where(X.columns == CITY)[0][0]

    x = np.zeros(len(X.columns))

    x[0] = Income
    x[1] = Age  # B
    x[2] = Experience  # B
    x[3] = marraige  # B
    x[4] = House_Ownership  # B
    x[5] = Car_Ownership  # B
    x[6] = CURRENT_JOB_YRS  # B
    x[7] = CURRENT_HOUSE_YRS  # B

    if loc_index >= 0:
        x[loc_index] = 1
    if prof_index >= 0:
        x[prof_index] = 1

    x = x.reshape(1, -1)
    df = pd.DataFrame(x)
    df = df.astype(int)
    print(df)

    ans = model.predict(df)[0]
    print("ans", ans)
    return model.predict(df)[0]
#######################################################################################################################################################################################
#ADMIN LOGIN
@app.route("/", methods=["GET", "POST"])
def login():
    if "user" in session and session["user"] == params["admin_user"]:
        return render_template("index.html", params=params)

    if request.method == "POST":
        username = request.form.get("uname")
        userpass = request.form.get("pass")
        if username == params["admin_user"] and userpass == params["admin_password"]:
            # set the session variable
            session["user"] = username
            return render_template("index.html", params=params)
        if username == params["emp_user"] and userpass == params["emp_password"]:
            # set the session variable
            session["user"] = username
            return render_template("index.html", params=params)
    return render_template("login.html", params=params)


@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    if request.method == "POST":

        Income = request.form.get("Income")
        Age = request.form.get("Age")
        Experience = request.form.get("Experience")
        marraige = request.form.get("marraige")
        House_Ownership = request.form.get("House_Ownership")
        Car_Ownership = request.form.get("Car_Ownership")
        CURRENT_JOB_YRS = request.form.get("CURRENT_JOB_YRS")
        CURRENT_HOUSE_YRS = request.form.get("CURRENT_HOUSE_YRS")
        CITY = request.form.get("CITY")
        Profession = request.form.get("Profession")

        # ------------------------------------------------------------------------------------------------------------------
        prediction = predict_price(
            Income,
            Age,
            Experience,
            marraige,
            House_Ownership,
            Car_Ownership,
            CURRENT_JOB_YRS,
            CURRENT_HOUSE_YRS,
            CITY,
            Profession,
            Model,
        )
        if prediction == 0:
            a = "We can give him a loan! "

        else:
            a = " High chances, he can default the loan"

        # -------------------------------------------------------------------------------------------------------------------
        return render_template("prediction.html", prediction_text="  {}".format(a))


app.run(debug=True)
