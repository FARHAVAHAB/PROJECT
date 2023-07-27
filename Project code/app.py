from flask import Flask,render_template,request,redirect,url_for,session
import mysql.connector
import os
# import pickle
import numpy as np
import pandas as pd	
import joblib
from flask_login import LoginManager,login_required,login_user,logout_user
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LogisticRegression
app=Flask(__name__)
conn=mysql.connector.connect(host="127.0.0.1",port="3306",user="root",password="",database="heart_disease")
cursor = conn.cursor()

@app.route('/')
def homes():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('home.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/login_validation',methods=['GET','POST'])
def login_validation():
    username=request.form.get('username')
    password=request.form.get('Password')


    cursor.execute("""SELECT * FROM `register` WHERE `username` LIKE '{}' AND `password` LIKE '{}'""".format(username,password))
    users=cursor.fetchall()
    print(users)

    if len(users)>=1:
        return render_template('home.html')
    else:
        return render_template('login.html')



@app.route('/add_user',methods=['GET','POST'])
def add_user():
    
    name=request.form.get ('name')
    email=request.form.get ('email')
    phone=request.form.get ('phone')
    username=request.form.get('username')
    Password=request.form.get('Password')

    cursor.execute("""INSERT INTO `register` (`name`,`email`,`phone`,`username`,`Password`)VALUES
                      ('{}','{}','{}','{}','{}')""".format(name,email,phone, username, Password,))
    conn. commit ()
    return render_template('login.html')

@app.route("/logout")
def logout():
    return render_template('index.html')

@app.route("/upload")
def upload():
    return render_template('upload.html')



#model=pickle.load(open('model.pkl','rb'))
@app.route('/prediction')
def prediction():
    return render_template('predict.html')




@app.route('/predict',methods=['post'])
def predict():
    from keras.models import load_model
    model = load_model("heart.h5")
  
    age=int(request.values['age'])
    sex=int(request.values['sex'])
    chest=int(request.values['chest'])
    bp=int(request.values['bp'])
    ch=int(request.values['ch'])
    sugar=int(request.values['sugar'])
    ecg=int(request.values['ecg'])
    heartrate=int(request.values['heartrate'])
    angina=int(request.values['angina'])
    rest=int(request.values['rest'])
    segment=int(request.values['segment'])
    flourosopy=int(request.values[' flourosopy'])
    Thalium=int(request.values['Thalium'])

    # print(age)
    # print(sex)
    # print(chest)
    # print(bp)
    # print(ch)
    # print(sugar)
    # print(ecg)
    # print(heartrate)
    # print(angina)
    # print(rest)
    # print(segment)
    # print(flourosopy)
    # print(Thalium)
    z = np.array([age,sex,chest,bp,ch,sugar,ecg,heartrate,angina,rest,segment,flourosopy,Thalium])
   

    # Load the dataset
    data = pd.read_csv(r'C:\Users\Farha Afsal\OneDrive\Desktop\vs code\Health Center Free Website Template - Free-CSS.com/2098_health/heart.csv')

    # Split the dataset into features and target
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    data = LogisticRegression(max_iter=1000)
    data.fit(X_train, y_train)
 

    # Build the ANN model
    model = Sequential()
    model.add(Dense(units=16, activation='relu', input_dim=13))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test accuracy: {accuracy}')
    # Predict on new data
    new_data = pd.DataFrame([z])
    new_data = sc.transform(new_data)
    Y_pred = data.predict(new_data)
    prediction = model.predict(new_data)
    print(f'Prediction: {Y_pred}')
    print(prediction>0.45)

    if(prediction>0.45):
        return render_template('animation2.html',prediction_text="TRUE")
    else:
        return render_template('animation1.html',prediction_text="NORMAL")
        



    #return render_template('animation1.html',prediction_text=y_pred)
# if_name_=='_main-':
#     app.run(port=8000)
if __name__ == '__main__':
    app.run(port=8000)
