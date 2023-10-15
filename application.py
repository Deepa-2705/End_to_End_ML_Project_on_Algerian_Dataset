# import jasonify so that we can return our result in form of json.
# import render_tempplate : it will help in finding out the url of the html file
from flask import Flask,request,jsonify,render_template
import pickle 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#we changed file name from app to applicaiton, there is a reason for this which we will get to know during deployment 
application = Flask(__name__)
app=application

# web application should interact with the ridge.pkl and scaler.pkl
# for this, import ridge.pkl and scaler.pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

# now create the home page of web application
@app.route("/")
def index():
    return render_template('index.html')
    # render_template will find this html file in the directory and specially in a template folder.

# Now run the file in the terminal write 'python application.py'
# Now observe we have given host address as 0.0.0.0 So, it will map to local ip address to any machine you are working 
# Suppose you are running this app in your local machine then also we will 0.0.0.0 In short it will get mapped to our local ip address and obviously local ip address is not publically available.
# The system address where this app is exactly running is 172.18.0.27:5000 So if wee have to access this we have to use the url of this app(from top)
# copy the url and paste it in new tab with port '/5000' as default port no of flask is 5000 here.We can also change the port no.
# 'https://blue-librarian-oanqq.pwskills.app:5000/' write this in new tab. Now we can see the content of index.html on the new tab

# Now for predictions we will create same app.route
@app.route("/predictdata",methods=['GET','POST']) # get: retriving ,post: sending some query on our server to execute
def predict_datapoints():
    # if request is get then interact with ridge and do prediction
    if request.method=="POST":
        # read all the input first here for the prediction
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])  # result will be list so return the first item

    else:  # if request is get then show this page 
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")

# Project is working fine so now do the deployment 
# Close the project clear screen 
# write ls -a  # when we open a lask file then there is a bydefault repository is present
# write git remote -v it shows that the project is present at the lab repository 
# remove it from its origin and keep it our repository
# so write git remote rm origin
# git remote -v it shows nothing is there in the origin
# Go to github and create a repository 
# follow below steps:
# git init
# git add .
# git status
# git commit -m "First commit"
# git config --global user.email "deepagupta35795@gmail.com"
# git config --global user.name "Deepa-2705"
# git commit -m "First commit"
# git branch -M main
# git remote add origin https://github.com/Deepa-2705/End_to_End_ML_Project_on_Algerian_Dataset.git
# git push -u origin main

# In linux we create a python environment,for creating that we need this python.config file
# Aws configuration will be looking for application that why we have rename our file from app to application
# write in terminal git add .
# git commit -m "Beanstalk configuration"
# git push -u origin main