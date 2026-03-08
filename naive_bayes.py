#-------------------------------------------------------------------------
# AUTHOR: Nareh Aghakian
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
X = []
for data in dbTraining:

    outlook = data[1].strip()
    temperature = data[2].strip()
    humidity = data[3].strip()
    wind = data[4].strip()
    
    if outlook == "Sunny":
        outlook = 1
    elif outlook == "Overcast":
        outlook = 2
    else:
        outlook = 3
    if temperature == "Hot":
        temperature = 1
    elif temperature == "Mild":
        temperature = 2
    else:
        temperature = 3
    if humidity == "High":
        humidity = 1
    else:
        humidity = 2
    if wind == "Weak":
        wind = 1
    else:
        wind = 2
    X.append([outlook, temperature, humidity, wind])     
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
Y = []
for data in dbTraining:
    if data[5] == "Yes":
        Y.append(1)
    else:
        Y.append(2)
#Fitting the naive bayes to the data using smoothing
#--> add your Python code here
print(X)
clf = GaussianNB()
clf.fit(X, Y)
#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
#--> add your Python code here
print("Day Outlook Temperature Humidity Wing PlayTennis Confidence")
#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for data in dbTest:
    day = data[0]
    outlook = data[1]
    temperature = data[2]
    humidity = data[3]
    wind = data[4]
    if outlook == "Sunny":
        outlook = 1
    elif outlook == "Overcast":
        outlook = 2
    else:
        outlook = 3
    if temperature == "Hot":
        temperature = 1
    elif temperature == "Mild":
        temperature = 2
    else:
        temperature = 3
    if humidity == "High":
        humidity = 1
    else:
        humidity = 2
    if wind == "Weak":
        wind = 1
    else:
        wind = 2
    probs = clf.predict_proba([[outlook, temperature, humidity, wind]])[0]
    confidence = max(probs)
    if confidence >= 0.75:
        predicition = clf.predict([[outlook, temperature, humidity, wind]])[0]
        if predicition == 1:
            label = "Yes"
        else:
            label = "No"
        print(day, data[1], data[2], data[3], data[4], label, round(confidence, 2))


