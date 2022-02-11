import csv
import panda as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEconder
from skllearn.model_selection import train_test_split
from skllearn.metrics import accuracy_score

LowCost = pd.read_csv("./Data.csv")
print(LowCost.head)

number = LabelEncoder()
LowCost.Platform = number.fit_transform(LowCost.Platform)
LowCost.Share_Screen = number.fit_transform(LowCost.Share_Screen)
LowCost.On_Cam = number.fit_transform(LowCost.On_Cam)
LowCost.Data_Consumption = number.fit_transform(LowCost.Data_Consumption)
LowCost.Low_Cost = number.fit_transform(LowCost.Low_Cost)
print(LowCost.head)

features = ["Platform", "Share_Screen", "On_Cam", "Data_Consumption"]
target = "Low_Cost"

print(features)
print(target)

features_train, features_test, target_train, target_test = train_test_split(LowCost[features], LowCost[target], test_size = 0.20, random_state = 42)

print('\tTraining Features\n', features_train)
print('\tTesting Features\n', features_test)
print('\tTraining Target\n', target_train)
print('\tTesting Target\n', target_test)

model = GaussianNB()
model.fit(features_train, target_train)

print('\tmodel.fit', model.fit)
pred = model.predict(features_test)
accuracy = accuracy_score(target_test, pred)
print('\nModel Accuracy = ', accuracy*100,'%')


answer = model.predict([[2,2,1,1]])

if answer == 1:
    print('\n First Prediction : Low Cost')
elif answer == 0:
    print('\n First Prediction : High Cost')












