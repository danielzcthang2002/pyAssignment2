import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import display

# Load the data
df=pd.read_csv('diabetes.csv')

print(df.head(), df.tail(), df.info())

#Find null columns
print("\nFind null columns\n", df.isnull().sum())

#Find the number of unique values for each column to find potential categorial data
print("\nFind the number of unique values for each column to find potential categorial data\n", df.nunique())

#Count and plot the number of people who have diabetes and who doesn't
sns.countplot(data=df, x='Outcome')
have_d = df['Outcome'].value_counts()[1]
no_d=df['Outcome'].value_counts()[0]
print("\n Count and plot the number of people who have diabetes and who doesn't\n")
print("Who have Diabetes\n", have_d, "\nWho doesn't have diabetes\n", no_d)
plt.show()

high_bmi = df[df.BMI >= 30]["Outcome"]
low_bmi = df[df.BMI < 30]['Outcome']
print(high_bmi.shape[0])
print(low_bmi.shape[0])

plt.hist(df[df['Outcome'] == 0]['BMI'], bins=20, alpha=0.5, label='No Diabetes')
plt.hist(df[df['Outcome'] == 1]['BMI'], bins=20, alpha=0.5, label='Diabetes')
plt.xlabel('BMI')
plt.legend()
plt.show()

old = df[df.Age >= 40]["Outcome"]
young = df[df.Age < 40]["Outcome"]
print(old.shape[0])
print(young.shape[0])
plt.hist(df[df['Outcome'] == 0]['Age'], bins=20, alpha=0.5, label='No Diabetes')
plt.hist(df[df['Outcome'] == 1]['Age'], bins=20, alpha=0.5, label='Diabetes')
plt.xlabel('AGE')
plt.legend()
plt.show()

high_preg = df[df.Pregnancies >= 5 ]["Outcome"]
low_preg = df[df.Pregnancies < 5]["Outcome"]
print(high_preg.shape[0])
print(low_preg.shape[0])
plt.hist(df[df['Outcome'] == 0]['Pregnancies'], bins=20, alpha=0.5, label='No Diabetes')
plt.hist(df[df['Outcome'] == 1]['Pregnancies'], bins=20, alpha=0.5, label='Diabetes')
plt.xlabel('Pregnancies')
plt.legend()
plt.show()

high_b = df[df.Glucose >= 70 ]["Outcome"]
low_b = df[df.Glucose < 70]["Outcome"]
print(high_b.shape[0])
print(low_b.shape[0])
plt.hist(df[df['Outcome'] == 0]['Glucose'], bins=20, alpha=0.5, label='No Diabetes')
plt.hist(df[df['Outcome'] == 1]['Glucose'], bins=20, alpha=0.5, label='Diabetes')
plt.xlabel('Glucose')
plt.legend()
plt.show()