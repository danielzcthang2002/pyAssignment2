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
import joblib

# Load the data
df=pd.read_csv('diabetes.csv')

print(df.head(), df.tail(), df.info())
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y=df['Outcome']

print(X.head(), y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 , random_state=1)

#create model
#LR_model=LogisticRegression()
LR_model = LogisticRegression(max_iter=200, random_state=1, solver='liblinear')
#default max_iter is 100, now we try with 200, the result doesn't imporve

#fit the model
LR_model.fit(X_train,y_train)

#predict the model
y_pred_lr=LR_model.predict(X_test)
print('Misclassified examples: %d' % (y_test != y_pred_lr).sum())

cm = confusion_matrix(y_test, y_pred_lr)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')

# Set labels, title, and ticks
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.xticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])
plt.yticks(ticks=[0, 1], labels=['Class 0', 'Class 1'])

# Show the plot
plt.show()

accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(accuracy_lr)

report = classification_report(y_test, y_pred_lr)
print(report)

joblib_file = "dibetes.pkl"
joblib.dump(LR_model, joblib_file)