import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
df=pd.read_csv(url, names=name)
print(df)

X=df.iloc[:,0:8]
y=df.iloc[:,8]
X_train , X_test, y_train, y_test = model_selection.train_test_split(X,y , test_size = 0.20 , random_state = 101)

# train the model
model = LogisticRegression()

model.fit(X_train , y_train)
print('[info] model has been trained')

# accuracy
result = model.score(X_test , y_test)
print(f'accuracy of the model is {result}')
# saving the model

joblib.dump(model , 'dib_79.pkl' )

