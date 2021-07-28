import joblib
loaded_model=joblib.load('dib_79.pkl')
pred=loaded_model.predict([[10,20,30,40,50,60,40,40]])
if pred[0]==1:
    print("Person is diabetic")
else:
    print("not diabetic")