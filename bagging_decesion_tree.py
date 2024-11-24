#bagging with decision tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

#synthetic data 
x,y = make_classification(
    n_samples=1000,
    n_features=10,
    random_state=20
    )
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=20)
#initialize the Decision tree Classifier
base_estimator= DecisionTreeClassifier(random_state=20)
#INITIALIZE THE BAGGING CLASSIFIER
bagging_model=BaggingClassifier(
    estimator=base_estimator,
    n_estimators=15,# number of trees
    random_state=20
)
#train the bagging classifier
bagging_model.fit(x_train,y_train)
y_predict=bagging_model.predict(x_test)
#evaluate model
accuracy_score=accuracy_score(y_test,y_predict)
print(f"accuracy:{accuracy_score}")