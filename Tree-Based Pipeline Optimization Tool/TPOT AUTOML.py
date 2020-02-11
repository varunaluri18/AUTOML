from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

var=pd.read_csv(c'://Users/Gopi/Desktop/iris.csv')
var.data

#Train Test Split
X_train,X_test,y_train,y_test=train_test_split(iris_df.data,iris_df.target,test_size=0.3,random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#Implementing the TPOT genetic algorithim
tpot = TPOTClassifier(verbosity=2, max_time_mins=10)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.fitted_pipeline_
print(tpot.score(X_test, y_test))