import sklearn,pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


dataset_iris = load_iris()
X,y = dataset_iris.data, dataset_iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=0)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# model_name="model.pkl"
# pickle.dump(gnb,open(model_name,'wb'))

# def target_name():
#     return dataset_iris.target_names

