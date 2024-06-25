import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris["data"], iris["target"], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
predicted = knn.predict(np.array([[5, 2.9, 1, 0.2]]))
print("Flower is : {} with : {}% chances".format(iris["target_names"][predicted], score))
