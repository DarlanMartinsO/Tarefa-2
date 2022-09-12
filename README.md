!wget https://raw.githubusercontent.com/jsansao/idl/main/cluster_data.csv
!wget https://raw.githubusercontent.com/jsansao/idl/main/cluster_label.csv
from numpy import genfromtxt
cluster_data = genfromtxt('cluster_data.csv', delimiter=',')
cluster_label = genfromtxt('cluster_label.csv', delimiter=',')
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], marker="o", c= cluster_label, s=25, edgecolor="k")
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cluster_data, cluster_label)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
print("coeficientes dos dados de treinamento {}".format(log_reg.coef_))
from sklearn.metrics import accuracy_score
pred = log_reg.predict(X_test)
pred_train = log_reg.predict(X_train)
score_treinamento = accuracy_score(pred_train,y_train)
score_teste = accuracy_score(pred,y_test)
print(" Acuracia do treinamento = ",score_treinamento)
print(" Acuracia do teste = ",score_teste)
from sklearn.svm import SVC
clf = SVC(kernel="linear", C=float("inf"))
clf.fit(X_train, y_train)
print("coeficientes: ", clf.coef_)
svc_pred_test = clf.predict(X_test)
svc_pred_train = clf.predict(X_train)
print("Acuracia para os dados de teste", accuracy_score(svc_pred_test, y_test))
print("Acuracia para os dados de treino", accuracy_score(svc_pred_train, y_train))
