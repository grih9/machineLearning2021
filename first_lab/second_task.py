import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split

x1a = np.random.normal(19, 4, 50)
x2a = np.random.normal(14, 4, 50)
x1b = np.random.normal(14, 3, 50)
x2b = np.random.normal(10, 3, 50)

plt.scatter(x1a, x2a, label="one class")
plt.scatter(x1b, x2b, label="minus one class")
plt.legend(loc="best")

plt.show()

one_class = [(x1a[i], x2a[i]) for i in range(50)]
minus_one_class = [(x1b[i], x2b[i]) for i in range(50)]
y_one = [1] * 50
y_minus = [-1] * 50

X = one_class + minus_one_class
y = y_one + y_minus

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = GaussianNB()
model.fit(X_train, y_train)
ypr = model.predict(X_test)
print(metrics.accuracy_score(y_test, ypr))
print(metrics.confusion_matrix(y_test, ypr, labels=[1, -1]))
m = metrics.confusion_matrix(y_test, ypr, labels=[1, -1])
fpr, tpr, threshold = metrics.roc_curve(y_test, ypr, pos_label=1)
TP = m[0][0]
FP = m[1][0]
TN = m[1][1]
FN = m[0][1]
pos = TP + FN
neg = FP + TN
coef = pos / (pos + neg)
print(TP, FP, FN, TN)
roc_auc = metrics.roc_auc_score(y_test, ypr)
plt.title('ROC')
plt.plot(fpr, tpr, label='AUC = %0.2f' % roc_auc)
plt.legend(loc='best')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()

precision, recall, thresholds = metrics.precision_recall_curve(y_test, ypr, pos_label=1)
pr_auc = metrics.auc(recall, precision)
plt.title('PR')
plt.plot(recall, precision, label='AUC = %0.2f' % pr_auc)
plt.legend(loc='best')
plt.plot([0, 1], [coef, coef],'r--')
plt.xlim([0, 1])
plt.ylim([0.2, 1])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


