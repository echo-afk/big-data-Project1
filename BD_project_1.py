from sklearn import datasets
import sklearn
import numpy as np 
import pandas as pd
import time

df = pd.read_csv('test_set.csv')
dataset = df.values
#print(dataset)
dataset = np.split(dataset,[2,4,5],axis=1)
a=dataset[3]
b=dataset[2]
c=dataset[1]
b=b.T[0]
a=np.hstack((c,a))
a=a.astype(int)
b=b.astype(int)


#print(a)
#print(b)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(a, b, test_size=0.3) 
#print(X_train)
#print(Y_train)


from sklearn import tree

ttemp=time.time()
dt = tree.DecisionTreeClassifier(max_depth=13) 
dt.fit(X_train, Y_train)
rate1=dt.score(X_train, Y_train)
#print(rate1)
t1=time.time()-ttemp
#print(t1,"s")

from sklearn.neighbors import KNeighborsClassifier

ttemp=time.time()
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train, Y_train)
rate2=knn.score(X_train, Y_train)
#print(rate2)
t2=time.time()-ttemp
#print(t2,"s")

from sklearn.svm import SVC

ttemp=time.time()
svc = SVC(kernel ='rbf',gamma='auto') 
svc.fit(X_train, Y_train)
rate3=svc.score(X_train, Y_train)
#print(rate3)
t3=time.time()-ttemp
#print(t3,"s")



df1 = pd.read_csv('new_data.csv')
dataset = df1.values
dataset = np.split(dataset,[2,4,5],axis=1)
a=dataset[3]
b=dataset[2]
c=dataset[1]
b=b.T[0]
a=np.hstack((c,a))
a=a.astype(int)
b=b.astype(int)


ans_dt = dt.predict(a)
#print(ans_dt)
ans_knn = knn.predict(a)
#print(ans_knn)
ans_svc = svc.predict(a)
#print(ans_svc)

er=0
for i in range(0,len(b)):
    if b[i]!=ans_dt[i]:
        er=er+1
        
acc1=1-er/len(b)
print("Accuracy for DT:",acc1)
print("Time Consumed for DT:",t1)

er=0
for i in range(0,len(b)):
    if b[i]!=ans_knn[i]:
        er=er+1
        
acc2=1-er/len(b)
print("Accuracy for KNN:",acc2)
print("Time Consumed for KNN:",t2)

er=0
for i in range(0,len(b)):
    if b[i]!=ans_svc[i]:
        er=er+1
        
acc3=1-er/len(b)
print("Accuracy for SVC:",acc3)
print("Time Consumed for SVC:",t3)
