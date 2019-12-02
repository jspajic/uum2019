import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

tableHeaders = ["Vlaga","Vjetar","O3","NO2","at. pr.","sun. zr.", "temp"]

podaci = pd.read_csv("meteo.csv",names = tableHeaders)

x_train, x_test, y_train, y_test = train_test_split(podaci[tableHeaders], podaci[["temp"]])

#print(y_test.head())x`

y_train_bin = y_train > 20
y_test_bin = y_test > 20

# knc = KNeighborsClassifier(n_neighbors=5)
# knc.fit(x_train,y_train_bin.values.ravel())
#
# print(knc.score(x_test, y_test_bin)) # vrijednost


maxValue = 0
best_i = 0

# 0.9522746993925871 za 100


for i in range(1,20):
    knc = KNeighborsClassifier(n_neighbors=i)
    knc.fit(x_train, y_train_bin.values.ravel())

    if(knc.score(x_test, y_test_bin) > maxValue):
        max = knc.score(x_test, y_test_bin)
        best_i = i

print("Najbolji k=", best_i)
print("Max value=", max)