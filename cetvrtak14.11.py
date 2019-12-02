import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tableHeaders = ["Vlaga", "Vjetar", "O3", "NO2", "at. pr.", "sun. zr.", "temp"]
data = pd.read_csv("meteo.csv", names=tableHeaders)

x_train, x_test, y_train, y_test = train_test_split(data[["Vlaga", "Vjetar", "O3", "NO2", "at. pr.", "sun. zr."]],
                                                    data[["temp"]], random_state=99)
#Za ovo ispod treba DecisionTreeClassifier

y_train_bin = y_train > 20
y_test_bin = y_test > 20
#
# dtc = DecisionTreeClassifier(max_depth=6)
#
# dtc.fit(x_train, y_train_bin)
#
# print(dtc.score(x_test, y_test_bin))

#RandomForestClassifier

# rfc = RandomForestClassifier(n_estimators=500)
#
# rfc.fit(x_train, y_train_bin.values.ravel())
#
# print(rfc.score(x_test, y_test_bin))