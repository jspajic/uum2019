import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

tableHeaders = ["Vlaga", "Vjetar", "O3", "NO2", "at. pr.", "sun. zr.", "Temperatura"]

data = pd.read_csv("meteo.csv", names=tableHeaders)

print(data[['Temperatura']].min())
print(data[['Temperatura']].max())


# Napraviti funkciju koja ce odrediti 4 klase na sljedeci nacin

# 1. klasa <= 20C
# 2. klasa 20-25C
# 3. klasa 25-30C
# 4. klasa >=30C


def odrediKlase(skupPodataka):
    # Stvoriti novi niz kroz for petlju sa ovim klasama
    klase = []

    for temp in skupPodataka:
        if temp <= 20:
            klase.append(1)
        elif (temp > 20 and temp < 25):
            klase.append(2)
        elif (temp > 25 and temp < 30):
            klase.append(3)
        else:
            klase.append(4)
    return klase


# x -> ulazni, y -> podatci prema kojima vrednujemo

x_train, x_test, y_train, y_test = train_test_split(data[["Vlaga", "Vjetar", "O3", "NO2", "at. pr.", "sun. zr."]],
                                                    data[["Temperatura"]])

y_train_cls = odrediKlase(y_train['Temperatura'])
y_test_cls = odrediKlase(y_test['Temperatura'])

# print(data.shape)
# print(X_train.shape)

# knc = KNeighborsClassifier(n_neighbors=7)
# knc.fit(x_train, y_train_cls)
#
# preciznost = knc.score(x_test, y_test_cls)
#
# print("Preciznost iznosi: ", preciznost * 100)


# Umjesto scorea
#
# predikcije = knc.predict(x_test)
# rezultati = predikcije == y_test_cls
#
# print(pd.DataFrame(rezultati).mean)

# dtc = DecisionTreeClassifier(max_depth=751)
#
# dtc.fit(x_train, y_train_cls)
#
# dtc_preciznost = dtc.score(x_test, y_test_cls)
#
# print("Preciznost stabla iznosi: ", dtc_preciznost * 100)


rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(x_train, y_train_cls)

rfc_preciznost = rfc.score(x_test, y_test_cls)

print("Preciznost sume: ", rfc_preciznost * 100)
