import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

podaci = pd.read_csv('dijamanti.csv')  # posto imamo prvi redak imena, ne trebamo ih specificirati

# prebroji jedinstvene vrijednosti
# print(podaci['cut'].value_counts())


podaci['cut_cls'] = podaci['cut'].astype('category').cat.codes
podaci['color_cls'] = podaci['color'].astype('category').cat.codes
podaci['clarity_cls'] = podaci['clarity'].astype('category').cat.codes

x_train, x_test, y_train, y_test = train_test_split(
    podaci[["carat", "depth", "table", "x", "y", "z", "cut_cls", "color_cls", "clarity_cls"]],
    podaci[["price"]])  # ako imamo random state, uvijek iste podatke vraca, inace vraca random set podataka

#linearna regresija

lr = LinearRegression()

lr.fit(x_train, y_train)

print("Prave vrijednosti(samo prvih 5): ")
print(y_test[:5])
print("Predvidene vrijednosti( samo prvih 5): ")
print(lr.predict(x_test[:5]))

from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors=3)

knr.fit(x_train, y_train)

print("Predvidene vrijednosti KNR( samo prvih 5): ")
print(knr.predict(x_test[:5]))
#probat score ako nam se da
#keras
