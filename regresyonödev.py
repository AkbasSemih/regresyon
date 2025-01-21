from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("C:\\Users\\semih\\Desktop\\siniraglarifinal\\archive.csv\\forestfires.csv")

numerical_data = data.select_dtypes(include=['float64', 'int64'])
X = numerical_data.drop(columns=['Yield'])  # Bağımsız d.
y = numerical_data['Yield']                # Bağımlı d.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=16)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}") #Ortalama kare hatası(MSE)
print(f"R² Score: {r2}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=1, color="blue", label="Tahminler")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label="Doğru Çizgi")
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Değerler")
plt.title("Gerçek vs Tahmin")
plt.legend()
plt.show()
