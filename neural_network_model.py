import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("imdb_top_1000.csv")

df["Gross"] = df["Gross"].str.replace(",", "")
df["Gross"] = df["Gross"].astype(float)

df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")

df_clean = df.dropna(subset=["Released_Year","Gross","No_of_Votes","Meta_score","IMDB_Rating"])

X = df_clean[["Released_Year", "Gross", "No_of_Votes", "Meta_score"]]
y = df_clean["IMDB_Rating"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(4,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mean_absolute_error")
history = model.fit(X_train_scaled, y_train, epochs=300, verbose=1)

nn_predictions = model.predict(X_test_scaled)
nn_predictions = nn_predictions.flatten()

nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)


print(model.summary())
print(nn_mae)
print(nn_r2)
