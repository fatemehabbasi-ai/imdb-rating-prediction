import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

df = pd.read_csv("imdb_top_1000.csv")

df["Gross"] = df["Gross"].str.replace(",", "")
df["Gross"] = df["Gross"].astype(float)

df["Released_Year"] = pd.to_numeric(df["Released_Year"], errors="coerce")

df_clean = df.dropna(subset=["Released_Year","Gross","No_of_Votes","Meta_score","IMDB_Rating"])

X = df_clean[["Released_Year", "Gross", "No_of_Votes", "Meta_score"]]
y = df_clean["IMDB_Rating"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = linear_model.LinearRegression()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train,y_train)
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)

print(mae)
print(r2)

print(rf_mae)
print(rf_r2)

print(xgb_mae)
print(xgb_r2)