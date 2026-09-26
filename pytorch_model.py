import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

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

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)


class RatingPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x
    
model = RatingPredictor()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    predictions = model(X_train_tensor)      
    loss = criterion(predictions, y_train_tensor) 
    optimizer.zero_grad()                     
    loss.backward()                            
    optimizer.step()                          

    if epoch % 50 == 0:
        print(epoch, loss.item())
        
with torch.no_grad():
    test_predictions = model(X_test_tensor)

test_predictions = test_predictions.numpy().flatten()
y_test_values = y_test_tensor.numpy().flatten()

pt_mae = mean_absolute_error(y_test_values, test_predictions)
pt_r2 = r2_score(y_test_values, test_predictions)

print(pt_mae, pt_r2)