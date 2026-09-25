# IMDB Rating Prediction

Predicting IMDB movie ratings from four numeric features (release year, gross revenue, number of votes, and metascore) using scikit-learn, and comparing two regression models.

## Project Files

- `regression_model.py` cleans the data, trains a Linear Regression model and a Random Forest model on the same train/test split, and compares their accuracy

## Tools

- pandas
- scikit-learn

## How to Run

```
pip install -r requirements.txt
python regression_model.py
```

The script prints the dataset size after cleaning, followed by the MAE and R² score for each model.

## Key Findings

- After removing rows with missing values (release year, gross, votes, or metascore), 749 of the 1000 movies remained.
- Linear Regression: MAE = 0.139, R² = 0.63
- Random Forest: MAE = 0.136, R² = 0.64
- XGBoost: MAE = 0.151, R² = 0.57
- Neural Network (Keras, 300 epochs): MAE = 0.145, R² = 0.55
- Random Forest performed only slightly better than Linear Regression, which suggests the relationship between these four features and the rating is close to linear.
- XGBoost and the neural network both underperformed the simpler models. With only 599 training rows, both complex models likely struggled to find a pattern that Linear Regression and Random Forest captured just as well with far fewer parameters to tune.
- The neural network needed feature scaling (`StandardScaler`) to train at all — without it, the loss exploded into the hundreds of thousands because features like `Gross` (in the hundreds of millions) and `Released_Year` (around 2000) are on wildly different scales. It also needed many more training epochs (300 instead of 50) to converge; with too few epochs, it hadn't finished learning and produced a negative R² (worse than just guessing the average).
- The real limitation is the features, not the model: even the best model here explains at most about 64% of the variation in rating. Getting a meaningfully better result would likely require better features (genre, director, cast, runtime), not a more complex algorithm.

## What I Learned

- Fixing a hidden data issue: `Released_Year` looked numeric but was stored as text, and one row even contained a certificate value (`'PG'`) instead of a year, which had to be converted with `pd.to_numeric(..., errors="coerce")`
- Selecting specific columns with `df[["A", "B"]]` versus dropping columns with `df.drop(...)`, and why explicit selection is safer
- Splitting data into train and test sets with `train_test_split`, and why testing on unseen data matters
- The standard scikit-learn pattern: create a model, `fit` it on training data, `predict` on test data
- Evaluating regression models with MAE (average error) and R² (proportion of variance explained)
- Comparing four models (Linear Regression, Random Forest, XGBoost, and a Keras neural network) on the exact same train/test split
- Why neural networks need feature scaling (`StandardScaler`) while tree-based models don't
- Building a simple `Sequential` model in Keras with `Dense` layers, and the role of `activation="relu"` and the output layer for regression
- The difference between `model.fit` in scikit-learn (one step) and in Keras (multiple epochs), and why too few epochs leads to underfitting
- A more complex model is not always better: both XGBoost and the neural network underperformed the simpler models here, a reminder that model choice should match the size and nature of the data