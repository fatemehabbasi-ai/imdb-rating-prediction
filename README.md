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
- Random Forest performed only slightly better than Linear Regression, which suggests the relationship between these four features and the rating is close to linear.
- XGBoost, despite being a more powerful model, performed worse than both simpler models. With only 599 training rows, it likely overfit the training data instead of learning a general pattern; XGBoost usually needs more data or tuned parameters (number of trees, tree depth, learning rate) to outperform simpler models.
- The real limitation is the features, not the model: these four columns alone explain at most about 64% of the variation in rating. Getting a meaningfully better model would likely require better features (genre, director, cast, runtime), not a more complex algorithm.

## What I Learned

- Fixing a hidden data issue: `Released_Year` looked numeric but was stored as text, and one row even contained a certificate value (`'PG'`) instead of a year, which had to be converted with `pd.to_numeric(..., errors="coerce")`
- Selecting specific columns with `df[["A", "B"]]` versus dropping columns with `df.drop(...)`, and why explicit selection is safer
- Splitting data into train and test sets with `train_test_split`, and why testing on unseen data matters
- The standard scikit-learn pattern: create a model, `fit` it on training data, `predict` on test data
- Evaluating regression models with MAE (average error) and R² (proportion of variance explained)
- Comparing three models (Linear Regression, Random Forest, XGBoost) on the exact same train/test split
- A more complex model is not always better: XGBoost underperformed the simpler models here, which is a reminder that model choice should match the size and nature of the data, not just pick the "strongest" algorithm