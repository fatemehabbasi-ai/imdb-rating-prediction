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
- Neural Network (PyTorch, 300 epochs, same architecture): MAE = 0.150, R² = 0.53
- Random Forest performed only slightly better than Linear Regression, which suggests the relationship between these four features and the rating is close to linear.
- XGBoost and both neural networks underperformed the simpler models. With only 599 training rows, these more complex models likely struggled to find a pattern that Linear Regression and Random Forest captured just as well with far fewer parameters to tune.
- The Keras and PyTorch neural networks, built with the same architecture (4 → 16 → 8 → 1) and the same number of epochs, produced very similar results (R² of 0.55 vs 0.53). This shows the two frameworks differ mainly in coding style (declarative vs. manual), not in learning capacity.
- The neural networks needed feature scaling (`StandardScaler`) to train at all — without it, the loss exploded into the hundreds of thousands because features like `Gross` (in the hundreds of millions) and `Released_Year` (around 2000) are on wildly different scales. They also needed many more training epochs (300 instead of 50) to converge; with too few epochs, training stopped before the model had learned enough, producing a negative R² at one point (worse than just guessing the average).
- The real limitation is the features, not the model: even the best model here explains at most about 64% of the variation in rating. Getting a meaningfully better result would likely require better features (genre, director, cast, runtime), not a more complex algorithm.

## What I Learned

- Fixing a hidden data issue: `Released_Year` looked numeric but was stored as text, and one row even contained a certificate value (`'PG'`) instead of a year, which had to be converted with `pd.to_numeric(..., errors="coerce")`
- Selecting specific columns with `df[["A", "B"]]` versus dropping columns with `df.drop(...)`, and why explicit selection is safer
- Splitting data into train and test sets with `train_test_split`, and why testing on unseen data matters
- The standard scikit-learn pattern: create a model, `fit` it on training data, `predict` on test data
- Evaluating regression models with MAE (average error) and R² (proportion of variance explained)
- Comparing five models (Linear Regression, Random Forest, XGBoost, a Keras neural network, and a PyTorch neural network) on the exact same train/test split
- Why neural networks need feature scaling (`StandardScaler`) while tree-based models don't
- Building a neural network two ways: declaratively with Keras (`Sequential`, `Dense` layers, `model.fit`) and manually with PyTorch (a custom `nn.Module` class, a hand-written training loop with `zero_grad`, `backward`, and `step`)
- What happens inside `model.fit`: computing predictions, calculating loss, backpropagating gradients, and updating weights — steps that are automatic in Keras but explicit in PyTorch
- `torch.no_grad()` for inference, to skip unnecessary gradient tracking when just making predictions
- A more complex model is not always better: XGBoost and both neural networks underperformed the simpler models here, a reminder that model choice should match the size and nature of the data