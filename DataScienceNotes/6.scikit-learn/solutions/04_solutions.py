from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
X_breast, y_breast = load_breast_cancer(return_X_y=True)
# print(X_breast, y_breast)
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, stratify=y_breast, random_state=0, test_size=0.3)

pipe = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000))
param_grid = {'sgdclassifier__loss': ['hinge', 'log'],
              'sgdclassifier__penalty': ['l2', 'l1']}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=3, n_jobs=-1)
scores = cross_validate(grid, X_breast, y_breast, scoring='balanced_accuracy', cv=3, return_train_score=True)
df_scores = pd.DataFrame(scores)
df_scores[['train_score', 'test_score']].boxplot()

grid.fit(X_breast_train, y_breast_train)
print(grid.best_params_)
