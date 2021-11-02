from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

# Read the data
data = pd.read_csv('../data/adult_openml.csv')
# Split the label from the data which we will use for classification
y = data['class']
X = data.drop(columns=['class', 'fnlwgt', 'capitalgain', 'capitalloss'])

# Encode the class
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Define the different column
col_cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
col_ord = ['sex']
col_num = ['age', 'hoursperweek']

# Define the different pipeline
pipe_cat = OneHotEncoder(handle_unknown='ignore')
pipe_ord = OrdinalEncoder()
pipe_num = KBinsDiscretizer()
preprocessor = ColumnTransformer([('trans_cat', pipe_cat, col_cat),
                                  ('trans_ord', pipe_ord, col_ord),
                                  ('trans_num', pipe_num, col_num)])
# Define the pipeline to use
pipe = make_pipeline(preprocessor, LogisticRegression(solver='lbfgs', max_iter=1000))
param_grid = {'logisticregression__C': [0.1, 1.0, 10]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)

scores = pd.DataFrame(cross_validate(grid, X, y, scoring='balanced_accuracy', cv=3, n_jobs=-1, return_train_score=True))
# scores[['train_score', 'test_score']].boxplot(whis=10)
print(scores)