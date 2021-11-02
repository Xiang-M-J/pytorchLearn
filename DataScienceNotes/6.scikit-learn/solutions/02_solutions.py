from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

X_breast, y_breast = load_breast_cancer(return_X_y=True)
# print(X_breast, y_breast)
X_breast_train, X_breast_test, y_breast_train, y_breast_test = train_test_split(X_breast, y_breast, stratify=y_breast, random_state=0, test_size=0.3)

pipe = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000))
pipe.fit(X_breast_train, y_breast_train)
y_pred = pipe.predict(X_breast_test)
accuracy = balanced_accuracy_score(y_breast_test, y_pred)

print('Accuracy score of the {} is {:.2f}'.format(pipe.__class__.__name__, accuracy))
