from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
import seaborn as sns

columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

fig, axes = plt.subplots(1, 1)
df = pd.read_csv('./car.data', header=None, names=columns)


enc = preprocessing.OneHotEncoder()

X = df[columns[:-1]]
y = df[columns[-1:]]

enc.fit(X)

X_encoded = enc.transform(X).toarray()
X_prepared = preprocessing.scale(X_encoded)

enc.fit(y)
pd.plotting.scatter_matrix(X_prepared, figsize=(12, 8), ax=axes, diagonal='hist')
plt.show()

y_encoded = enc.transform(y).toarray()
y_labels = np.argmax(y_encoded, axis=1)  # Преобразуем one-hot в метки классов

X_train, X_test, y_train, y_test = train_test_split(X_prepared, y_labels, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)

# Кросс-валидация
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Средняя точность при кросс-валидации: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Обучение на всем тренировочном наборе
model.fit(X_train, y_train)

# Оценка на тестовом наборе
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность на тестовом наборе: {accuracy:.4f}")
print("\nОтчет о классификации:")
print(classification_report(y_test, y_pred))






















