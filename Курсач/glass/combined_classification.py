import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import os
from itertools import cycle
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Создание директории для сохранения графиков
if not os.path.exists('plots'):
    os.makedirs('plots')

# Загрузка датасета
columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
glass = pd.read_csv("glass.csv", names=columns)

# Словарь для маппинга типов стекла
glass_types = {
    1: 'building_windows_float',
    2: 'building_windows_non_float',
    3: 'vehicle_windows_float',
    4: 'vehicle_windows_non_float',
    5: 'containers',
    6: 'tableware',
    7: 'headlamps'
}

# Добавление текстового описания типа стекла
glass['glass_type'] = glass['Type'].map(glass_types)

# Удаление столбца Id, так как он не несет полезной информации
glass = glass.drop('Id', axis=1)

print("Информация о датасете:")
print(f"Размер датасета: {glass.shape}")
print(f"Количество классов: {glass['Type'].nunique()}")
print(f"Распределение классов:\n{glass['Type'].value_counts()}")

# Подготовка данных для классификации
X = glass.drop(['Type', 'glass_type'], axis=1)
y = glass['Type']

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Функция для оценки и визуализации модели
def evaluate_model(model, model_name, X, y, X_scaled):
    # Обучение модели
    model.fit(X_scaled, y)
    
    # Предсказания
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)
    
    # Оценка точности
    accuracy = accuracy_score(y, y_pred)
    print(f"\n{model_name} - Точность на всем датасете: {accuracy:.4f}")
    
    # Отчет о классификации
    print(f"\n{model_name} - Отчет о классификации:")
    print(classification_report(y, y_pred))
    
    # Матрица ошибок
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[glass_types[i] for i in sorted(glass['Type'].unique())],
                yticklabels=[glass_types[i] for i in sorted(glass['Type'].unique())])
    plt.title(f'{model_name} - Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()
    
    # ROC кривая
    classes = sorted(glass['Type'].unique())
    n_classes = len(classes)
    y_bin = label_binarize(y, classes=classes)
    y_score = y_pred_proba
    
    # ROC кривая для каждого класса
    plt.figure(figsize=(14, 10))
    lw = 2
    colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink'])
    
    # Вычисление ROC кривой и AUC для каждого класса
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i, color, class_id in zip(range(n_classes), colors, classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=f'ROC класса {class_id} ({glass_types[class_id]}) (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC кривые для многоклассовой классификации')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_roc_curves.png')
    plt.close()
    
    # Precision-Recall кривая
    plt.figure(figsize=(14, 10))
    
    # Вычисление Precision-Recall кривой и AP для каждого класса
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i, color, class_id in zip(range(n_classes), colors, classes):
        precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
        avg_precision[i] = average_precision_score(y_bin[:, i], y_score[:, i])
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label=f'PR класса {class_id} ({glass_types[class_id]}) (AP = {avg_precision[i]:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall кривые для многоклассовой классификации')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_pr_curves.png')
    plt.close()
    
    return {
        'model': model,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

# Функция для форматирования отчета о классификации в виде списка
def format_classification_report_as_list(y_true, y_pred, target_names=None):
    report = classification_report(y_true, y_pred, output_dict=True)
    
    formatted_report = "### Результаты по классам:\n\n"
    
    # Добавляем информацию по каждому классу
    for class_label in sorted([label for label in report.keys() if label not in ['accuracy', 'macro avg', 'weighted avg']]):
        class_info = report[class_label]
        class_num = int(class_label)
        if class_num in glass_types:
            class_name = f"{glass_types[class_num]}"
        else:
            class_name = f"Класс {class_label}"
        
        formatted_report += f"**{class_name}**:\n"
        formatted_report += f"    Precision (точность): {class_info['precision']:.2f}\n"
        formatted_report += f"    Recall (полнота): {class_info['recall']:.2f}\n"
        formatted_report += f"    F1-score (F-мера): {class_info['f1-score']:.2f}\n"
        formatted_report += f"    Support (количество образцов): {class_info['support']}\n\n"
    
    # Добавляем общую информацию
    formatted_report += "### Общие метрики:\n\n"
    formatted_report += f"**Accuracy (точность)**: {report['accuracy']:.2f}\n\n"
    formatted_report += f"**Macro Avg**:\n"
    formatted_report += f"    Precision: {report['macro avg']['precision']:.2f}\n"
    formatted_report += f"    Recall: {report['macro avg']['recall']:.2f}\n"
    formatted_report += f"    F1-score: {report['macro avg']['f1-score']:.2f}\n\n"
    formatted_report += f"**Weighted Avg**:\n"
    formatted_report += f"    Precision: {report['weighted avg']['precision']:.2f}\n"
    formatted_report += f"    Recall: {report['weighted avg']['recall']:.2f}\n"
    formatted_report += f"    F1-score: {report['weighted avg']['f1-score']:.2f}\n"
    
    return formatted_report

# 1. Логистическая регрессия
print("\n" + "="*50)
print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
print("="*50)

# Подбор гиперпараметров для логистической регрессии
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000, 2000]
}

grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_scaled, y)

print("\nЛучшие параметры для Логистической регрессии:")
print(grid_search_lr.best_params_)
print(f"Лучшая точность при кросс-валидации: {grid_search_lr.best_score_:.4f}")

# Обучение модели с лучшими параметрами
best_lr = grid_search_lr.best_estimator_
lr_results = evaluate_model(best_lr, "Логистическая регрессия", X, y, X_scaled)

# Коэффициенты логистической регрессии
if hasattr(best_lr, 'coef_'):
    coef_df = pd.DataFrame(
        best_lr.coef_,
        columns=X.columns,
        index=[f'Класс {i}' for i in best_lr.classes_]
    )
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(coef_df, annot=True, cmap='coolwarm', center=0)
    plt.title('Коэффициенты логистической регрессии')
    plt.tight_layout()
    plt.savefig('plots/logistic_regression_coefficients.png')
    plt.close()
    
    print("\nКоэффициенты логистической регрессии:")
    print(coef_df)

# 2. Случайный лес
print("\n" + "="*50)
print("СЛУЧАЙНЫЙ ЛЕС")
print("="*50)

# Подбор гиперпараметров для Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_scaled, y)

print("\nЛучшие параметры для Random Forest:")
print(grid_search_rf.best_params_)
print(f"Лучшая точность при кросс-валидации: {grid_search_rf.best_score_:.4f}")

# Обучение модели с лучшими параметрами
best_rf = grid_search_rf.best_estimator_
rf_results = evaluate_model(best_rf, "Случайный лес", X, y, X_scaled)

# Важность признаков для Random Forest
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Важность признаков (Случайный лес)')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.tight_layout()
plt.savefig('plots/random_forest_feature_importance.png')
plt.close()

print("\nВажность признаков (Случайный лес):")
print(feature_importance)

# 3. Нейронная сеть
print("\n" + "="*50)
print("НЕЙРОННАЯ СЕТЬ")
print("="*50)

# Поиск оптимальных параметров для нейронной сети
param_grid_nn = {
    'hidden_layer_sizes': [(100,), (50, 50), (100, 50), (100, 100)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'solver': ['adam', 'sgd'],
    'batch_size': [32, 64, 'auto'],
    'max_iter': [2000],
    'early_stopping': [True],
    'n_iter_no_change': [10],
    'validation_fraction': [0.2]
}

# Вычисление весов классов для информации
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
print("\nВеса классов (для информации):")
print(class_weight_dict)

# Создание и обучение модели нейронной сети с оптимальными параметрами
nn_model = MLPClassifier(random_state=42, verbose=False)
grid_search_nn = GridSearchCV(nn_model, param_grid_nn, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_nn.fit(X_scaled, y)

print("\nЛучшие параметры для Нейронной сети:")
print(grid_search_nn.best_params_)
print(f"Лучшая точность при кросс-валидации: {grid_search_nn.best_score_:.4f}")

# Обучение модели с лучшими параметрами
best_params = grid_search_nn.best_params_
best_nn = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    activation=best_params['activation'],
    solver=best_params['solver'],
    alpha=best_params['alpha'],
    batch_size=best_params['batch_size'],
    learning_rate=best_params['learning_rate'],
    max_iter=best_params['max_iter'],
    early_stopping=best_params['early_stopping'],
    n_iter_no_change=best_params['n_iter_no_change'],
    validation_fraction=best_params['validation_fraction'],
    random_state=42,
    verbose=False
)

# Дополнительная предобработка данных для нейронной сети
power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
X_nn = power_transformer.fit_transform(X_scaled)

# Обучение и оценка модели
best_nn.fit(X_nn, y)

# Предсказания на датасете
y_pred = best_nn.predict(X_nn)
y_pred_proba = best_nn.predict_proba(X_nn)

# Оценка точности
accuracy = accuracy_score(y, y_pred)
print(f"\nНейронная сеть - Точность на датасете: {accuracy:.4f}")

# Отчет о классификации
print(f"\nНейронная сеть - Отчет о классификации:")
print(classification_report(y, y_pred))

# Сохраняем результаты для сравнения моделей
nn_results = {
    'model': best_nn,
    'accuracy': accuracy,
    'y_pred': y_pred,
    'y_pred_proba': y_pred_proba
}

# Матрица ошибок
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[glass_types[i] for i in sorted(glass['Type'].unique())],
            yticklabels=[glass_types[i] for i in sorted(glass['Type'].unique())])
plt.title('Нейронная сеть - Матрица ошибок')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/нейронная_сеть_confusion_matrix.png')
plt.close()

# ROC кривая
classes = sorted(glass['Type'].unique())
n_classes = len(classes)
y_bin = label_binarize(y, classes=classes)
y_score = y_pred_proba

# ROC кривая для каждого класса
plt.figure(figsize=(14, 10))
lw = 2
colors = cycle(['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink'])

# Вычисление ROC кривой и AUC для каждого класса
fpr = dict()
tpr = dict()
roc_auc = dict()

for i, color, class_id in zip(range(n_classes), colors, classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=f'ROC класса {class_id} ({glass_types[class_id]}) (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Нейронная сеть - ROC кривые для многоклассовой классификации')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('plots/нейронная_сеть_roc_curves.png')
plt.close()

# Precision-Recall кривая
plt.figure(figsize=(14, 10))

# Вычисление Precision-Recall кривой и AP для каждого класса
precision = dict()
recall = dict()
avg_precision = dict()

for i, color, class_id in zip(range(n_classes), colors, classes):
    precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
    avg_precision[i] = average_precision_score(y_bin[:, i], y_score[:, i])
    plt.plot(recall[i], precision[i], color=color, lw=lw,
             label=f'PR класса {class_id} ({glass_types[class_id]}) (AP = {avg_precision[i]:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Нейронная сеть - Precision-Recall кривые для многоклассовой классификации')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig('plots/нейронная_сеть_pr_curves.png')
plt.close()

# Визуализация процесса обучения нейронной сети
if hasattr(best_nn, 'loss_curve_'):
    plt.figure(figsize=(10, 6))
    plt.plot(best_nn.loss_curve_)
    plt.title('Кривая потерь при обучении нейронной сети')
    plt.xlabel('Итерации')
    plt.ylabel('Потери')
    plt.grid(True)
    plt.savefig('plots/neural_network_loss_curve.png')
    plt.close()

# 4. Пользовательская нейронная сеть
print("\n" + "="*50)
print("ПОЛЬЗОВАТЕЛЬСКАЯ НЕЙРОННАЯ СЕТЬ")
print("="*50)

# --- Вспомогательные функции ---
def sigmoid(Z):
    Z = np.clip(Z, -500, 500)  # Ограничиваем значения Z
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(p):
    return p * (1 - p)

def softmax(Z):
    # Стабильная версия softmax
    Z = np.clip(Z, -500, 500)
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# --- Класс нейронной сети ---
class NeuralNetwork:
    def __init__(self, x, y, hidden_neurons=50, learning_rate=0.01, n_classes=6):
        self.learning_rate = learning_rate
        self.input = x
        self.y = y  # One-hot encoded
        self.n_classes = n_classes
        self.hidden_neurons = hidden_neurons
        
        n_features = x.shape[1]
        
        # Поддержка нескольких скрытых слоев
        self.is_multi_layer = isinstance(hidden_neurons, tuple)
        
        if self.is_multi_layer:
            # Инициализация весов для многослойной сети
            self.weights = []
            self.biases = []
            
            # Первый слой
            self.weights.append(np.random.randn(n_features, hidden_neurons[0]) * np.sqrt(2.0 / n_features))
            self.biases.append(np.zeros((1, hidden_neurons[0])))
            
            # Промежуточные слои
            for i in range(1, len(hidden_neurons)):
                self.weights.append(np.random.randn(hidden_neurons[i-1], hidden_neurons[i]) * 
                                   np.sqrt(2.0 / hidden_neurons[i-1]))
                self.biases.append(np.zeros((1, hidden_neurons[i])))
            
            # Выходной слой
            self.weights.append(np.random.randn(hidden_neurons[-1], n_classes) * 
                               np.sqrt(2.0 / hidden_neurons[-1]))
            self.biases.append(np.zeros((1, n_classes)))
        else:
            # Инициализация весов для однослойной сети
            self.weights1 = np.random.randn(n_features, hidden_neurons) * np.sqrt(2.0 / n_features)
            self.bias1 = np.zeros((1, hidden_neurons))
            
            self.weights2 = np.random.randn(hidden_neurons, n_classes) * np.sqrt(2.0 / hidden_neurons)
            self.bias2 = np.zeros((1, n_classes))
        
        # Для хранения истории обучения
        self.loss_history = []
        self.accuracy_history = []
        
        # Вычисление весов классов для борьбы с несбалансированностью
        self.class_weights = None
        if y is not None:
            class_counts = np.sum(y, axis=0)
            self.class_weights = np.max(class_counts) / (class_counts + 1e-5)
            self.class_weights = self.class_weights / np.sum(self.class_weights)
            print("Веса классов:", self.class_weights)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def feedforward(self, X=None):
        if X is None:
            X = self.input
        
        if self.is_multi_layer:
            # Прямой проход для многослойной сети
            self.layer_outputs = []
            self.layer_activations = []
            
            # Входной слой
            self.layer_outputs.append(np.dot(X, self.weights[0]) + self.biases[0])
            self.layer_activations.append(self.relu(self.layer_outputs[0]))
            
            # Скрытые слои
            for i in range(1, len(self.weights) - 1):
                self.layer_outputs.append(np.dot(self.layer_activations[i-1], self.weights[i]) + self.biases[i])
                self.layer_activations.append(self.relu(self.layer_outputs[i]))
            
            # Выходной слой
            output = np.dot(self.layer_activations[-1], self.weights[-1]) + self.biases[-1]
            return self.softmax(output)
        else:
            # Прямой проход для однослойной сети
            self.layer1 = np.dot(X, self.weights1) + self.bias1
            self.activation1 = self.relu(self.layer1)
            self.layer2 = np.dot(self.activation1, self.weights2) + self.bias2
            return self.softmax(self.layer2)
    
    def backprop(self, X=None, y=None):
        if X is None:
            X = self.input
        if y is None:
            y = self.y
        
        batch_size = X.shape[0]
        
        # Прямой проход
        output = self.feedforward(X)
        
        if self.is_multi_layer:
            # Обратное распространение для многослойной сети
            # Градиент выходного слоя
            d_output = output - y  # Градиент кросс-энтропии с softmax
            
            # Градиенты и обновления для всех слоев в обратном порядке
            d_weights = []
            d_biases = []
            
            # Градиент для выходного слоя
            d_weights.append(np.dot(self.layer_activations[-1].T, d_output) / batch_size)
            d_biases.append(np.sum(d_output, axis=0, keepdims=True) / batch_size)
            
            # Градиент для скрытых слоев
            d_activation = np.dot(d_output, self.weights[-1].T)
            
            for i in range(len(self.weights) - 2, -1, -1):
                d_layer = d_activation * self.relu_derivative(self.layer_outputs[i])
                
                if i == 0:
                    # Первый скрытый слой получает вход от X
                    d_weights.insert(0, np.dot(X.T, d_layer) / batch_size)
                else:
                    # Остальные слои получают вход от предыдущего слоя активации
                    d_weights.insert(0, np.dot(self.layer_activations[i-1].T, d_layer) / batch_size)
                
                d_biases.insert(0, np.sum(d_layer, axis=0, keepdims=True) / batch_size)
                
                if i > 0:  # Если это не первый слой
                    d_activation = np.dot(d_layer, self.weights[i].T)
            
            # Обновление весов и смещений
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * d_weights[i]
                self.biases[i] -= self.learning_rate * d_biases[i]
        else:
            # Обратное распространение для однослойной сети
            # Градиент выходного слоя
            d_output = output - y  # Градиент кросс-энтропии с softmax
            
            # Градиент для скрытого слоя
            d_weights2 = np.dot(self.activation1.T, d_output) / batch_size
            d_bias2 = np.sum(d_output, axis=0, keepdims=True) / batch_size
            
            d_hidden = np.dot(d_output, self.weights2.T)
            d_hidden = d_hidden * self.relu_derivative(self.layer1)
            
            # Градиент для входного слоя
            d_weights1 = np.dot(X.T, d_hidden) / batch_size
            d_bias1 = np.sum(d_hidden, axis=0, keepdims=True) / batch_size
            
            # Обновление весов и смещений
            self.weights1 -= self.learning_rate * d_weights1
            self.bias1 -= self.learning_rate * d_bias1
            self.weights2 -= self.learning_rate * d_weights2
            self.bias2 -= self.learning_rate * d_bias2
    
    def compute_loss(self, y_true, y_pred):
        # Кросс-энтропия для многоклассовой классификации с весами классов
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Применение весов классов
        if self.class_weights is not None:
            # Создаем матрицу весов той же формы, что и y_true
            weights_matrix = np.zeros_like(y_true)
            for i in range(self.n_classes):
                weights_matrix[:, i] = self.class_weights[i]
            
            # Взвешенная кросс-энтропия
            return -np.sum(weights_matrix * y_true * np.log(y_pred)) / y_true.shape[0]
        else:
            return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
    def compute_accuracy(self, y_true, y_pred):
        # Точность классификации
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        return np.mean(y_pred_classes == y_true_classes)
    
    def train(self, epochs=100, batch_size=32, verbose=True):
        n_samples = self.input.shape[0]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Перемешивание данных
            indices = np.random.permutation(n_samples)
            X_shuffled = self.input[indices]
            y_shuffled = self.y[indices]
            
            # Мини-пакетное обучение
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Прямой и обратный проход
                self.backprop(X_batch, y_batch)
            
            # Вычисление потерь и точности на всем наборе данных
            predictions = self.feedforward()
            loss = self.compute_loss(self.y, predictions)
            accuracy = self.compute_accuracy(self.y, predictions)
            
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Эпоха {epoch+1}/{epochs}, Потери: {loss:.4f}, Точность: {accuracy:.4f}")
    
    def predict(self, X):
        # Предсказание классов
        probs = self.feedforward(X)
        return np.argmax(probs, axis=1)
    
    def predict_proba(self, X):
        # Вероятности классов
        return self.feedforward(X)

# --- Оптимизация гиперпараметров для пользовательской нейронной сети ---
def optimize_custom_nn(X, y):
    print("\nОптимизация гиперпараметров для пользовательской нейронной сети...")
    best_accuracy = 0
    best_params = {}
    
    # Расширенная сетка гиперпараметров
    hidden_neurons_options = [30, 50, 100, (50, 30)]
    learning_rate_options = [0.01, 0.005, 0.001]
    
    for hidden_neurons in hidden_neurons_options:
        for lr in learning_rate_options:
            print(f"Тестирование: hidden_neurons={hidden_neurons}, learning_rate={lr}")
            
            # Создание и обучение модели
            nn = NeuralNetwork(
                X, y,
                hidden_neurons=hidden_neurons,
                learning_rate=lr,
                n_classes=y.shape[1]
            )
            
            # Обучение с ранней остановкой
            patience = 10
            best_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(100):  # Уменьшаем количество эпох для оптимизации
                # Мини-эпоха обучения
                nn.train(epochs=1, batch_size=32, verbose=False)
                
                # Проверка ранней остановки
                current_loss = nn.loss_history[-1]
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"  Ранняя остановка на эпохе {epoch+1}")
                    break
            
            # Оценка на всем датасете
            predictions = nn.predict(X)
            accuracy = np.mean(predictions == np.argmax(y, axis=1))
            print(f"  Точность: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'hidden_neurons': hidden_neurons,
                    'learning_rate': lr
                }
    
    print(f"Лучшие параметры: {best_params}, Точность: {best_accuracy:.4f}")
    return best_params

# --- Подготовка данных для пользовательской нейронной сети ---
# One-hot encoding для меток классов
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(np.array(y).reshape(-1, 1))

# Оптимизация гиперпараметров
best_params_custom = optimize_custom_nn(X_scaled, y_encoded)

# Обучение модели с оптимальными параметрами
print("\nОбучение пользовательской нейронной сети с оптимальными параметрами...")
custom_nn = NeuralNetwork(
    X_scaled, y_encoded,
    hidden_neurons=best_params_custom['hidden_neurons'],
    learning_rate=best_params_custom['learning_rate'],
    n_classes=y_encoded.shape[1]
)

# Обучение модели
custom_nn.train(epochs=500, batch_size=32, verbose=True)

# Визуализация метрик обучения
print("\nВизуализация метрик обучения...")
plt.figure(figsize=(10, 5))
plt.plot(custom_nn.loss_history, label='Потери на обучающей выборке')
plt.title(f'Пользовательская нейронная сеть - График потерь')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()
plt.grid(True)
plt.savefig('plots/custom_nn_loss_plot.png')
plt.close()

# График точности
plt.figure(figsize=(10, 5))
plt.plot(custom_nn.accuracy_history, label='Точность на обучающей выборке')
plt.title(f'Пользовательская нейронная сеть - График точности')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.grid(True)
plt.savefig('plots/custom_nn_accuracy_plot.png')
plt.close()

# Оценка модели
print("\nОценка пользовательской нейронной сети...")
y_pred_custom = custom_nn.predict(X_scaled)
y_pred_proba_custom = custom_nn.predict_proba(X_scaled)

# Оценка точности
accuracy_custom = np.mean(y_pred_custom == np.argmax(y_encoded, axis=1))
print(f"Пользовательская нейронная сеть - Точность на датасете: {accuracy_custom:.4f}")

# Отчет о классификации
print(f"\nПользовательская нейронная сеть - Отчет о классификации:")
print(classification_report(np.argmax(y_encoded, axis=1), y_pred_custom))

# Сохраняем результаты для сравнения моделей
custom_nn_results = {
    'model': custom_nn,
    'accuracy': accuracy_custom,
    'y_pred': y_pred_custom,
    'y_pred_proba': y_pred_proba_custom
}

# Матрица ошибок
plt.figure(figsize=(10, 8))
cm = confusion_matrix(np.argmax(y_encoded, axis=1), y_pred_custom)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[glass_types[i] for i in sorted(glass['Type'].unique())],
            yticklabels=[glass_types[i] for i in sorted(glass['Type'].unique())])
plt.title('Пользовательская нейронная сеть - Матрица ошибок')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/custom_nn_confusion_matrix.png')
plt.close()

# Создаем словарь для сравнения моделей
models_comparison = {
    'Логистическая регрессия': lr_results['accuracy'],
    'Случайный лес': rf_results['accuracy'],
    'Нейронная сеть': nn_results['accuracy'],
    'Пользовательская нейронная сеть': accuracy_custom
}

# Обновление визуализации сравнения моделей
plt.figure(figsize=(12, 6))
models_df = pd.DataFrame({
    'Модель': list(models_comparison.keys()),
    'Точность': list(models_comparison.values())
})
sns.barplot(x='Модель', y='Точность', data=models_df)
plt.title('Сравнение точности моделей')
plt.ylim(0.5, 1.0)  # Установка диапазона для лучшей визуализации различий
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('plots/models_comparison.png')
plt.close()

# Обновление отчета о классификации в формате Markdown
with open('glass_classification_report.md', 'a', encoding='utf-8') as f:
    # Пользовательская нейронная сеть
    f.write("\n## 4. Пользовательская нейронная сеть\n")
    f.write(f"### Гиперпараметры\n")
    f.write(f"* Лучшие параметры: {best_params_custom}\n")
    f.write(f"* Точность: {accuracy_custom:.4f}\n\n")
    
    f.write("### Результаты классификации\n")
    f.write(f"* Точность на всем датасете: {accuracy_custom:.4f}\n\n")
    
    f.write("#### Отчет о классификации\n")
    f.write(format_classification_report_as_list(np.argmax(y_encoded, axis=1), y_pred_custom))
    f.write("\n")
    
    f.write("#### Матрица ошибок\n")
    f.write("![Матрица ошибок пользовательской нейронной сети](plots/custom_nn_confusion_matrix.png)\n\n")
    
    f.write("#### График потерь при обучении\n")
    f.write("![График потерь пользовательской нейронной сети](plots/custom_nn_loss_plot.png)\n\n")
    
    f.write("#### График точности при обучении\n")
    f.write("![График точности пользовательской нейронной сети](plots/custom_nn_accuracy_plot.png)\n\n")
    
    # Обновление сравнения моделей
    f.write("\n## 5. Сравнение моделей\n")
    f.write("### Сравнение точности на всем датасете\n")
    f.write("![Сравнение моделей](plots/models_comparison.png)\n\n")
    
    f.write("| Модель | Точность |\n")
    f.write("|--------|----------|\n")
    for model_name, accuracy in models_comparison.items():
        f.write(f"| {model_name} | {accuracy:.4f} |\n")

print("\nКлассификация завершена. Результаты сохранены в файле glass_classification_report.md и графики в директории plots/") 