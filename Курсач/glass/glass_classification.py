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
from imblearn.over_sampling import SMOTE

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Создание директории для сохранения графиков
if not os.path.exists('plots'):
    os.makedirs('plots')

# Загрузка датасета
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
glass = pd.read_csv(url, names=columns)

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

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Стандартизация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Функция для оценки и визуализации модели
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    # Обучение модели
    model.fit(X_train_scaled, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} - Точность на тестовой выборке: {accuracy:.4f}")
    
    # Отчет о классификации
    print(f"\n{model_name} - Отчет о классификации:")
    print(classification_report(y_test, y_pred))
    
    # Матрица ошибок
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
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
    y_test_bin = label_binarize(y_test, classes=classes)
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
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
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
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        avg_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])
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
grid_search_lr.fit(X_train_scaled, y_train)

print("\nЛучшие параметры для Логистической регрессии:")
print(grid_search_lr.best_params_)
print(f"Лучшая точность при кросс-валидации: {grid_search_lr.best_score_:.4f}")

# Обучение модели с лучшими параметрами
best_lr = grid_search_lr.best_estimator_
lr_results = evaluate_model(best_lr, "Логистическая регрессия", X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

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
grid_search_rf.fit(X_train_scaled, y_train)

print("\nЛучшие параметры для Random Forest:")
print(grid_search_rf.best_params_)
print(f"Лучшая точность при кросс-валидации: {grid_search_rf.best_score_:.4f}")

# Обучение модели с лучшими параметрами
best_rf = grid_search_rf.best_estimator_
rf_results = evaluate_model(best_rf, "Случайный лес", X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

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
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
print("\nВеса классов (для информации):")
print(class_weight_dict)

# Обработка несбалансированности классов с помощью SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
print(f"Размер обучающей выборки после SMOTE: {X_train_resampled.shape}")
print(f"Распределение классов после SMOTE:\n{pd.Series(y_train_resampled).value_counts()}")

# Создание и обучение модели нейронной сети с оптимальными параметрами
nn_model = MLPClassifier(random_state=42, verbose=False)
grid_search_nn = GridSearchCV(nn_model, param_grid_nn, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_nn.fit(X_train_resampled, y_train_resampled)

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
X_train_nn = power_transformer.fit_transform(X_train_resampled)
X_test_nn = power_transformer.transform(X_test)

# Обучение и оценка модели
best_nn.fit(X_train_nn, y_train_resampled)
nn_results = evaluate_model(best_nn, "Нейронная сеть", X_train, X_test, y_train_resampled, y_test, X_train_nn, X_test_nn)

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

# Сравнение моделей
print("\n" + "="*50)
print("СРАВНЕНИЕ МОДЕЛЕЙ")
print("="*50)

models_comparison = {
    'Логистическая регрессия': lr_results['accuracy'],
    'Случайный лес': rf_results['accuracy'],
    'Нейронная сеть': nn_results['accuracy']
}

print("\nСравнение точности моделей на тестовой выборке:")
for model_name, accuracy in models_comparison.items():
    print(f"{model_name}: {accuracy:.4f}")

# Визуализация сравнения моделей
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

# Создание отчета о классификации в формате Markdown
with open('glass_classification_report.md', 'w', encoding='utf-8') as f:
    f.write("# Отчет о классификации стекла\n\n")
    
    f.write("## Информация о датасете\n")
    f.write(f"* Размер датасета: {glass.shape[0]} образцов, {glass.shape[1]-2} параметров\n")
    f.write(f"* Количество классов: {glass['Type'].nunique()}\n")
    f.write("* Распределение классов:\n")
    for class_type, count in glass['Type'].value_counts().items():
        f.write(f"  - Класс {class_type} ({glass_types[class_type]}): {count} образцов\n")
    
    # Логистическая регрессия
    f.write("\n## 1. Логистическая регрессия\n")
    f.write(f"### Гиперпараметры\n")
    f.write(f"* Лучшие параметры: {grid_search_lr.best_params_}\n")
    f.write(f"* Лучшая точность при кросс-валидации: {grid_search_lr.best_score_:.4f}\n\n")
    
    f.write("### Результаты классификации\n")
    f.write(f"* Точность на тестовой выборке: {lr_results['accuracy']:.4f}\n\n")
    
    f.write("#### Отчет о классификации\n")
    f.write(format_classification_report_as_list(y_test, lr_results['y_pred']))
    f.write("\n")
    
    f.write("#### Матрица ошибок\n")
    f.write("![Матрица ошибок логистической регрессии](plots/логистическая_регрессия_confusion_matrix.png)\n\n")
    
    f.write("#### ROC кривые\n")
    f.write("![ROC кривые логистической регрессии](plots/логистическая_регрессия_roc_curves.png)\n\n")
    
    f.write("#### Precision-Recall кривые\n")
    f.write("![PR кривые логистической регрессии](plots/логистическая_регрессия_pr_curves.png)\n\n")
    
    f.write("#### Коэффициенты модели\n")
    f.write("Коэффициенты логистической регрессии показывают влияние каждого признака на вероятность принадлежности к определенному классу:\n\n")
    f.write("![Коэффициенты логистической регрессии](plots/logistic_regression_coefficients.png)\n\n")
    
    # Случайный лес
    f.write("\n## 2. Случайный лес\n")
    f.write(f"### Гиперпараметры\n")
    f.write(f"* Лучшие параметры: {grid_search_rf.best_params_}\n")
    f.write(f"* Лучшая точность при кросс-валидации: {grid_search_rf.best_score_:.4f}\n\n")
    
    f.write("### Результаты классификации\n")
    f.write(f"* Точность на тестовой выборке: {rf_results['accuracy']:.4f}\n\n")
    
    f.write("#### Отчет о классификации\n")
    f.write(format_classification_report_as_list(y_test, rf_results['y_pred']))
    f.write("\n")
    
    f.write("#### Матрица ошибок\n")
    f.write("![Матрица ошибок случайного леса](plots/случайный_лес_confusion_matrix.png)\n\n")
    
    f.write("#### ROC кривые\n")
    f.write("![ROC кривые случайного леса](plots/случайный_лес_roc_curves.png)\n\n")
    
    f.write("#### Precision-Recall кривые\n")
    f.write("![PR кривые случайного леса](plots/случайный_лес_pr_curves.png)\n\n")
    
    f.write("#### Важность признаков\n")
    f.write("Важность признаков в модели случайного леса показывает, насколько каждый признак влияет на точность предсказания:\n\n")
    f.write("![Важность признаков случайного леса](plots/random_forest_feature_importance.png)\n\n")
    
    f.write("| Признак | Важность |\n")
    f.write("|---------|----------|\n")
    for _, row in feature_importance.iterrows():
        f.write(f"| {row['feature']} | {row['importance']:.4f} |\n")
    
    # Нейронная сеть
    f.write("\n## 3. Нейронная сеть\n")
    f.write(f"### Гиперпараметры\n")
    f.write(f"* Лучшие параметры: {grid_search_nn.best_params_}\n")
    f.write(f"* Лучшая точность при кросс-валидации: {grid_search_nn.best_score_:.4f}\n\n")
    
    f.write("### Результаты классификации\n")
    f.write(f"* Точность на тестовой выборке: {nn_results['accuracy']:.4f}\n\n")
    
    f.write("#### Отчет о классификации\n")
    f.write(format_classification_report_as_list(y_test, nn_results['y_pred']))
    f.write("\n")
    
    f.write("#### Матрица ошибок\n")
    f.write("![Матрица ошибок нейронной сети](plots/нейронная_сеть_confusion_matrix.png)\n\n")
    
    f.write("#### ROC кривые\n")
    f.write("![ROC кривые нейронной сети](plots/нейронная_сеть_roc_curves.png)\n\n")
    
    f.write("#### Precision-Recall кривые\n")
    f.write("![PR кривые нейронной сети](plots/нейронная_сеть_pr_curves.png)\n\n")
    
    if hasattr(best_nn, 'loss_curve_'):
        f.write("#### Кривая потерь при обучении\n")
        f.write("![Кривая потерь нейронной сети](plots/neural_network_loss_curve.png)\n\n")
    
    # Сравнение моделей
    f.write("\n## 4. Сравнение моделей\n")
    f.write("### Сравнение точности на тестовой выборке\n")
    f.write("![Сравнение моделей](plots/models_comparison.png)\n\n")
    
    f.write("| Модель | Точность |\n")
    f.write("|--------|----------|\n")
    for model_name, accuracy in models_comparison.items():
        f.write(f"| {model_name} | {accuracy:.4f} |\n")
    
    # Общие выводы
    f.write("\n## 5. Общие выводы\n")
    
    # Определение лучшей модели
    best_model = max(models_comparison.items(), key=lambda x: x[1])[0]
    best_accuracy = max(models_comparison.values())
    
    f.write(f"1. Наилучшие результаты показала модель **{best_model}** с точностью {best_accuracy:.4f} на тестовой выборке.\n")
    
    # Анализ важности признаков (на основе Random Forest)
    f.write(f"2. Наиболее важными признаками для классификации типов стекла являются: ")
    f.write(", ".join([f"**{feature}**" for feature in feature_importance['feature'].iloc[:3].values]))
    f.write(".\n")
    
    # Анализ классов
    f.write("3. Анализ матриц ошибок показывает, что некоторые классы классифицируются лучше других. ")
    f.write("Это может быть связано с неравномерным распределением классов в датасете и особенностями химического состава разных типов стекла.\n")
    
    # Сравнение моделей
    f.write("4. Сравнение моделей показывает, что:\n")
    f.write("   - Логистическая регрессия обеспечивает хорошую интерпретируемость результатов через анализ коэффициентов.\n")
    f.write("   - Случайный лес позволяет оценить важность признаков и обеспечивает высокую точность.\n")
    f.write("   - Нейронная сеть способна улавливать сложные нелинейные зависимости в данных.\n")
    
    # Рекомендации
    f.write("5. Для улучшения результатов можно рекомендовать:\n")
    f.write("   - Применение методов балансировки классов для улучшения классификации малочисленных классов.\n")
    f.write("   - Использование ансамблевых методов, комбинирующих предсказания разных моделей.\n")
    f.write("   - Дополнительный сбор данных для классов с малым количеством образцов.\n")
    
    # Практическое применение
    f.write("6. Полученные модели могут быть использованы для автоматической классификации типов стекла в криминалистике, ")
    f.write("производстве стекла и других областях, где требуется определение типа стекла на основе его химического состава.\n")

print("\nКлассификация завершена. Результаты сохранены в файле glass_classification_report.md и графики в директории plots/") 