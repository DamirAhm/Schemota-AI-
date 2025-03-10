import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

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

# Выбор параметров для анализа
param_n = 'Na'  # Натрий
param_m = 'Mg'  # Магний

print("Информация о датасете:")
print(f"Размер датасета: {glass.shape}")
print(f"Количество классов: {glass['Type'].nunique()}")
print(f"Распределение классов:\n{glass['Type'].value_counts()}")
print("\nОписательная статистика:")
print(glass.describe())

# 1. Построить график распределения параметров. Разделить выборку по классам.
plt.figure(figsize=(14, 10))
sns.scatterplot(x=param_n, y=param_m, hue='glass_type', data=glass, palette='viridis', s=100)
plt.title(f'Распределение параметров {param_n} и {param_m} по типам стекла')
plt.xlabel(f'Содержание {param_n} (%)')
plt.ylabel(f'Содержание {param_m} (%)')
plt.grid(True)
plt.legend(title='Тип стекла')
plt.savefig('plots/1_scatter_by_class.png')
plt.show()
plt.close()

# 2. Рассчитать медиану параметра n: для выборки в целом, для каждого класса отдельно
median_n_overall = glass[param_n].median()
median_n_by_class = glass.groupby('Type')[param_n].median()

print(f"\nМедиана параметра {param_n}:")
print(f"Для всей выборки: {median_n_overall:.4f}")
print("Для каждого класса:")
for class_type, median_value in median_n_by_class.items():
    print(f"  Класс {class_type} ({glass_types[class_type]}): {median_value:.4f}")

# 3. Рассчитать медиану параметра m: для выборки в целом, для каждого класса отдельно
median_m_overall = glass[param_m].median()
median_m_by_class = glass.groupby('Type')[param_m].median()

print(f"\nМедиана параметра {param_m}:")
print(f"Для всей выборки: {median_m_overall:.4f}")
print("Для каждого класса:")
for class_type, median_value in median_m_by_class.items():
    print(f"  Класс {class_type} ({glass_types[class_type]}): {median_value:.4f}")

# 4. Построить график распределения параметров для объектов выше медианного значения в выборке и ниже
# Создаем новые признаки для обозначения положения относительно медианы
glass['above_median_n'] = glass[param_n] > median_n_overall
glass['above_median_m'] = glass[param_m] > median_m_overall

# График для параметра n
plt.figure(figsize=(14, 10))
sns.scatterplot(x=param_n, y=param_m, hue='above_median_n', 
                data=glass, palette=['red', 'blue'], s=100)
plt.axvline(x=median_n_overall, color='green', linestyle='--', label=f'Медиана {param_n}')
plt.title(f'Распределение параметров относительно медианы {param_n}')
plt.xlabel(f'Содержание {param_n} (%)')
plt.ylabel(f'Содержание {param_m} (%)')
plt.grid(True)
plt.legend(title=f'Выше медианы {param_n}', labels=['Нет', 'Да'])
plt.savefig('plots/4_scatter_by_median_n.png')
plt.show()
plt.close()

# График для параметра m
plt.figure(figsize=(14, 10))
sns.scatterplot(x=param_n, y=param_m, hue='above_median_m', 
                data=glass, palette=['purple', 'orange'], s=100)
plt.axhline(y=median_m_overall, color='green', linestyle='--', label=f'Медиана {param_m}')
plt.title(f'Распределение параметров относительно медианы {param_m}')
plt.xlabel(f'Содержание {param_n} (%)')
plt.ylabel(f'Содержание {param_m} (%)')
plt.grid(True)
plt.legend(title=f'Выше медианы {param_m}', labels=['Нет', 'Да'])
plt.savefig('plots/4_scatter_by_median_m.png')
plt.show()
plt.close()

# 5. Построить гистограмму распределения, скатерограмму и боксплот параметров для объектов выше медианного значения в выборке и ниже
# Гистограммы
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Гистограмма для параметра n
sns.histplot(data=glass, x=param_n, hue='above_median_n', 
             palette=['red', 'blue'], ax=axes[0, 0], kde=True)
axes[0, 0].set_title(f'Гистограмма распределения {param_n}')
axes[0, 0].set_xlabel(f'Содержание {param_n} (%)')
axes[0, 0].set_ylabel('Частота')
axes[0, 0].grid(True)
axes[0, 0].axvline(x=median_n_overall, color='green', linestyle='--', label=f'Медиана {param_n}')
axes[0, 0].legend(title=f'Выше медианы {param_n}', labels=['Нет', 'Да'])

# Гистограмма для параметра m
sns.histplot(data=glass, x=param_m, hue='above_median_m', 
             palette=['purple', 'orange'], ax=axes[0, 1], kde=True)
axes[0, 1].set_title(f'Гистограмма распределения {param_m}')
axes[0, 1].set_xlabel(f'Содержание {param_m} (%)')
axes[0, 1].set_ylabel('Частота')
axes[0, 1].grid(True)
axes[0, 1].axvline(x=median_m_overall, color='green', linestyle='--', label=f'Медиана {param_m}')
axes[0, 1].legend(title=f'Выше медианы {param_m}', labels=['Нет', 'Да'])

# Боксплоты
sns.boxplot(data=glass, x='above_median_n', y=param_n, ax=axes[1, 0])
axes[1, 0].set_title(f'Боксплот {param_n} относительно медианы')
axes[1, 0].set_xlabel(f'Выше медианы {param_n}')
axes[1, 0].set_ylabel(f'Содержание {param_n} (%)')
axes[1, 0].grid(True)
axes[1, 0].set_xticklabels(['Нет', 'Да'])

sns.boxplot(data=glass, x='above_median_m', y=param_m, ax=axes[1, 1])
axes[1, 1].set_title(f'Боксплот {param_m} относительно медианы')
axes[1, 1].set_xlabel(f'Выше медианы {param_m}')
axes[1, 1].set_ylabel(f'Содержание {param_m} (%)')
axes[1, 1].grid(True)
axes[1, 1].set_xticklabels(['Нет', 'Да'])

plt.tight_layout()
plt.savefig('plots/5_histograms_boxplots.png')
plt.show()
plt.close()

# Скаттерплот
plt.figure(figsize=(14, 10))
above_median_both = (glass['above_median_n'] & glass['above_median_m'])
above_median_n_only = (glass['above_median_n'] & ~glass['above_median_m'])
above_median_m_only = (~glass['above_median_n'] & glass['above_median_m'])
below_median_both = (~glass['above_median_n'] & ~glass['above_median_m'])

colors = ['red', 'blue', 'green', 'purple']
groups = [below_median_both, above_median_n_only, above_median_m_only, above_median_both]
labels = ['Ниже обеих медиан', f'Выше медианы {param_n}', f'Выше медианы {param_m}', 'Выше обеих медиан']

for color, group, label in zip(colors, groups, labels):
    plt.scatter(glass[param_n][group], glass[param_m][group], c=color, label=label, s=100, alpha=0.7)

plt.axvline(x=median_n_overall, color='black', linestyle='--')
plt.axhline(y=median_m_overall, color='black', linestyle='--')
plt.title(f'Скаттерплот {param_n} и {param_m} относительно медиан')
plt.xlabel(f'Содержание {param_n} (%)')
plt.ylabel(f'Содержание {param_m} (%)')
plt.grid(True)
plt.legend()
plt.savefig('plots/5_scatterplot_by_medians.png')
plt.show()
plt.close()

# 6. Рассчитать среднее значение и стандартное отклонение для параметров: для всей выборки и для каждого класса отдельно
mean_n_overall = glass[param_n].mean()
std_n_overall = glass[param_n].std()
mean_m_overall = glass[param_m].mean()
std_m_overall = glass[param_m].std()

mean_n_by_class = glass.groupby('Type')[param_n].mean()
std_n_by_class = glass.groupby('Type')[param_n].std()
mean_m_by_class = glass.groupby('Type')[param_m].mean()
std_m_by_class = glass.groupby('Type')[param_m].std()

print(f"\nСреднее значение и стандартное отклонение для параметра {param_n}:")
print(f"Для всей выборки: среднее = {mean_n_overall:.4f}, стд. откл. = {std_n_overall:.4f}")
print("Для каждого класса:")
for class_type in glass_types.keys():
    if class_type in mean_n_by_class.index:
        print(f"  Класс {class_type} ({glass_types[class_type]}): среднее = {mean_n_by_class[class_type]:.4f}, стд. откл. = {std_n_by_class[class_type]:.4f}")

print(f"\nСреднее значение и стандартное отклонение для параметра {param_m}:")
print(f"Для всей выборки: среднее = {mean_m_overall:.4f}, стд. откл. = {std_m_overall:.4f}")
print("Для каждого класса:")
for class_type in glass_types.keys():
    if class_type in mean_m_by_class.index:
        print(f"  Класс {class_type} ({glass_types[class_type]}): среднее = {mean_m_by_class[class_type]:.4f}, стд. откл. = {std_m_by_class[class_type]:.4f}")

# 7. Статистически оценить различия в классах по параметру n
print(f"\nСтатистическая оценка различий в классах по параметру {param_n}:")
# Используем ANOVA для проверки различий между классами
from scipy.stats import f_oneway

# Создаем списки значений параметра n для каждого класса
class_values_n = [glass[glass['Type'] == class_type][param_n].values for class_type in sorted(glass['Type'].unique())]
f_stat_n, p_value_n = f_oneway(*class_values_n)

print(f"ANOVA тест для параметра {param_n}:")
print(f"F-статистика: {f_stat_n:.4f}")
print(f"p-значение: {p_value_n:.8f}")
if p_value_n < 0.05:
    print(f"Вывод: Существуют статистически значимые различия между классами по параметру {param_n} (p < 0.05)")
else:
    print(f"Вывод: Нет статистически значимых различий между классами по параметру {param_n} (p >= 0.05)")

# 8. Статистически оценить различия в классах по параметру m
print(f"\nСтатистическая оценка различий в классах по параметру {param_m}:")
# Используем ANOVA для проверки различий между классами
class_values_m = [glass[glass['Type'] == class_type][param_m].values for class_type in sorted(glass['Type'].unique())]
f_stat_m, p_value_m = f_oneway(*class_values_m)

print(f"ANOVA тест для параметра {param_m}:")
print(f"F-статистика: {f_stat_m:.4f}")
print(f"p-значение: {p_value_m:.8f}")
if p_value_m < 0.05:
    print(f"Вывод: Существуют статистически значимые различия между классами по параметру {param_m} (p < 0.05)")
else:
    print(f"Вывод: Нет статистически значимых различий между классами по параметру {param_m} (p >= 0.05)")

# Дополнительно: визуализация различий между классами
plt.figure(figsize=(16, 10))
sns.boxplot(x='Type', y=param_n, data=glass)
plt.title(f'Распределение параметра {param_n} по классам')
plt.xlabel('Тип стекла')
plt.ylabel(f'Содержание {param_n} (%)')
plt.grid(True)
# Исправление: получаем уникальные типы стекла в датасете и создаем метки только для них
unique_types = sorted(glass['Type'].unique())
plt.xticks(range(len(unique_types)), [f"{t} ({glass_types[t]})" for t in unique_types], rotation=90, ha='center')
plt.tight_layout(pad=2.0)
plt.savefig('plots/7_boxplot_n_by_class.png')
plt.show()
plt.close()

plt.figure(figsize=(16, 10))
sns.boxplot(x='Type', y=param_m, data=glass)
plt.title(f'Распределение параметра {param_m} по классам')
plt.xlabel('Тип стекла')
plt.ylabel(f'Содержание {param_m} (%)')
plt.grid(True)
# Исправление: используем те же уникальные типы для второго графика
plt.xticks(range(len(unique_types)), [f"{t} ({glass_types[t]})" for t in unique_types], rotation=90, ha='center')
plt.tight_layout(pad=2.0)
plt.savefig('plots/8_boxplot_m_by_class.png')
plt.show()
plt.close()

# Создание отчета в формате Markdown
with open('glass_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write("# Анализ датасета Glass Identification\n\n")
    
    f.write("## Информация о датасете\n")
    f.write(f"* Размер датасета: {glass.shape[0]} образцов, {glass.shape[1]-2} параметров\n")
    f.write(f"* Количество классов: {glass['Type'].nunique()}\n")
    f.write("* Распределение классов:\n")
    for class_type, count in glass['Type'].value_counts().items():
        f.write(f"  - Класс {class_type} ({glass_types[class_type]}): {count} образцов\n")
    
    f.write("\n## Анализ параметров\n")
    f.write(f"* Параметр n: {param_n} (содержание натрия)\n")
    f.write(f"* Параметр m: {param_m} (содержание магния)\n\n")
    
    f.write("### Медианные значения\n")
    f.write(f"* Медиана {param_n} (общая): {median_n_overall:.4f}\n")
    f.write("* Медиана по классам:\n")
    for class_type, median_value in median_n_by_class.items():
        f.write(f"  - Класс {class_type} ({glass_types[class_type]}): {median_value:.4f}\n")
    
    f.write(f"\n* Медиана {param_m} (общая): {median_m_overall:.4f}\n")
    f.write("* Медиана по классам:\n")
    for class_type, median_value in median_m_by_class.items():
        f.write(f"  - Класс {class_type} ({glass_types[class_type]}): {median_value:.4f}\n")
    
    f.write("\n### Средние значения и стандартные отклонения\n")
    f.write(f"* {param_n} (общее): среднее = {mean_n_overall:.4f}, стд. откл. = {std_n_overall:.4f}\n")
    f.write("* По классам:\n")
    for class_type in glass_types.keys():
        if class_type in mean_n_by_class.index:
            f.write(f"  - Класс {class_type} ({glass_types[class_type]}): среднее = {mean_n_by_class[class_type]:.4f}, стд. откл. = {std_n_by_class[class_type]:.4f}\n")
    
    f.write(f"\n* {param_m} (общее): среднее = {mean_m_overall:.4f}, стд. откл. = {std_m_overall:.4f}\n")
    f.write("* По классам:\n")
    for class_type in glass_types.keys():
        if class_type in mean_m_by_class.index:
            f.write(f"  - Класс {class_type} ({glass_types[class_type]}): среднее = {mean_m_by_class[class_type]:.4f}, стд. откл. = {std_m_by_class[class_type]:.4f}\n")
    
    f.write("\n## Статистическая оценка различий\n")
    f.write(f"### Параметр {param_n}\n")
    f.write(f"* ANOVA тест: F = {f_stat_n:.4f}, p = {p_value_n:.8f}\n")
    if p_value_n < 0.05:
        f.write(f"* Вывод: Существуют статистически значимые различия между классами по параметру {param_n} (p < 0.05)\n")
    else:
        f.write(f"* Вывод: Нет статистически значимых различий между классами по параметру {param_n} (p >= 0.05)\n")
    
    f.write(f"\n### Параметр {param_m}\n")
    f.write(f"* ANOVA тест: F = {f_stat_m:.4f}, p = {p_value_m:.8f}\n")
    if p_value_m < 0.05:
        f.write(f"* Вывод: Существуют статистически значимые различия между классами по параметру {param_m} (p < 0.05)\n")
    else:
        f.write(f"* Вывод: Нет статистически значимых различий между классами по параметру {param_m} (p >= 0.05)\n")
    
    f.write("\n## Визуализации\n")
    f.write("Все визуализации сохранены в директории `plots/`:\n")
    f.write("1. Распределение параметров по классам\n")
    f.write("2. Распределение параметров относительно медианных значений\n")
    f.write("3. Гистограммы распределения\n")
    f.write("4. Скаттерплоты\n")
    f.write("5. Боксплоты\n")

print("\nАнализ завершен. Результаты сохранены в файле glass_analysis_report.md и графики в директории plots/") 