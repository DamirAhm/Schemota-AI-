import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
param_n = 'RI'  # Натрий
param_m = 'Na'  # Магний

# 1. Построить график распределения параметров. Разделить выборку по классам.
plt.figure(figsize=(14, 10))
sns.scatterplot(x=param_n, y=param_m, hue='glass_type', data=glass, palette='viridis', s=100)
plt.title(f'Распределение параметров {param_n} и {param_m} по типам стекла')
plt.xlabel(f'Содержание {param_n} (%)')
plt.ylabel(f'Содержание {param_m} (%)')
plt.grid(True)
plt.legend(title='Тип стекла', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plots/1_scatter_by_class.png')
plt.show()
plt.close()

# Рассчитать медианы
median_n_overall = glass[param_n].median()
median_m_overall = glass[param_m].median()

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

# 5. Построить гистограмму распределения, скатерограмму и боксплот параметров
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
sns.boxplot(data=glass, x='Type', y=param_n, ax=axes[1, 0])
axes[1, 0].set_title(f'Боксплот {param_n} по классам')
axes[1, 0].set_xlabel('Тип стекла')
axes[1, 0].set_ylabel(f'Содержание {param_n} (%)')
axes[1, 0].grid(True)
unique_types = sorted(glass['Type'].unique())
# Правильно устанавливаем позиции тиков перед установкой меток
tick_positions = range(len(unique_types))
axes[1, 0].set_xticks(tick_positions)
axes[1, 0].set_xticklabels([f"{t}" for t in unique_types], rotation=90, ha='center')

sns.boxplot(data=glass, x='Type', y=param_m, ax=axes[1, 1])
axes[1, 1].set_title(f'Боксплот {param_m} по классам')
axes[1, 1].set_xlabel('Тип стекла')
axes[1, 1].set_ylabel(f'Содержание {param_m} (%)')
axes[1, 1].grid(True)
# Правильно устанавливаем позиции тиков перед установкой меток
axes[1, 1].set_xticks(tick_positions)
axes[1, 1].set_xticklabels([f"{t}" for t in unique_types], rotation=90, ha='center')

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

# Дополнительно: визуализация различий между классами
plt.figure(figsize=(16, 10))
sns.boxplot(x='Type', y=param_n, data=glass)
plt.title(f'Распределение параметра {param_n} по классам')
plt.xlabel('Тип стекла')
plt.ylabel(f'Содержание {param_n} (%)')
plt.grid(True)
# Исправление: получаем уникальные типы стекла в датасете и создаем метки только для них
unique_types = sorted(glass['Type'].unique())
# Правильно устанавливаем позиции тиков перед установкой меток
tick_positions = range(len(unique_types))
plt.xticks(tick_positions, [f"{t}" for t in unique_types], rotation=90, ha='center')
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
# Правильно устанавливаем позиции тиков перед установкой меток
plt.xticks(tick_positions, [f"{t}" for t in unique_types], rotation=90, ha='center')
plt.tight_layout(pad=2.0)
plt.savefig('plots/8_boxplot_m_by_class.png')
plt.show()
plt.close()

# Матрица корреляций
plt.figure(figsize=(12, 10))
corr = glass.drop(['Type', 'glass_type', 'above_median_n', 'above_median_m'], axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', square=True, linewidths=.5)
plt.title('Матрица корреляций параметров')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.show()
plt.close()

# Парные графики для основных параметров
plt.figure(figsize=(16, 14))
pair_plot = sns.pairplot(glass, hue='Type', vars=['Mg', 'Ba', 'Al', 'Na', 'K',], palette='viridis', height=3, aspect=1.2)
plt.suptitle('Парные графики для основных параметров', y=1.02)
# Улучшаем легенду
handles = pair_plot._legend_data.values()
# Преобразуем ключи в целые числа перед доступом к словарю glass_types
labels = [f"{int(t)}" for t in pair_plot._legend_data.keys()]
pair_plot._legend.remove()
pair_plot.fig.legend(handles=handles, labels=labels, title='Тип стекла', 
                    loc='upper right', bbox_to_anchor=(0.99, 0.99))
plt.tight_layout()
plt.savefig('plots/pairplot.png')
plt.show()
plt.close()

print("Графики успешно созданы и сохранены в директории plots/") 