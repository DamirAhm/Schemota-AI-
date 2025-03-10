import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import itertools
from scipy.stats import f_oneway

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Создание директории для сохранения графиков
if not os.path.exists('plots_all'):
    os.makedirs('plots_all')

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

# Получение списка всех параметров для анализа (исключая Type и glass_type)
parameters = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']

# Словарь с описаниями параметров
param_descriptions = {
    'RI': 'Показатель преломления',
    'Na': 'Натрий',
    'Mg': 'Магний',
    'Al': 'Алюминий',
    'Si': 'Кремний',
    'K': 'Калий',
    'Ca': 'Кальций',
    'Ba': 'Барий',
    'Fe': 'Железо'
}

print("Информация о датасете:")
print(f"Размер датасета: {glass.shape}")
print(f"Количество классов: {glass['Type'].nunique()}")
print(f"Распределение классов:\n{glass['Type'].value_counts()}")
print("\nОписательная статистика:")
print(glass.describe())

# Создание отчета в формате Markdown
with open('glass_analysis_all_params_report.md', 'w', encoding='utf-8') as f:
    f.write("# Анализ датасета Glass Identification (все параметры)\n\n")
    
    f.write("## Информация о датасете\n")
    f.write(f"* Размер датасета: {glass.shape[0]} образцов, {len(parameters)} параметров\n")
    f.write(f"* Количество классов: {glass['Type'].nunique()}\n")
    f.write("* Распределение классов:\n")
    for class_type, count in glass['Type'].value_counts().items():
        f.write(f"  - Класс {class_type} ({glass_types[class_type]}): {count} образцов\n")
    
    f.write("\n## Анализ параметров\n")
    
    # Создаем словари для хранения результатов анализа
    medians = {}
    means = {}
    stds = {}
    medians_by_class = {}
    means_by_class = {}
    stds_by_class = {}
    f_stats = {}
    p_values = {}
    
    # Анализ каждого параметра
    for param in parameters:
        print(f"\nАнализ параметра {param} ({param_descriptions[param]})...")
        
        # Рассчитываем медиану для всей выборки и по классам
        median_overall = glass[param].median()
        median_by_class = glass.groupby('Type')[param].median()
        
        # Рассчитываем среднее и стандартное отклонение для всей выборки и по классам
        mean_overall = glass[param].mean()
        std_overall = glass[param].std()
        mean_by_class = glass.groupby('Type')[param].mean()
        std_by_class = glass.groupby('Type')[param].std()
        
        # Статистическая оценка различий между классами (ANOVA)
        class_values = [glass[glass['Type'] == class_type][param].values for class_type in sorted(glass['Type'].unique())]
        f_stat, p_value = f_oneway(*class_values)
        
        # Сохраняем результаты в словари
        medians[param] = median_overall
        means[param] = mean_overall
        stds[param] = std_overall
        medians_by_class[param] = median_by_class
        means_by_class[param] = mean_by_class
        stds_by_class[param] = std_by_class
        f_stats[param] = f_stat
        p_values[param] = p_value
        
        # Создаем признак для обозначения положения относительно медианы
        glass[f'above_median_{param}'] = glass[param] > median_overall
        
        # Запись результатов в отчет
        f.write(f"\n### Параметр: {param} ({param_descriptions[param]})\n")
        
        f.write("#### Медианные значения\n")
        f.write(f"* Медиана (общая): {median_overall:.4f}\n")
        f.write("* Медиана по классам:\n")
        for class_type, median_value in median_by_class.items():
            f.write(f"  - Класс {class_type} ({glass_types[class_type]}): {median_value:.4f}\n")
        
        f.write("\n#### Средние значения и стандартные отклонения\n")
        f.write(f"* Общее: среднее = {mean_overall:.4f}, стд. откл. = {std_overall:.4f}\n")
        f.write("* По классам:\n")
        for class_type in glass_types.keys():
            if class_type in mean_by_class.index:
                f.write(f"  - Класс {class_type} ({glass_types[class_type]}): среднее = {mean_by_class[class_type]:.4f}, стд. откл. = {std_by_class[class_type]:.4f}\n")
        
        f.write("\n#### Статистическая оценка различий\n")
        f.write(f"* ANOVA тест: F = {f_stat:.4f}, p = {p_value:.8f}\n")
        if p_value < 0.05:
            f.write(f"* Вывод: Существуют статистически значимые различия между классами по параметру {param} (p < 0.05)\n")
        else:
            f.write(f"* Вывод: Нет статистически значимых различий между классами по параметру {param} (p >= 0.05)\n")
        
        # Визуализация 1: Боксплот по классам
        plt.figure(figsize=(16, 10))
        sns.boxplot(x='Type', y=param, data=glass)
        plt.title(f'Распределение параметра {param} по классам')
        plt.xlabel('Тип стекла')
        plt.ylabel(f'Содержание {param} (%)')
        plt.grid(True)
        unique_types = sorted(glass['Type'].unique())
        plt.xticks(range(len(unique_types)), [f"{t} ({glass_types[t]})" for t in unique_types], rotation=90, ha='center')
        plt.tight_layout(pad=2.0)
        plt.savefig(f'plots_all/boxplot_{param}_by_class.png')
        plt.close()
        
        # Визуализация 2: Гистограмма распределения
        plt.figure(figsize=(14, 8))
        sns.histplot(data=glass, x=param, hue=f'above_median_{param}', 
                    palette=['red', 'blue'], kde=True)
        plt.title(f'Гистограмма распределения {param}')
        plt.xlabel(f'Содержание {param} (%)')
        plt.ylabel('Частота')
        plt.grid(True)
        plt.axvline(x=median_overall, color='green', linestyle='--', label=f'Медиана {param}')
        plt.legend(title=f'Выше медианы {param}', labels=['Нет', 'Да'])
        plt.savefig(f'plots_all/histogram_{param}.png')
        plt.close()
    
    # Создание матрицы корреляций и ее визуализация
    f.write("\n## Корреляционный анализ\n")
    
    # Рассчитываем корреляции между всеми параметрами
    correlation_matrix = glass[parameters].corr()
    
    # Визуализация корреляционной матрицы
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Корреляционная матрица параметров стекла')
    plt.tight_layout()
    plt.savefig('plots_all/correlation_matrix.png')
    plt.close()
    
    # Запись корреляционной матрицы в отчет
    f.write("### Корреляционная матрица\n")
    f.write("Корреляционная матрица показывает взаимосвязь между различными параметрами стекла:\n\n")
    
    # Форматирование корреляционной матрицы для Markdown
    f.write("| Параметр |")
    for param in parameters:
        f.write(f" {param} |")
    f.write("\n|")
    for _ in parameters:
        f.write(" --- |")
    f.write("\n")
    
    for param_row in parameters:
        f.write(f"| {param_row} |")
        for param_col in parameters:
            f.write(f" {correlation_matrix.loc[param_row, param_col]:.2f} |")
        f.write("\n")
    
    # Анализ наиболее сильных корреляций
    f.write("\n### Наиболее сильные корреляции\n")
    
    # Получаем пары параметров с корреляцией выше 0.5 или ниже -0.5 (исключая диагональ)
    strong_correlations = []
    for i, param1 in enumerate(parameters):
        for param2 in parameters[i+1:]:
            corr = correlation_matrix.loc[param1, param2]
            if abs(corr) > 0.5:
                strong_correlations.append((param1, param2, corr))
    
    # Сортируем по абсолютному значению корреляции
    strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if strong_correlations:
        f.write("Параметры с сильной корреляцией (|r| > 0.5):\n\n")
        for param1, param2, corr in strong_correlations:
            corr_type = "положительная" if corr > 0 else "отрицательная"
            f.write(f"* {param1} и {param2}: r = {corr:.2f} ({corr_type})\n")
            
            # Визуализация пар с сильной корреляцией
            plt.figure(figsize=(12, 8))
            sns.scatterplot(x=param1, y=param2, hue='glass_type', data=glass, palette='viridis', s=100)
            plt.title(f'Корреляция между {param1} и {param2} (r = {corr:.2f})')
            plt.xlabel(f'{param1} ({param_descriptions[param1]})')
            plt.ylabel(f'{param2} ({param_descriptions[param2]})')
            plt.grid(True)
            plt.legend(title='Тип стекла')
            plt.savefig(f'plots_all/correlation_{param1}_{param2}.png')
            plt.close()
    else:
        f.write("Не найдено параметров с сильной корреляцией (|r| > 0.5).\n")
    
    # Визуализация распределения параметров по парам
    f.write("\n## Визуализации распределения параметров\n")
    
    # Создаем pairplot для всех параметров
    plt.figure(figsize=(20, 20))
    sns.pairplot(glass, vars=parameters, hue='glass_type', palette='viridis', diag_kind='kde')
    plt.savefig('plots_all/pairplot_all_params.png')
    plt.close()
    
    f.write("### Матрица диаграмм рассеяния (Pairplot)\n")
    f.write("Матрица диаграмм рассеяния показывает взаимосвязь между всеми параметрами и их распределение.\n")
    f.write("Файл: `plots_all/pairplot_all_params.png`\n\n")
    
    # Анализ параметров, наиболее различающихся между классами
    f.write("\n## Параметры, наиболее различающиеся между классами\n")
    
    # Сортируем параметры по F-статистике (чем выше, тем сильнее различия между классами)
    discriminative_params = [(param, f_stats[param], p_values[param]) for param in parameters]
    discriminative_params.sort(key=lambda x: x[1], reverse=True)
    
    f.write("Параметры, отсортированные по степени различия между классами (на основе F-статистики ANOVA):\n\n")
    for param, f_stat, p_value in discriminative_params:
        significance = "статистически значимо" if p_value < 0.05 else "статистически не значимо"
        f.write(f"* {param} ({param_descriptions[param]}): F = {f_stat:.2f}, p = {p_value:.8f} ({significance})\n")
    
    # Визуализация топ-3 наиболее различающихся параметров
    top_params = [param for param, _, _ in discriminative_params[:3]]
    f.write("\n### Топ-3 параметра с наибольшими различиями между классами\n")
    
    for param in top_params:
        f.write(f"* {param} ({param_descriptions[param]}): F = {f_stats[param]:.2f}, p = {p_values[param]:.8f}\n")
    
    # Создаем 3D визуализацию для топ-3 параметров
    if len(top_params) >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        for class_type in sorted(glass['Type'].unique()):
            class_data = glass[glass['Type'] == class_type]
            ax.scatter(
                class_data[top_params[0]], 
                class_data[top_params[1]], 
                class_data[top_params[2]],
                label=f"{class_type} ({glass_types[class_type]})",
                s=50, alpha=0.7
            )
        
        ax.set_xlabel(f'{top_params[0]} ({param_descriptions[top_params[0]]})')
        ax.set_ylabel(f'{top_params[1]} ({param_descriptions[top_params[1]]})')
        ax.set_zlabel(f'{top_params[2]} ({param_descriptions[top_params[2]]})')
        ax.set_title('3D визуализация топ-3 параметров с наибольшими различиями между классами')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots_all/top3_discriminative_params_3d.png')
        plt.close()
        
        f.write("\n3D визуализация топ-3 параметров сохранена в файле: `plots_all/top3_discriminative_params_3d.png`\n")
    
    # Заключение
    f.write("\n## Заключение\n")
    
    # Наиболее дискриминативные параметры
    f.write("### Наиболее информативные параметры для различения типов стекла\n")
    significant_params = [(param, f_stat) for param, f_stat, p_value in discriminative_params if p_value < 0.05]
    if significant_params:
        f.write("Параметры, которые статистически значимо различаются между классами (в порядке убывания F-статистики):\n")
        for param, f_stat in significant_params:
            f.write(f"* {param} ({param_descriptions[param]}): F = {f_stat:.2f}\n")
    else:
        f.write("Не найдено параметров, которые статистически значимо различаются между классами.\n")
    
    # Корреляции
    f.write("\n### Взаимосвязи между параметрами\n")
    if strong_correlations:
        f.write("Наиболее сильные корреляции между параметрами:\n")
        for param1, param2, corr in strong_correlations[:3]:  # Топ-3 корреляции
            corr_type = "положительная" if corr > 0 else "отрицательная"
            f.write(f"* {param1} и {param2}: r = {corr:.2f} ({corr_type})\n")
    else:
        f.write("Не найдено сильных корреляций между параметрами.\n")
    
    f.write("\n### Общие выводы\n")
    f.write("1. Наиболее информативными параметрами для различения типов стекла являются: ")
    if significant_params:
        f.write(", ".join([f"{param} ({param_descriptions[param]})" for param, _ in significant_params[:3]]))
    f.write("\n")
    f.write("2. Между некоторыми параметрами существуют сильные корреляции, что указывает на взаимосвязь химического состава стекла.\n")
    f.write("3. Различные типы стекла имеют характерные особенности химического состава, что позволяет их классифицировать.\n")
    
    f.write("\n## Визуализации\n")
    f.write("Все визуализации сохранены в директории `plots_all/`:\n")
    f.write("1. Боксплоты распределения параметров по классам\n")
    f.write("2. Гистограммы распределения параметров\n")
    f.write("3. Корреляционная матрица\n")
    f.write("4. Диаграммы рассеяния для параметров с сильной корреляцией\n")
    f.write("5. Матрица диаграмм рассеяния (Pairplot)\n")
    f.write("6. 3D визуализация топ-3 параметров с наибольшими различиями между классами\n")

print("\nАнализ всех параметров завершен. Результаты сохранены в файле glass_analysis_all_params_report.md и графики в директории plots_all/") 