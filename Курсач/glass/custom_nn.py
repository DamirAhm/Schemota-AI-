import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import seaborn as sns

# --- Функции активации ---
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
        self.val_loss_history = []
        self.val_accuracy_history = []
        
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

# --- Подготовка данных ---
def prepare_glass_data():
    # Загрузка датасета
    try:
        # Попытка загрузить из локального CSV файла
        # Файл не имеет заголовков, поэтому добавляем их
        columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
        glass = pd.read_csv('glass.csv', names=columns, header=None)
    except:
        # Если локальный файл не найден, загружаем из UCI репозитория
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
        glass = pd.read_csv(url, names=columns, header=None)
        # Сохраняем для будущего использования
        glass.to_csv('glass.csv', index=False, header=False)
    
    # Удаление столбца Id, если он существует
    if 'Id' in glass.columns:
        glass = glass.drop('Id', axis=1)
    
    # Словарь типов стекла
    glass_types = {
        1: 'building_windows_float',
        2: 'building_windows_non_float',
        3: 'vehicle_windows_float',
        4: 'vehicle_windows_non_float',
        5: 'containers',
        6: 'tableware',
        7: 'headlamps'
    }
    
    # Подготовка данных
    X = glass.drop('Type', axis=1).values
    y = glass['Type'].values
    
    # Стандартизация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encoding для меток классов
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))
    
    print(f"Размер датасета: {X_scaled.shape[0]}")
    print(f"Распределение классов: {np.sum(y_encoded, axis=0)}")
    
    return X_scaled, y_encoded, glass_types, encoder

# --- Оптимизация гиперпараметров ---
def optimize_hyperparameters(X, y):
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

# --- Визуализация результатов ---
def plot_metrics(nn, title):
    # Создание директории для графиков, если она не существует
    os.makedirs('plots', exist_ok=True)
    
    # График потерь
    plt.figure(figsize=(10, 5))
    plt.plot(nn.loss_history, label='Потери на обучающей выборке')
    plt.title(f'{title} - График потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/loss_plot.png')
    plt.close()
    
    # График точности
    plt.figure(figsize=(10, 5))
    plt.plot(nn.accuracy_history, label='Точность на обучающей выборке')
    plt.title(f'{title} - График точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/accuracy_plot.png')
    plt.close()

# --- Оценка модели ---
def evaluate_model(nn, X, y, encoder, glass_types):
    # Предсказания на данных
    y_pred = nn.predict(X)
    y_pred_proba = nn.predict_proba(X)
    
    # Преобразование one-hot encoded y обратно в метки классов
    y_true_classes = np.argmax(y, axis=1)
    
    # Оценка точности
    accuracy = np.mean(y_pred == y_true_classes)
    print(f"Точность на датасете: {accuracy:.4f}")
    
    # Отчет о классификации
    # Получаем уникальные классы в данных
    unique_classes = np.unique(y_true_classes)
    class_names = [glass_types[i+1] for i in range(len(unique_classes))]
    
    report = classification_report(y_true_classes, y_pred, labels=range(len(unique_classes)))
    print("\nОтчет о классификации:")
    print(report)
    
    # Форматированный отчет
    report_dict = classification_report(y_true_classes, y_pred, output_dict=True, labels=range(len(unique_classes)))
    formatted_report = "### Результаты по классам:\n\n"
    
    # Добавляем информацию по каждому классу
    for i, class_name in enumerate(class_names):
        if str(i) in report_dict:
            class_info = report_dict[str(i)]
            formatted_report += f"**{class_name}**:\n"
            formatted_report += f"    Precision (точность): {class_info['precision']:.2f}\n"
            formatted_report += f"    Recall (полнота): {class_info['recall']:.2f}\n"
            formatted_report += f"    F1-score (F-мера): {class_info['f1-score']:.2f}\n"
            formatted_report += f"    Support (количество образцов): {class_info['support']}\n\n"
    
    # Добавляем общую информацию
    formatted_report += "### Общие метрики:\n\n"
    formatted_report += f"**Accuracy (точность)**: {report_dict['accuracy']:.2f}\n\n"
    formatted_report += f"**Macro Avg**:\n"
    formatted_report += f"    Precision: {report_dict['macro avg']['precision']:.2f}\n"
    formatted_report += f"    Recall: {report_dict['macro avg']['recall']:.2f}\n"
    formatted_report += f"    F1-score: {report_dict['macro avg']['f1-score']:.2f}\n\n"
    formatted_report += f"**Weighted Avg**:\n"
    formatted_report += f"    Precision: {report_dict['weighted avg']['precision']:.2f}\n"
    formatted_report += f"    Recall: {report_dict['weighted avg']['recall']:.2f}\n"
    formatted_report += f"    F1-score: {report_dict['weighted avg']['f1-score']:.2f}\n"
    
    print(formatted_report)
    
    # Матрица ошибок
    cm = confusion_matrix(y_true_classes, y_pred, labels=range(len(unique_classes)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Матрица ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.savefig('plots/custom_nn_confusion_matrix.png')
    plt.close()
    
    return report

# --- Основная функция ---
if __name__ == "__main__":
    print("Загрузка и подготовка данных...")
    X_data, y_data, glass_types, encoder = prepare_glass_data()
    
    print("\nОптимизация гиперпараметров...")
    best_params = optimize_hyperparameters(X_data, y_data)
    
    print("\nОбучение модели с оптимальными параметрами...")
    nn = NeuralNetwork(
        X_data, y_data,
        hidden_neurons=best_params['hidden_neurons'],
        learning_rate=best_params['learning_rate'],
        n_classes=y_data.shape[1]
    )
    
    # Обучение модели
    nn.train(epochs=500, batch_size=32, verbose=True)
    
    print("\nВизуализация метрик обучения...")
    plot_metrics(nn, f"Нейронная сеть (hidden_neurons={best_params['hidden_neurons']}, lr={best_params['learning_rate']})")
    
    print("\nОценка модели на датасете...")
    evaluate_model(nn, X_data, y_data, encoder, glass_types)
    
    print("\nОбучение завершено. Результаты сохранены в директории 'plots/'.") 