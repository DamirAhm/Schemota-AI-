import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

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
        
        # Инициализация весов с масштабированием Xavier/He
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
    
    def feedforward(self, X=None):
        if X is None:
            X = self.input
        
        # Первый слой
        self.layer1 = sigmoid(np.dot(X, self.weights1) + self.bias1)
        
        # Выходной слой с softmax
        self.output = softmax(np.dot(self.layer1, self.weights2) + self.bias2)
        
        return self.output
    
    def backprop(self):
        batch_size = self.input.shape[0]
        
        # Градиент для выходного слоя
        d_output = self.output - self.y  # Градиент для softmax с кросс-энтропией
        
        # Градиенты для второго слоя весов
        d_weights2 = np.dot(self.layer1.T, d_output) / batch_size
        d_bias2 = np.sum(d_output, axis=0, keepdims=True) / batch_size
        
        # Градиенты для первого слоя
        d_layer1 = np.dot(d_output, self.weights2.T) * sigmoid_derivative(self.layer1)
        d_weights1 = np.dot(self.input.T, d_layer1) / batch_size
        d_bias1 = np.sum(d_layer1, axis=0, keepdims=True) / batch_size
        
        # Обновление весов с L2 регуляризацией
        reg_lambda = 0.001  # Коэффициент регуляризации
        self.weights2 -= self.learning_rate * (d_weights2 + reg_lambda * self.weights2)
        self.bias2 -= self.learning_rate * d_bias2
        
        self.weights1 -= self.learning_rate * (d_weights1 + reg_lambda * self.weights1)
        self.bias1 -= self.learning_rate * d_bias1
    
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
    
    def train(self, X_val=None, y_val=None, epochs=1000, batch_size=32, verbose=True):
        n_samples = self.input.shape[0]
        n_batches = max(n_samples // batch_size, 1)
        
        for epoch in range(epochs):
            # Перемешиваем данные
            indices = np.random.permutation(n_samples)
            X_shuffled = self.input[indices]
            y_shuffled = self.y[indices]
            
            epoch_loss = 0
            
            # Мини-батчи
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Временно заменяем входные данные на текущий батч
                self.input = X_batch
                self.y = y_batch
                
                # Прямой проход
                self.feedforward()
                
                # Обратное распространение
                self.backprop()
                
                # Вычисляем потери для текущего батча
                batch_loss = self.compute_loss(y_batch, self.output)
                epoch_loss += batch_loss * (end_idx - start_idx) / n_samples
            
            # Восстанавливаем полный набор данных
            self.input = X_shuffled
            self.y = y_shuffled
            
            # Прямой проход для всего набора данных
            train_pred = self.feedforward()
            train_accuracy = self.compute_accuracy(self.y, train_pred)
            
            # Сохраняем метрики
            self.loss_history.append(epoch_loss)
            self.accuracy_history.append(train_accuracy)
            
            # Валидация
            if X_val is not None and y_val is not None:
                val_pred = self.feedforward(X_val)
                val_loss = self.compute_loss(y_val, val_pred)
                val_accuracy = self.compute_accuracy(y_val, val_pred)
                
                self.val_loss_history.append(val_loss)
                self.val_accuracy_history.append(val_accuracy)
                
                if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                    print(f"Эпоха {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                    print(f"Эпоха {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.4f}")
    
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
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    glass = pd.read_csv(url, names=columns)
    
    # Удаление столбца Id
    glass = glass.drop('Id', axis=1)
    
    # Словарь для маппинга типов стекла
    glass_types = {
        1: 'building_windows_float',
        2: 'building_windows_non_float',
        3: 'vehicle_windows_float',
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
    
    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y
    )
    
    # Разделение тестовой выборки на валидационную и тестовую
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    # Балансировка классов с помощью SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled_indices = smote.fit_resample(X_train, np.argmax(y_train, axis=1))
        
        # Преобразование индексов обратно в one-hot encoding
        y_train_resampled = np.zeros((X_train_resampled.shape[0], y_train.shape[1]))
        for i, idx in enumerate(y_train_resampled_indices):
            y_train_resampled[i, idx] = 1
        
        print(f"Размер обучающей выборки после SMOTE: {X_train_resampled.shape[0]}")
        print(f"Распределение классов после SMOTE: {np.sum(y_train_resampled, axis=0)}")
        
        return X_train_resampled, X_val, X_test, y_train_resampled, y_val, y_test, glass_types, encoder
    except ImportError:
        print("SMOTE не установлен. Используем оригинальные данные.")
        return X_train, X_val, X_test, y_train, y_val, y_test, glass_types, encoder

# --- Оптимизация гиперпараметров ---
def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    best_val_accuracy = 0
    best_params = {}
    
    # Расширенная сетка гиперпараметров
    hidden_neurons_options = [30, 50, 100, (50, 30)]
    learning_rate_options = [0.01, 0.005, 0.001]
    
    for hidden_neurons in hidden_neurons_options:
        for lr in learning_rate_options:
            print(f"Тестирование: hidden_neurons={hidden_neurons}, learning_rate={lr}")
            
            # Создание и обучение модели
            nn = NeuralNetwork(
                X_train, y_train,
                hidden_neurons=hidden_neurons,
                learning_rate=lr,
                n_classes=y_train.shape[1]
            )
            
            # Обучение с ранней остановкой
            patience = 10  # Увеличиваем терпение
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(500):  # Максимум 500 эпох
                # Мини-эпоха обучения
                nn.train(X_val, y_val, epochs=1, batch_size=32, verbose=False)
                
                # Проверка ранней остановки
                current_val_loss = nn.val_loss_history[-1]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"  Ранняя остановка на эпохе {epoch+1}")
                    break
            
            # Оценка на валидационной выборке
            val_accuracy = nn.val_accuracy_history[-1]
            print(f"  Валидационная точность: {val_accuracy:.4f}")
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = {
                    'hidden_neurons': hidden_neurons,
                    'learning_rate': lr
                }
    
    print(f"Лучшие параметры: {best_params}, Валидационная точность: {best_val_accuracy:.4f}")
    return best_params

# --- Визуализация результатов ---
def plot_metrics(nn, title):
    # Создание директории для графиков, если она не существует
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    plt.figure(figsize=(15, 5))
    
    # График потерь
    plt.subplot(1, 2, 1)
    plt.plot(nn.loss_history, label='Train Loss')
    if nn.val_loss_history:
        plt.plot(nn.val_loss_history, label='Validation Loss')
    plt.title(f'Loss - {title}')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # График точности
    plt.subplot(1, 2, 2)
    plt.plot(nn.accuracy_history, label='Train Accuracy')
    if nn.val_accuracy_history:
        plt.plot(nn.val_accuracy_history, label='Validation Accuracy')
    plt.title(f'Accuracy - {title}')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/custom_nn_metrics.png')
    plt.close()

# --- Оценка модели ---
def evaluate_model(nn, X_test, y_test, encoder, glass_types):
    # Предсказания
    y_pred = nn.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    
    # Преобразование индексов в исходные метки классов
    original_classes = encoder.categories_[0]
    y_true_original = original_classes[y_true]
    y_pred_original = original_classes[y_pred]
    
    # Точность
    accuracy = accuracy_score(y_true_original, y_pred_original)
    print(f"Точность на тестовой выборке: {accuracy:.4f}")
    
    # Отчет о классификации
    print("\nОтчет о классификации:")
    report = classification_report(y_true_original, y_pred_original, output_dict=True)
    
    # Форматированный вывод отчета
    print("### Результаты по классам:\n")
    for class_label in sorted(report.keys()):
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            class_info = report[class_label]
            class_name = glass_types.get(int(class_label), f"Класс {class_label}")
            
            print(f"**{class_name}**:")
            print(f"    Precision (точность): {class_info['precision']:.2f}")
            print(f"    Recall (полнота): {class_info['recall']:.2f}")
            print(f"    F1-score (F-мера): {class_info['f1-score']:.2f}")
            print(f"    Support (количество образцов): {class_info['support']}\n")
    
    print("### Общие метрики:\n")
    print(f"**Accuracy (точность)**: {report['accuracy']:.2f}\n")
    
    print("**Macro Avg**:")
    print(f"    Precision: {report['macro avg']['precision']:.2f}")
    print(f"    Recall: {report['macro avg']['recall']:.2f}")
    print(f"    F1-score: {report['macro avg']['f1-score']:.2f}\n")
    
    print("**Weighted Avg**:")
    print(f"    Precision: {report['weighted avg']['precision']:.2f}")
    print(f"    Recall: {report['weighted avg']['recall']:.2f}")
    print(f"    F1-score: {report['weighted avg']['f1-score']:.2f}")
    
    # Матрица ошибок
    cm = confusion_matrix(y_true_original, y_pred_original)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Матрица ошибок')
    plt.colorbar()
    
    # Метки для матрицы ошибок
    class_names = [glass_types.get(int(cls), f"Класс {cls}") for cls in sorted(original_classes)]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Аннотации значений
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.savefig('plots/custom_nn_confusion_matrix.png')
    plt.close()
    
    return report

# --- Основной код ---
if __name__ == "__main__":
    print("Загрузка и подготовка данных...")
    X_train, X_val, X_test, y_train, y_val, y_test, glass_types, encoder = prepare_glass_data()
    
    print("\nОптимизация гиперпараметров...")
    best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val)
    
    print("\nОбучение модели с оптимальными параметрами...")
    nn = NeuralNetwork(
        X_train, y_train,
        hidden_neurons=best_params['hidden_neurons'],
        learning_rate=best_params['learning_rate'],
        n_classes=y_train.shape[1]
    )
    
    nn.train(X_val, y_val, epochs=1000, batch_size=32, verbose=True)
    
    print("\nВизуализация метрик обучения...")
    plot_metrics(nn, "Glass Classification")
    
    print("\nОценка модели на тестовой выборке...")
    report = evaluate_model(nn, X_test, y_test, encoder, glass_types)
    
    print("\nОбучение завершено. Результаты сохранены в директории 'plots/'.") 