
import numpy as np
import matplotlib.pyplot as plt
import DataGenerator as dg

# --- Функции активации ---
def sigmoid(Z):
    Z = np.clip(Z, -500, 500)  # Ограничиваем значения Z
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(p):
    return p * (1 - p)

# --- Класс нейронной сети ---
class NeuralNetwork:
    def __init__(self, x, y, n_neuro=4, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.input = x
        self.y = y.reshape(-1, 1)
        n_inp = x.shape[1]
        self.n_neuro = n_neuro
        self.weights1 = np.random.randn(n_inp, n_neuro) * 0.1  # Хэвисайд-инициализация
        self.weights2 = np.random.randn(n_neuro, 1) * 0.1
        self.output = np.zeros(self.y.shape)
        self.loss_history = []
        self.accuracy_history = []
        self.val_loss = []
        self.val_accuracy = []

    def feedforward(self, X=None):
        if X is None:
            X = self.input
        layer1 = sigmoid(np.dot(X, self.weights1))
        return sigmoid(np.dot(layer1, self.weights2))

    def backprop(self):
        # Ошибки на выходном слое
        output = self.output
        d_output = (output - self.y) * sigmoid_derivative(output)
        
        # Градиенты для weights2
        d_weights2 = np.dot(self.layer1.T, d_output)
        
        # Градиенты для weights1
        d_layer1 = np.dot(d_output, self.weights2.T) * sigmoid_derivative(self.layer1)
        d_weights1 = np.dot(self.input.T, d_layer1)
        
        # Обновление весов
        self.weights1 -= self.learning_rate * d_weights1 # Добавлен learning_rate
        self.weights2 -= self.learning_rate * d_weights2

    def train(self, X_val=None, Y_val=None, epochs=100, verbose=True):
        if X_val is not None:
            X_val = X_val
            Y_val = Y_val.reshape(-1, 1)
        
        for epoch in range(epochs):
            # Прямой проход
            self.layer1 = sigmoid(np.dot(self.input, self.weights1))
            output = sigmoid(np.dot(self.layer1, self.weights2))
            self.output = output
            
            # Обратное распространение
            self.backprop()
            
            # Сохранение метрик
            loss = -np.mean(self.y * np.log(output) + (1 - self.y) * np.log(1 - output)) #Изменено
            self.loss_history.append(loss)
            accuracy = np.mean((output > 0.5) == self.y)
            self.accuracy_history.append(accuracy)
            
            if X_val is not None:
                val_pred = self.feedforward(X_val)
                val_loss = np.mean(np.square(Y_val - val_pred))
                self.val_loss.append(val_loss)
                val_acc = np.mean((val_pred > 0.5) == Y_val)
                self.val_accuracy.append(val_acc)
            
            if verbose and epoch % 10 == 0:
                print(f"Эпоха {epoch}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")

    def get_weights(self):
        return {
            'weights1': self.weights1,
            'weights2': self.weights2
        }

# --- Генерация данных ---
def prepare_data(dataset_type="norm", test_size=0.2):
    if dataset_type == "norm":
        mu = [[0, 2, 3], [3, 5, 1]]
        sigma = [[2, 1, 2], [1, 2, 1]]
        N = 1000
        X, Y, _, _ = dg.norm_dataset(mu, sigma, N)
    else:
        N = 500
        X, Y, _, _ = dg.nonlinear_dataset_8(N)
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    return X_train, X_test, Y_train, Y_test

# --- Оптимизация гиперпараметров ---
def optimize_hyperparameters(X_train, Y_train, X_val, Y_val):
    best_accuracy = 0
    best_params = {'n_neuro': 0, 'epochs': 0}
    
    for n_neuro in [4, 8, 12, 16]:
        for epochs in [100, 200, 300]:
            nn = NeuralNetwork(X_train, Y_train, 
                              n_neuro=n_neuro, 
                              learning_rate=0.001)
            
            nn.train(X_val, Y_val, epochs=epochs, verbose=False)
            
            max_val_acc = max(nn.val_accuracy) if nn.val_accuracy else 0
            
            if max_val_acc > best_accuracy:
                best_accuracy = max_val_acc
                best_params = {'n_neuro': n_neuro, 'epochs': epochs}
                
    return best_params

# --- Основной код ---
# Нормальный датасет
print("=== Нормальный датасет ===")
X_train_n, X_test_n, Y_train_n, Y_test_n = prepare_data(dataset_type="norm")
X_val_n, X_test_n = X_test_n[:len(X_test_n)//2], X_test_n[len(X_test_n)//2:]
Y_val_n, Y_test_n = Y_test_n[:len(Y_test_n)//2], Y_test_n[len(Y_test_n)//2:]

best_params_norm = optimize_hyperparameters(X_train_n, Y_train_n, X_val_n, Y_val_n)
print(f"Оптимальные параметры: {best_params_norm}")

nn_norm = NeuralNetwork(X_train_n, Y_train_n, n_neuro=best_params_norm['n_neuro'])
nn_norm.train(X_val_n, Y_val_n, epochs=best_params_norm['epochs'])
weights_norm = nn_norm.get_weights()

# Нелинейный датасет
print("\n=== Нелинейный датасет ===")
X_train_nl, X_test_nl, Y_train_nl, Y_test_nl = prepare_data(dataset_type="nonlinear")
X_val_nl, X_test_nl = X_test_nl[:len(X_test_nl)//2], X_test_nl[len(X_test_nl)//2:]
Y_val_nl, Y_test_nl = Y_test_nl[:len(Y_test_nl)//2], Y_test_nl[len(Y_test_nl)//2:]

best_params_nonlin = optimize_hyperparameters(X_train_nl, Y_train_nl, X_val_nl, Y_val_nl)
print(f"Оптимальные параметры: {best_params_nonlin}")

nn_nonlin = NeuralNetwork(X_train_nl, Y_train_nl, n_neuro=best_params_nonlin['n_neuro'])
nn_nonlin.train(X_val_nl, Y_val_nl, epochs=best_params_nonlin['epochs'])
weights_nonlin = nn_nonlin.get_weights()

# --- Построение графиков ---
def plot_metrics(nn, title):
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn.loss_history, label='Train Loss')
    if nn.val_loss:
        plt.plot(nn.val_loss, label='Validation Loss', linestyle='--')
    plt.title(f'Loss на {title} датасете')
    plt.xlabel('Эпоха')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(nn.accuracy_history, label='Train Accuracy')
    if nn.val_accuracy:
        plt.plot(nn.val_accuracy, label='Validation Accuracy', linestyle='--')
    plt.title(f'Точность на {title} датасете')
    plt.xlabel('Эпоха')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_metrics(nn_norm, "нормальном")
plot_metrics(nn_nonlin, "нелинейном")

# --- Сохранение весов ---
print("\nВеса для нормального датасета:")
print(f"Weights1 shape: {weights_norm['weights1'].shape}")
print(f"Weights2 shape: {weights_norm['weights2'].shape}")

print("\nВеса для нелинейного датасета:")
print(f"Weights1 shape: {weights_nonlin['weights1'].shape}")
print(f"Weights2 shape: {weights_nonlin['weights2'].shape}")