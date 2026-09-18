import numpy as np

input('For START Press Enter.....')

# Подготовка данных
# определяем входные данные (X) и целевую переменную (y) 
# таблицы истинности для XOR 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
y = np.array([[0], [1], [1], [0]]) 
print("Входные данные (X):") 
print(X) 
print("\nЦелевые значения (y):") 
print(y)

# Создаём архитектуру нейросети
input('Создаём архитектуру.....')
def initialize_weights(input_size, hidden_size, output_size): 
    np.random.seed(42)  
    weights1 = np.random.randn(input_size, hidden_size) 
    weights2 = np.random.randn(hidden_size, output_size) 
    return weights1, weights2 
 
# Задаем размеры слоев 
input_size = 2 
hidden_size = 4 
output_size = 1 
 
# Инициализируем веса 
weights1, weights2 = initialize_weights(input_size, hidden_size, 
output_size) 
 
print("Веса W1 (между входным и скрытым слоем):") 
print(weights1) 
print("\nВеса W2 (между скрытым и выходным слоем):") 
print(weights2)

# Функция активации
def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 
 
# производная сигмоиды (нужна для обратного распространения ошибки, об этом написано далее) 
def sigmoid_derivative(x): 
    return x * (1 - x) 
  
  
# Прямое распространение (Forward Pass)
def forward_pass(X, weights1, weights2): 
    #умножаем входные данные на веса между входным и скрытым слоем 
    hidden_layer_input = np.dot(X, weights1) 
    #применяем функцию активации к скрытому слою 
    hidden_layer_output = sigmoid(hidden_layer_input)
    #умножаем выход скрытого слоя на веса между скрытым и выходным слоем 
    output_layer_input = np.dot(hidden_layer_output, weights2) 
    # применяем функцию активации к выходному слою (здесь тоже сигмоида) 
    predicted_output = sigmoid(output_layer_input) 
    return hidden_layer_output, predicted_output


# Обратное распространение ошибки (Backpropagation)
def backward_pass(X, y, hidden_layer_output, predicted_output, weights1, weights2, learning_rate): 
    #вычисляем ошибку на выходном слое 
    output_error = y - predicted_output
    #вычисляем дельту для выходного слоя (ошибка * производную функции активации) 
    output_delta = output_error * sigmoid_derivative(predicted_output)
    # вычисляем ошибку на скрытом слое 
    hidden_layer_error = output_delta.dot(weights2.T)
    # вычисляем разницу или дельту для скрытого слоя 
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)
    #обновляем веса: добавляем долю от произведения входного сигнала и дельты
    weights2 += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights1 += X.T.dot(hidden_layer_delta) * learning_rate
    return weights1, weights2
    

# Цикл обучения и запуск нейросети
# гиперпараметры (настраиваются экспериментально) 
learning_rate = 10  # по умолчанию было 0.1
epochs = 30000   # по умолчанию было 10000
 
# инициализируем веса 
weights1, weights2 = initialize_weights(input_size, hidden_size, output_size) 
 
#цикл обучения 
for i in range(epochs): 
    #прямой проход 
    hidden_layer_output, predicted_output = forward_pass(X, weights1, weights2) 
     
    #обратный проход и обновление весов 
    weights1, weights2 = backward_pass(X, y, hidden_layer_output, predicted_output, weights1, weights2, learning_rate)

#периодический вывод ошибки для отслеживания процесса 
    if i % 1000 == 0: 
        error = np.mean(np.abs(y - predicted_output)) 
        print(f"Эпоха {i}, Ошибка: {error:.6f}") 
 
# финальное предсказание после обучения 
print("\nРезультат после обучения:") 
hidden_layer_output, predicted_output = forward_pass(X, weights1, 
weights2) 
print("Округленные предсказания:") 
print(np.round(predicted_output))

input('Готово...   Press Enter.....')
