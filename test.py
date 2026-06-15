import mchine as nn
import numpy as np

# Создаем сеть для XOR
net = nn.NeuralNet()

# Добавляем слои
l1 = nn.Layer(2, 3)  # 2 входа, 3 нейрона
l1.setActivation(nn.ActivationType.ReLU)

l2 = nn.Layer(3, 1)  # 3 входа, 1 выход  
l2.setActivation(nn.ActivationType.Sigmoid)

net.addLayer(l1)
net.addLayer(l2)

# Данные для XOR
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([[0.], [1.], [1.], [0.]])

# Альтернативный способ через списки
samples = [
    nn.TrainingSample(),
    nn.TrainingSample(),
    nn.TrainingSample(),
    nn.TrainingSample()
]

samples[0].input = nn.tensor_from_list([0.0, 0.0])
samples[0].target = nn.tensor_from_list([0.0])

samples[1].input = nn.tensor_from_list([0.0, 1.0])
samples[1].target = nn.tensor_from_list([1.0])

samples[2].input = nn.tensor_from_list([1.0, 0.0])
samples[2].target = nn.tensor_from_list([1.0])

samples[3].input = nn.tensor_from_list([1.0, 1.0])
samples[3].target = nn.tensor_from_list([0.0])

# Обучение (можно использовать любой из методов)
print("Training...")
# Метод 1: через numpy массивы
net.train_from_numpy(X, y, learning_rate=0.1, epochs=1000)

# Метод 2: через список TrainingSample
# net.train_on_batch(samples, learning_rate=0.1, epochs=1000)

# Предсказания
print("\nPredictions:")
for x in X:
    pred = net.predict_from_numpy(x)
    print(f"Input: {x} -> Output: {pred(0, 0):.4f}")

# Пакетные предсказания
print("\nBatch predictions:")
all_preds = net.predict_batch(X)
for i, pred in enumerate(all_preds):
    print(f"Sample {i}: {pred}")

# Получаем веса
weights = net.get_weights()
print(f"\nTotal weights: {len(weights)}")
print(f"First few weights: {weights[:5]}")

# Экспорт в numpy
weights_np = net.exportWeights()
print(f"Weights as numpy: {weights_np}")

# Тест softmax
logits = np.array([2.0, 1.0, 0.1])
tensor_logits = nn.tensor_from_list(logits.tolist())
probs = nn.softmax(tensor_logits)
print(f"\nSoftmax probabilities: {probs.to_list()}")
