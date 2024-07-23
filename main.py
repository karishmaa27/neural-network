import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from dense import Dense
from activations import Tanh, Softmax
from losses import mse, mse_prime
from network import train, predict

iris_df = pd.read_csv("iris.csv")
iris_df = iris_df.sample(frac=1).reset_index(drop=True)  # Shuffle

X = iris_df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]
X = np.array(X)

one_hot_encoder = OneHotEncoder(sparse_output=False)
Y = iris_df['variety']
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.15)


network = [
    Dense(4, 10),  
    Tanh(),
    Dense(10, 3),  
    Softmax()
]


train(network, mse, mse_prime, X_train, Y_train, epochs=1000, learning_rate=0.01)


def evaluate(network, X, Y):
    predictions = []
    for x in X:
        output = predict(network, x.reshape(-1, 1))
        predictions.append(output)
    predictions = np.array(predictions).squeeze()
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(Y, axis=1))
    return accuracy

train_accuracy = evaluate(network, X_train, Y_train)
val_accuracy = evaluate(network, X_val, Y_val)
test_accuracy = evaluate(network, X_test, Y_test)

print(f"Training accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")
