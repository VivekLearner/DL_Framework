import numpy as np

from core.Linear import Linear 
from core.Sequential import Sequential
from core.Flatten import Flatten
from activations.ReLU import ReLU
from activations.Softmax import Softmax
from losses.CCE import CCE
from optimizers.Adam import Adam
from initialization.He import He 
from initialization.Xavier import Xavier
from data.DataLoader import DataLoader
from data.Trainer import Trainer
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    shuffle=True
)
model = Sequential(
    Flatten(),

    Linear(784, 64, initializer=He()),
    ReLU(),

    Linear(64, 64, initializer=He()),
    ReLU(),

    Linear(64, 10, initializer=Xavier()),
    Softmax()
)

dataloader = DataLoader(
    X_train,
    y_train,
    batch_size=32,
    shuffle=True
)

optimizer = Adam(
    learning_rate=0.001
)

loss_fn = CCE()

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    dataloader=dataloader
)

history = trainer.fit(
    epochs=10,
    X_val=X_val,
    y_val=y_val
)

test_loss, test_accuracy = trainer.evaluate(
    X_test,
    y_test
)

print(f"Test Loss: {test_loss:.6f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt

predictions = trainer.predict(X_test)

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

plt.figure(figsize=(10, 6))

for i in range(10):

    plt.subplot(2, 5, i + 1)

    plt.imshow(X_test[i], cmap="gray")

    plt.title(
        f"Pred: {class_names[predictions[i]]}\n"
        f"Actual: {class_names[y_test[i]]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()