import numpy as np

from core.Linear import Linear 
from core.Sequential import Sequential
from core.Flatten import Flatten
from activations.ReLU import ReLU
from activations.Sigmoid import Sigmoid
from losses.BCE import BCE
from optimizers.Adam import Adam
from initialization.He import He 
from initialization.Xavier import Xavier
from data.DataLoader import DataLoader
from data.Trainer import Trainer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    random_state=42,
    stratify=y_train
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
model = Sequential(
    Linear(30, 16, initializer=He()),
    ReLU(),

    Linear(16, 16, initializer=He()),
    ReLU(),

    Linear(16, 8, initializer=He()),
    ReLU(),

    Linear(8, 1, initializer=Xavier()),
    Sigmoid()
)
loss_fn = BCE()

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

dataloader = DataLoader(
    X_train,
    y_train,
    batch_size=1,
    shuffle=True
)

optimizer = Adam(
    learning_rate=0.001
)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    dataloader=dataloader
)

history = trainer.fit(
    epochs=1,
    X_val=X_val,
    y_val=y_val
)

test_loss, test_accuracy = trainer.evaluate(
    X_test,
    y_test
)

print(f"Test Loss: {test_loss:.6f}")
print(f"Test Accuracy: {test_accuracy:.4f}")