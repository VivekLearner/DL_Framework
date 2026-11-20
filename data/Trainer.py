import numpy as np


class Trainer:

    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        dataloader,
        scheduler=None,
        regularizer=None
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.scheduler = scheduler
        self.regularizer = regularizer
    def fit(self, epochs, X_val=None, y_val=None):

        history = {
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(epochs):

            epoch_loss = 0.0
            samples = 0

            self.model.train()

            for X_batch, y_batch in self.dataloader:

                # Forward
                y_pred = self.model.forward(X_batch)

                # Loss
                loss = self.loss_fn.forward(
                    y_pred,
                    y_batch
                )

                # Regularization
                if self.regularizer is not None:

                    loss += self.regularizer.penalty(
                        self.model.regularizable_parameters()
                    )

                # Backward
                loss_grad = self.loss_fn.backward()

                self.model.backward(loss_grad)
                # print("loss_grad shape:", loss_grad.shape)
                # print("first 3 rows:\n", loss_grad[:3])
                # print("min/max:", loss_grad.min(), loss_grad.max())

                gradients = self.model.gradients()

                # Regularization gradients
                if self.regularizer is not None:

                    reg_gradients = self.regularizer.gradients(
                        self.model.regularizable_parameters()
                    )

                    self.model.add_regularization_gradients(
                        reg_gradients
                    )

                    gradients = self.model.gradients()

                # Optimizer
                self.optimizer.step(
                    self.model.parameters(),
                    gradients
                )

                # Loss accumulation
                batch_size = len(X_batch)

                epoch_loss += loss * batch_size
                samples += batch_size

            epoch_loss /= samples

            # Validation
            val_loss = None

            if X_val is not None and y_val is not None:

                val_loss = self.validation_loss(
                    X_val,
                    y_val
                )

            # Scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # History
            history["train_loss"].append(epoch_loss)

            if val_loss is not None:
                history["val_loss"].append(val_loss)

            if val_loss is not None:

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {epoch_loss:.6f} "
                    f"Val Loss: {val_loss:.6f} "
                    f"LR: {self.optimizer.learning_rate:.6f}"
                )

            else:

                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Train Loss: {epoch_loss:.6f} "
                    f"LR: {self.optimizer.learning_rate:.6f}"
                )

        return history

    def validation_loss(self, X, y):

        self.model.eval()

        y_pred = self.model.forward(X)
        loss = self.loss_fn.forward(y_pred,y)
        return loss

    def predict(self, X):
        self.model.eval()
        y_pred = self.model.forward(X)
        return self.convert_predictions(y_pred)


    def convert_predictions(self, y_pred):

        # Binary classification
        if y_pred.shape[1] == 1:
            return (y_pred >= 0.5).astype(int).ravel()

        # Multiclass classification
        else:
            return np.argmax(y_pred,axis=1)

  
    def accuracy(self, X, y):

        predictions = self.predict(X)
        y = y.ravel()
        return np.mean(predictions == y)

    def evaluate(self, X, y):

        self.model.eval()
        y_pred = self.model.forward(X)
        loss = self.loss_fn.forward(y_pred,y)

        predictions = self.convert_predictions(y_pred)

        accuracy = np.mean(predictions == y.ravel())

        return loss, accuracy