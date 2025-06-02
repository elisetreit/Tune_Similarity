class Trainer:
    def __init__(self, model, train_dataset, val_dataset, callbacks=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = callbacks if callbacks is not None else []

    def fit(self, epochs, steps_per_epoch, validation_steps):
        history = self.model.fit(
            self.train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.val_dataset,
            validation_steps=validation_steps,
            callbacks=self.callbacks
        )
        return history

    def evaluate(self, test_dataset):
        evaluation_results = self.model.evaluate(test_dataset)
        return evaluation_results

    def save_model(self, filepath):
        self.model.save(filepath)