# File: /tune-similarity-pipeline/tune-similarity-pipeline/src/training/callbacks.py

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

def get_callbacks(model_name, log_dir):
    callbacks = [
        ModelCheckpoint(
            filepath=f"{log_dir}/checkpoints/{model_name}_{{epoch:02d}}.h5",
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        TensorBoard(log_dir=log_dir)
    ]
    return callbacks