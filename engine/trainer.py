# encoding: utf-8
"""
modify from https://github.com/Ahmkel/Keras-Project-Template/blob/master/trainers/simple_mnist_trainer.py
"""
import os
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.models import Model
from config import cfg
from utils.history import save_history


class ModelTrainer:
    def __init__(self, model, train_data, val_data):
        self.callbacks = []
        self.model : Model= model
        self.train_data = train_data
        self.val_data = val_data
        self.init_callbacks()

    def init_callbacks(self):
        self.callbacks.append(
            ModelCheckpoint(
                filepath=cfg.CHECKPOINTS.FORMAT,
                monitor='val_loss',
                save_best_only=cfg.CHECKPOINTS.SAVE_BEST_ONLY,
                verbose=1,
            )
        )
        self.callbacks.append(
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
        )

    def train(self):
        history = self.model.fit_generator(
            self.train_data, steps_per_epoch=cfg.TRAIN.TRAIN_STEPS,
            epochs=cfg.TRAIN.EPOCHS,
            validation_data=self.val_data, validation_steps=cfg.TRAIN.VAL_STEPS,
            workers=cfg.TRAIN.WORKERS, use_multiprocessing=cfg.TRAIN.WORKERS not in [0, 1, -1]
        )
        if cfg.TRAIN.SAVE_HISTORY:
            # save training history
            save_history(history)
