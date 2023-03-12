# encoding: utf-8

import keras
import argparse
import os
import sys
from config import cfg
from engine.trainer import ModelTrainer
from data import ClsDataGenerator
from data.transforms import build_transforms

def train():
    transforms = build_transforms()
    # load the training data and valid data
    train_data = ClsDataGenerator(cfg.DATASET.TRAIN_PATH, cfg.MODEL.NUM_CLASSES,
                                  is_gray=cfg.INPUT.IS_GRAY, batch_size=cfg.TRAIN.BATCH_SIZE,
                                  transforms=transforms)
    val_data = ClsDataGenerator(cfg.DATASET.TRAIN_PATH, cfg.MODEL.NUM_CLASSES,
                                is_gray=cfg.INPUT.IS_GRAY, batch_size=cfg.TRAIN.BATCH_SIZE,
                                transforms=transforms)

    # build model
    model = ...

    # start training
    trainer = ModelTrainer(model, train_data, val_data)
    trainer.train()
