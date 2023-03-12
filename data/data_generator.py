import os
import cv2
import keras
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Union, Callable


class DataGenerator:
    def __init__(self):
        pass

    def generate(self):
        return

    def __iter__(self):
        return self.generate()

    def __next__(self):
        return self.generate()

class ClsDataGenerator:
    """
    This data generate is used to image classification
    """
    def __init__(self, data_root, num_classes, batch_size=32,
                 is_gray=True, transforms: Callable = None):
        self.transforms = transforms
        self.img_flags = cv2.IMREAD_GRAYSCALE if is_gray else cv2.IMREAD_UNCHANGED
        self.batch_size = batch_size
        self.img_map_cate = self.get_paths(data_root)
        self.num_classes = num_classes

    def get_paths(self, data_root):
        """ Return image paths that are used to load current data, and its category.

        The dataset's directory as follow:
        data
        |
        |--- dog
        |   |---*.jpg(image)
        |--- cat
        |   |---*.jpg(image)
        |...

        Returns: List[Tuple(img_path: str, category_id: int)]
        """
        # path = ""
        # # ps. 建议将类别cat转为整数
        # cat = "1"
        # self.img_map_cate = [(path, cat)]
        raise NotImplementedError

    def get_img_label(self, img_path, cate, *args, **kwargs):
        # img_path, cat = random.choice(self.img_map_cate)
        img = cv2.imread(img_path, self.img_flags)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, cate

    def generate(self):
        selected_sample = random.sample(self.img_map_cate, self.batch_size)
        with ThreadPoolExecutor(max_workers=40) as t:
            obj_list = [t.submit(self.get_img_label, img_path, cate)
                        for img_path, cate in selected_sample]
            data = [future.result() for future in as_completed(obj_list)]

        X, y = zip(*data)
        X = np.stack(X)
        if len(X.shape) == 3:
            X = [..., None]
        # 归一化
        X /= 255.
        y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        return X, y

    def __iter__(self):
        return self.generate()

    def __next__(self):
        return self.generate()
