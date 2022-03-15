"""Запуск уже тренированной модели"""
import os
import sys

import cv2 as cv  # Импорт модуля OpenCV для компьютерного видения
import numpy as np
from object_detection.builders import model_builder
from object_detection.utils import config_util, label_map_util
from object_detection.utils import visualization_utils as viz_utilz

import tensorflow as tf

APP_TITLE = "Обнаружение маски. Нажмите Q для выхода"
MODEL_PATH = "Tensorflow/workspace/models"
LABEL_MAP_PATH = "Tensorflow/workspace/annotations/label_map.pbtxt"
CHECKPOINT_PATH = MODEL_PATH + "/my_ssd_mobnet/"
CUSTOM_MODEL_NAME = "my_ssd_mobnet"
CONFIG_FILE = MODEL_PATH + "/" + CUSTOM_MODEL_NAME + "/pipeline.config"


def main():
    # Загрузка конфигурации и построение модели обнаружения
    configs = config_util.get_configs_from_pipeline_file(CONFIG_FILE)
    detection_model = model_builder.build(model_config=configs["model"], is_training=False)

    # Восстановление точки сохранения
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, "ckpt-6")).expect_partial()

    @tf.function
    def detect(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

    # Видеозахват веб-камеры. В качестве аргумента указывается ID-устройства, либо путь к медиа-файлу
    camera = cv.VideoCapture(0)
    # # Получение ширины и высоты окна
    # width, height = (
    #     int(camera.get(cv.CAP_PROP_FRAME_WIDTH)),
    #     int(camera.get(cv.CAP_PROP_FRAME_HEIGHT)),
    # )

    while True:
        ret, frame = camera.read()  # Считывание кадров с камеры
        image_arr = np.array(frame)  # Перевод кадров в массив

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_arr, 0), dtype=tf.float32)
        detections = detect(input_tensor)

        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(np.int64)

        label_id_offset = 1  # Для отсчёта классов с единицы, вместо нуля
        image_arr_with_detections = (
            image_arr.copy()
        )  # Копирование оригинального кадра для наложения визуализации

        # Визуализация
        viz_utilz.visualize_boxes_and_labels_on_image_array(
            image_arr_with_detections,
            detections["detection_boxes"],
            detections["detection_classes"] + label_id_offset,
            detections["detection_scores"],
            category_index,
            use_normalized_coordinates=True,  # Установка визуализации поверх объекта, а не в координатах (0, 0)
            max_boxes_to_draw=2,  # Максимальное количество визуализированных квадратов
            line_thickness=2,  # Толщина линий квадрата
            min_score_thresh=0.5,  # Минимальная отметка точности - 50%
            agnostic_mode=False,
        )
        # Трансляция с камеры и установка размера окна
        cv.imshow(APP_TITLE, cv.resize(image_arr_with_detections, (1000, 750)))

        if cv.waitKey(1) == ord("q"):  # Привязка завершения обнаружения на клавишу 'q'
            camera.release()  # Высвобождение камеры
            cv.destroyAllWindows()  # Закрытие окон OpenCV
            break


if __name__ == '__main__':
    main()
    sys.exit()
