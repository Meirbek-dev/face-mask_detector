from os.path import join
from sys import exit
from tkinter import Tk
from tkinter.messagebox import askyesno

import cv2
import tensorflow as tf
from numpy import array, expand_dims, int64
from object_detection.builders.model_builder import build
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection.utils.label_map_util import create_category_index_from_labelmap
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

MODEL_PATH = "Tensorflow/workspace/models"
LABEL_MAP_PATH = "Tensorflow/workspace/annotations/label_map.pbtxt"
CHECKPOINT_PATH = MODEL_PATH + "/my_ssd_mobnet/"
CUSTOM_MODEL_NAME = "my_ssd_mobnet"
CONFIG_FILE = MODEL_PATH + "/" + CUSTOM_MODEL_NAME + "/pipeline.config"


def main():
    # Загрузка конфигурации и построение модели обнаружения
    configs = get_configs_from_pipeline_file(CONFIG_FILE)
    detection_model = build(model_config=configs["model"], is_training=False)

    # Восстановление точки сохранения
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(join(CHECKPOINT_PATH, "ckpt-11")).expect_partial()

    @tf.function
    def detect(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = create_category_index_from_labelmap(LABEL_MAP_PATH)

    # Видеозахват веб-камеры. В качестве аргумента указывается ID-устройства, либо путь к медиа-файлу
    camera = cv2.VideoCapture(0)
    # # Получение ширины и высоты окна
    # width, height = (
    #     int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #     int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    # )
    root = Tk()
    root.withdraw()

    while camera.isOpened():
        # Считывание кадров с камеры
        _, frame = camera.read()
        # Перевод кадров в массив
        image_arr = array(frame)
        # Перевод массива в тензоры
        input_tensor = tf.convert_to_tensor(expand_dims(image_arr, 0), dtype=tf.float32)
        detections = detect(input_tensor)
        num_detections = int(detections.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy() for key, value in detections.items()
        }
        detections["num_detections"] = num_detections
        detections["detection_classes"] = detections["detection_classes"].astype(int64)

        # Для отсчёта индекса классов с единицы, вместо нуля
        label_id_offset = 1
        # Копирование оригинального кадра для наложения визуализации
        image_arr_with_detections = image_arr.copy()

        # Визуализация
        visualize_boxes_and_labels_on_image_array(
            image_arr_with_detections,
            detections["detection_boxes"],
            detections["detection_classes"] + label_id_offset,
            detections["detection_scores"],
            category_index,
            # Установка визуализации поверх объекта обнаружения, а не в координатах (0, 0)
            use_normalized_coordinates=True,
            # Максимальное количество визуализированных полей
            max_boxes_to_draw=1,
            # Толщина линий поля
            line_thickness=2,
            # Минимальная отметка точности - 50%
            min_score_thresh=0.5
            #         agnostic_mode=False
        )

        # Трансляция с камеры и установка размера окна
        cv2.imshow(
            'Обнаружение маски. Для выхода нажмите "q"',
            cv2.resize(image_arr_with_detections, (1000, 750))
        )
        # Проверка на наличие маски
        no_mask = (detections["detection_classes"][0])
        if no_mask:
            prompt = askyesno(
                'Предупреждение',
                'Наденьте маску и нажмите "Yes", чтобы продолжить\nНажмите "No", чтобы выйти'
            )
            if prompt:
                continue
            elif not prompt:
                break
        # Привязка завершения цикла обнаружения на клавишу 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    root.quit()
    # Высвобождение камеры
    camera.release()
    # Закрытие окон OpenCV
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    exit()
