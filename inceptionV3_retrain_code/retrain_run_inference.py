# -*- coding: utf-8 -*-

"""
Inception v3 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)

Usage
python retrain_run_inference.py
"""


import numpy as np
import tensorflow as tf
import os, os.path

imageFolderPath = '/tmp/test_image/bird/'      # 추론을 진행할 이미지 경로
modelPath = '/tmp/output_graph.pb'             # 사용할 graph 파일 경로
labelPath = '/tmp/output_labels.txt'           # label 파일 경로

def create_graph():
    """미리 학습된 graph파일을 이용하여 graph를 생성하고 saver를 반환한다."""
    with tf.gfile.FastGFile(modelPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(imagePath):
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer
    else:
        image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
        f = open(labelPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]
        human_string = labels[top_k[0]]
        prob = predictions[top_k[0]]
        print('%s (prob = %.5f)' % (human_string, prob))

        answer = labels[top_k[0]]
        return answer


if __name__ == '__main__':

    fileName = [name for name in os.listdir(imageFolderPath) if os.path.isfile(os.path.join(imageFolderPath, name))]

    create_graph()
    for n in fileName:
        run_inference_on_image(imageFolderPath+n)