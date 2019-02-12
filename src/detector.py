import numpy as np
import tensorflow as tf

modelFullPath = '../model/bird,others/output_graph.pb'


def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run(image_set):
    for image in image_set:
        with tf.Session() as sess:
            soft_max_tensor = sess.graph.get_tensor_by_name('final_result:0')

            predictions = sess.run(soft_max_tensor,
                                   {'DecodeJpeg/contents:0': image})

            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]

            labels = ['birds', 'others']

            human_string = labels[top_k[0]]
            if ("bird" in human_string):
                print("BIRD DETECTED!!!")
                return 1
    return 0
