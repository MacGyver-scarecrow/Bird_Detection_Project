import numpy as np
import tensorflow as tf

modelFullPath = '../model/output_graph.pb'

def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run(image_set):
    for image in image_set:
        create_graph()

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]

            labels = ['airplanes', 'bird', 'skys', 'trees']

            # 아래 코드는 refactoring이 필요
            order=1
            for node_id in top_k:
                human_string = labels[node_id]
                order = order+1
                if ("bird" in human_string) and (order == 2) :
                    print("BIRD DETECTED!!!")
                    return 1
    return 0