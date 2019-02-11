import numpy as np
import tensorflow as tf
import time

modelFullPath = '../model/bird,others/output_graph.pb'
labelsFullPath = '../model/bird,others/output_labels.txt'

def create_graph():
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run(image_set):
    for image in image_set:

        start = time.time()

        print("create_graph부분")
        print(time.time()-start)

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            start = time.time()

            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image})

            print("sess_run 부분")
            print(time.time() - start)
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]


            ####이거 아직 해결 안됨!!##################
            #텍스트 파일로 바꿔서 시도해보기
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w) for w in lines]
            # labels = ['birds', 'others',' ',' ',' ']

            # 아래 코드는 refactoring이 필요
            order=1
            for node_id in top_k:
                human_string = labels[node_id]
                order = order+1
                if ("bird" in human_string) and (order == 2) :
                    print("BIRD DETECTED!!!")
                    return 1
    return 0
