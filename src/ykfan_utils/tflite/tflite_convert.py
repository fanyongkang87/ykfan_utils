import tensorflow as tf
import tensorflow.contrib.lite as lite
from tensorflow.python.framework import graph_util
# from train_models.mtcnn_model_v3 import P_Net_V3, R_Net_V3


def net_freeze(net_factory, model_path, net_name, output_array, tflite_name):
    image_size = 48
    if net_name == "P_Net":
        image_size = 24

    graph = tf.Graph()
    with graph.as_default():
        # define tensor and op in graph(-1,1)
        image_op = tf.placeholder(tf.float32, (1, image_size, image_size, 3), name='input_image')
        print(image_op)

        cls_prob, bbox_pred, _ = net_factory(image_op, training=False)
        for op in tf.get_default_graph().get_operations():
            print(op.name)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            # check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            print(model_path)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert readstate, "the params dictionary is not valid"
            print("restore models' param")
            saver.restore(sess, model_path)


            # output_graph_def = graph_util.convert_variables_to_constants(
            #     sess,
            #     graph.as_graph_def(), output_array # We split on comma for convenience
            # )
            # # Finally we serialize and dump the output graph to the filesystem
            # with tf.gfile.GFile('{}_froze.pb'.format(net_name), "wb") as f:
            #     f.write(output_graph_def.SerializeToString())
            # print("%d ops in the final graph." % len(output_graph_def.node))

            input_tensor = [graph.get_tensor_by_name('input_image:0')]
            output_tensor = [graph.get_tensor_by_name(output_name+':0') for output_name in output_array]

            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(), output_array # We split on comma for convenience
            )

            model_f = tf.gfile.GFile(tflite_name, "wb")
            model_f.write(output_graph_def.SerializeToString())

            # tflite_model = tf.contrib.lite.toco_convert(output_graph_def, input_tensor, output_tensor)

            # converter = tf.contrib.lite.TocoConverter.from_session(sess, input_tensor, output_tensor)
            # tflite_model = converter.convert()

            # input_tensor = ['input_image']
            # output_tensor = [output_name for output_name in output_array]
            # converter = tf.contrib.lite.TocoConverter.from_saved_model(pnet_path, input_tensor, output_tensor)
            # tflite_model = converter.convert()

            # Converting a GraphDef from session.
            # converter = lite.TFLiteConverter.from_session(sess, input_tensor, output_tensor)
            # tflite_model = converter.convert()

            # print 'save tflite {}'.format(tflite_name)
            # open(tflite_name, "wb").write(tflite_model)

version ='v36'
pnet_path = '/Users/yongkangfan/Documents/DL/mtcnn_v3/data/MTCNN_model_v36/PNet_landmark/PNet-12'
rnet_path = '/Users/yongkangfan/Documents/DL/mtcnn_v3/data/MTCNN_model_v36/RNet_landmark/RNet-12'
onet_path = '../../data/MTCNN_model_{}/ONet_landmark/ONet-30'.format(version)

if __name__ == '__main__':
    pass
    # [-1.1272619 - 0.23438178  0.56261474 - 0.03256138  0.79474616 - 0.07583803 0.6205427 - 0.22639471]
    # net_freeze(P_Net_V2, pnet_path, 'P_Net', ['P_Net/conv1/BiasAdd'], "p_net_0903.tflite")

    # [-0.3766677   0.17552039 - 0.0210284   0.02764815  0.8198816   0.06546781 1.713414 - 0.29070193]
    # net_freeze(P_Net_V2, pnet_path, 'P_Net', ['P_Net/PReLU1/add'], "p_net_0903.tflite")


    # net_freeze(P_Net_V3, pnet_path, 'P_Net', ['P_Net/conv5-1/conv2d/BiasAdd','P_Net/conv5-2/conv2d/BiasAdd'], "p_net_{}.pb".format(version))
    # net_freeze(R_Net_V3, rnet_path, 'R_Net', ['R_Net/conv5-1/conv2d/BiasAdd','R_Net/conv5-2/conv2d/BiasAdd','R_Net/conv5-3/conv2d/BiasAdd'], "r_net_{}.pb".format(version))
