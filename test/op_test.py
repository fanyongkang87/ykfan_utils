# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
# import tensorflow as tf

# import tensorflow as tf
# # test reuce_mean
# xx = tf.constant([[[1., 1, 1, 1],
#                    [2., 2, 2, 2],
#                    [3., 3, 3, 3]],
#
#                   [[4, 4, 4, 4],
#                    [5, 5, 5, 5],
#                    [6, 6, 6, 6]]])
# m3 = tf.reduce_mean(xx, axis=(1, 2), keepdims=True) # [2.5 2.5 2.5]
# m4 = tf.multiply(xx, m3)
#
# b_val = tf.constant([True, False, False, True])
# f_val = tf.to_float(b_val)
#
# # with tf.control_dependencies([tf.assert_equal(f_val, tf.constant([1.0, 0.0, 0.0, 1.0]))]):
# #     i_val = tf.to_int32(f_val)
# with tf.control_dependencies([tf.Print(f_val, [f_val], message='print f_val ')]):
#     i_val = tf.to_int32(f_val)
#
# add_val = i_val + 2
# # seem tensorflow assert dont need to run.
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(xx.get_shape())
#     print(m3.get_shape())
#     print(sess.run(m3))
#     print(sess.run(m4))
#     # print(sess.run(add_val))
#     # for op in tf.get_default_graph().get_operations():
#     #     print op



# def tf_add_one(tf_val):
#     b = tf.constant([1.0], name='add_one')
#     return tf_val + b
#
# with tf.variable_scope('s1'):
#     input_val = tf.constant([2.0], name='input')
#     a = tf.get_variable('plus_val', [1])
#     val = input_val*a
#     with tf.variable_scope('s2', reuse=True):
#         val = tf_add_one(val)
#         c = tf.get_variable('plus_val', [1], )
#         val = val*c
#
# for op in tf.get_default_graph().get_operations():
#     print op


# # gpu test
# import tensorflow as tf
#
# c = []
# with tf.device('/cpu:0'):
#     # device name not int the op name or scope.
#     a = tf.get_variable("a", [2, 2], initializer=tf.random_uniform_initializer(-1, 1))
#     b = tf.get_variable("b", [2, 2], initializer=tf.random_uniform_initializer(-1, 1))
#
# with tf.device('/gpu:0'):
#     c.append(tf.matmul(a, b))
#
# with tf.device('/gpu:1'):
#     c.append(a + b)
#
# with tf.device('/cpu:0'):
#     sum = tf.add_n(c)
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))
#
# init = tf.global_variables_initializer()
# sess.run(init)
#
# print(sess.run(sum))
#
# #operation and variable different.
# for op in tf.get_default_graph().get_operations():
#     print op
# for var in tf.get_collection(tf.GraphKeys.VARIABLES):
#     print var


# # name scope and variable scope
# import tensorflow as tf
#
# with tf.variable_scope('a'):
#     with tf.name_scope('b'):
#         with tf.variable_scope('c'):
#             # variable_scope work on get_variable
#             val = tf.get_variable('var', shape=[1], initializer=tf.zeros_initializer)
#             # variable_scop and name_scope work on tf.Variable
#             n = tf.Variable([2.0], name='n')
#             # variable_scop and name_scope work on op tensor.
#             d = val+n
# print val
# print n
# print d


# # test the wing loss
# import tensorflow as tf
# import math
# tf.enable_eager_execution()
#
# w = 10.0
# epsilon = 2.0
# label = tf.constant([1, 0, -2, -2])
# landmark_pred = tf.constant([
#     [0.49, 0.26, 0.45, 0.52],
#     [0.51, 0.24, 0.46, 0.52],
#     [0.36, 0.43, 0.47, 0.51],
#     [0.46, 0.22, 0.53, 0.55],
# ])
#
# lanmark_target = tf.constant([
#     [0.5, 0.25, 0.5, 0.5],
#     [0.5, 0.25, 0.5, 0.5],
#     [0.5, 0.25, 0.5, 0.5],
#     [0.5, 0.25, 0.5, 1.0],
# ])
#
# with tf.name_scope('wing_loss'):
#     ones = tf.ones_like(label, dtype=tf.float32)
#     zeros = tf.zeros_like(label, dtype=tf.float32)
#     valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
#
#     c = w * (1.0 - math.log(1.0 + w / epsilon))
#     absolute_x = tf.abs(landmark_pred - lanmark_target)
#     losses = tf.where(
#         tf.greater(w, absolute_x),
#         w * tf.log(1.0 + absolute_x / epsilon),
#         absolute_x - c
#     )
#     losses = tf.reduce_sum(losses, axis=1)
#     num_valid = tf.reduce_sum(valid_inds)
#     keep_num = tf.cast(num_valid, dtype=tf.int32)
#     losses = losses * valid_inds
#     _, k_index = tf.nn.top_k(losses, k=keep_num)
#     losses = tf.gather(losses, k_index)
#     loss = tf.reduce_mean(losses)


# # test tf.gradients
# import tensorflow as tf
# with tf.name_scope('scope_1'):
#     a = tf.Variable(1.0, name='a')
# with tf.name_scope('scope_2'):
#     b = tf.Variable(1.0, name='b')
# c = a + b
# d = a + b
#
# loss = c + d
# gradients = tf.gradients(loss, [a, b])
# gradients_with_stop = tf.gradients(loss, [a, b], stop_gradients=[d])
# init = tf.global_variables_initializer()
#
# opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# # print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#
# selected_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='scope_1')
# selected_opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss, var_list=selected_var_list)
#
# with tf.Session() as sess:
#     sess.run(init)
#     print sess.run([gradients])
#     print sess.run([gradients_with_stop])
#     print sess.run([opt, a, b])
#     sess.run(init)
#     print sess.run([selected_opt, a, b])


# #test operstion and tensor
# import tensorflow as tf
# with tf.name_scope('n1'):
#     a = tf.Variable(1.0, name='a')
#     b = tf.Variable(1.0, name='b')
#     c = tf.Variable(1.0, name='c')
#
#     d = a+b
#     e = tf.add(b, c, name='e')
#
#     f = d+e
#
# for op in tf.get_default_graph().get_operations():
#     print op.name
# # tensor.name = op.name + ':0'
# print e

# import tensorflow as tf
#
# a = tf.get_variable('a', shape =[], initializer=tf.zeros_initializer, dtype=tf.int32)
# b = tf.get_variable('b', shape =[], initializer=tf.zeros_initializer, dtype=tf.int32)
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(5):
#         debug_val = tf.assign(b, a)
#         with tf.control_dependencies([debug_val]):
#             a = b+1
#         print sess.run([a, b])


# import tensorflow as tf
#
# h = tf.Variable(48, dtype=tf.int32)
# x = tf.constant(128, dtype=tf.int32)
# new_h = x * tf.to_int32(tf.ceil(tf.to_float(h)/tf.to_float(x)))
#
# # h = tf.Variable(48, dtype=tf.int32)
# # x = 128
# # new_h = x * tf.to_int32(tf.ceil(h / x))
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(sess.run(new_h))
#
# import tensorflow as tf
# import copy
# from tensorflow.core.framework import graph_pb2
#
# def get_add_graph():
#     with tf.Graph().as_default() as add_graph:
#         a = tf.placeholder(tf.float32, (), name='a')
#         b = tf.Variable([2.0], trainable=False, name='b')
#         c = tf.add(a, b, name='c')
#
#         for op in add_graph.get_operations():
#             print(op.name)
#
#         for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#             print(var)
#
#         init = tf.global_variables_initializer()
#         with tf.Session(graph=add_graph) as sess:
#             sess.run(init)
#             print('\n', sess.run([c], feed_dict={'a:0': 1.0}), '\n')
#     return add_graph
#
# def get_plus_grpah():
#     with tf.Graph().as_default() as plus_graph:
#         d = tf.placeholder(tf.float32, (), name='d')
#         e = tf.Variable([2.0], trainable=False, name='e')
#         f = tf.multiply(d, e, name='f')
#     return plus_graph
#
# def combineGraph(add_graph, plus_graph):
#
#     out = graph_pb2.GraphDef()
#     for node in add_graph.node:
#         out.node.extend([copy.deepcopy(node)])
#
#     for node in plus_graph.node:
#         if node.name == 'd':
#             continue
#         idx = 0
#         for x in node.input:
#              if x == 'd':
#                  #idx = node.input.index(x)
#                  node.input.remove(x)
#                  node.input.insert(idx, 'c')
#                  break
#              idx = idx+1
#         out.node.extend([copy.deepcopy(node)])
#
#     out.library.CopyFrom(add_graph.library)
#     out.versions.CopyFrom(add_graph.versions)
#     return out
#
# # add_graph = get_add_graph()
# # add_graph_def = add_graph.as_graph_def().node
# # for node in add_graph_def:
# #     print node
#
# combine_graph = combineGraph(get_add_graph().as_graph_def(), get_plus_grpah().as_graph_def())
# with tf.Graph().as_default() as run_graph:
#     tf.import_graph_def(combine_graph)
#     for op in run_graph.get_operations():
#         print(op.name)
#
#     var_b = run_graph.get_tensor_by_name('import/b:0')
#     var_d = run_graph.get_tensor_by_name('import/e:0')
#
#     tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_b)
#     tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, var_d)
#
#     for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
#         print(var)
#
#
#     res_tensor_c = run_graph.get_tensor_by_name('import/c:0')
#     res_tensor_f = run_graph.get_tensor_by_name('import/f:0')
#
#
#     init = tf.global_variables_initializer()
#     with tf.Session(graph=run_graph) as sess:
#         sess.run(iniy = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))
#
# with tf.Session() as sess:
#   # Define options for the `sess.run()` call.
#   options = tf.RunOptions()
#   options.output_partition_graphs = True
#   options.trace_level = tf.RunOptions.FULL_TRACE
#
#   # Define a container for the returned metadata.
#   metadata = tf.RunMetadata()
#
#   sess.run(y, options=options, run_metadata=metadata)
#
#   # Print the subgraphs that executed on each device.
#   print(metadata.partition_graphs)
#
#   # Print the timings of each operation that executed.
#   print(metadata.step_stats)t)
#         print(sess.run(res_tensor_c, feed_dict={'import/a:0': 1.0}))


# import tensorflow as tf
#
# a = tf.constant([1.0, 2.0])
# print_op = tf.print(a)
# with tf.control_dependencies([print_op]):
#     b = a + 1
#
# with tf.Session() as sess:
#     print(sess.run([b]))


import tensorflow as tf
y = tf.matmul([[37.0, -23.0], [1.0, 4.0]], tf.random_uniform([2, 2]))

with tf.Session() as sess:
  # Define options for the `sess.run()` call.
  options = tf.RunOptions()
  options.output_partition_graphs = True
  options.trace_level = tf.RunOptions.FULL_TRACE

  # Define a container for the returned metadata.
  metadata = tf.RunMetadata()

  sess.run(y, options=options, run_metadata=metadata)

  # Print the subgraphs that executed on each device.
  print(metadata.partition_graphs)

  # Print the timings of each operation that executed.
  print(metadata.step_stats)