# estimator test

from __future__ import print_function
import shutil
import os
import tensorflow as tf
import test.iris_data as iris_data

batch_size = 100
num_step = 2000

model_dir = './log'
# clear the model dir
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.mkdir(model_dir)

def model_fun(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_cloumns'])
    for units in params['hidden_layers']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, units=params['num_classes'])
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    accuracy = tf.metrics.accuracy(labels, predicted_classes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    tf.summary.scalar('accuracy', accuracy[0])

    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.train.get_global_step()
    boundaries = [1000, 1500]
    values = [0.03, 0.01, 0.005]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(mode, train_op=train_op, loss=loss)



def main(unuse_argv):
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()
    MY_FEATURE_COLUMN = [tf.feature_column.numeric_column(_) for _ in train_x.keys()]

    # params are passed to model_fun.
    config = tf.estimator.RunConfig(model_dir=model_dir)
    iris_estimator = tf.estimator.Estimator(model_fn=model_fun,
                                            config=config,
                                            params={'feature_cloumns': MY_FEATURE_COLUMN,
                                                    'hidden_layers': [10, 10],
                                                    'num_classes': 3})
    # when train input dataset repeat(n), n is not empty, the train_and_evaluate will eval every epoch train input dataset run out.
    train_spec = tf.estimator.TrainSpec(input_fn=lambda :iris_data.train_input_fn(train_x, train_y, batch_size), max_steps=num_step)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda : iris_data.eval_input_fn(test_x, test_y, batch_size))
    tf.estimator.train_and_evaluate(iris_estimator, train_spec, eval_spec)

    # #train_and_evaluate finally return the eval_spec result.
    # eval_result = tf.estimator.train_and_evaluate(iris_estimator, train_spec, eval_spec)
    # print(eval_result)

    # Generate predictions from the model
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = iris_estimator.predict(lambda : iris_data.eval_input_fn(predict_x, None, batch_size))
    for pred_dict, expe in zip(predictions, expected):
        tf.logging.debug(pred_dict)
        pred_class = int(pred_dict['class_ids'])
        pred_prob = pred_dict['probabilities'][pred_class]
        tf.logging.info('pred class {} with probability {:.3f}, with expect {}'.format(iris_data.SPECIES[pred_class], pred_prob, expe))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)