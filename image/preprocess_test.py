import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data = tf.gfile.FastGFile('./Kyrie_Irving.jpg', 'rb',).read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    img_convert = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    # it seems the random convert methon need [0,255] range date instead of [0, 1.0] range.
    img_convert = tf.image.random_brightness(img_convert, max_delta=0.5)

    img_rotate = tf.contrib.image.rotate(tf.expand_dims(img_convert/255.0, 0), tf.constant([0.5]))*255.0

    batched = tf.expand_dims(img_convert, 0)
    box = tf.constant([[[0.07, 0.45, 0.3, 0.6]]])
    img_box = tf.image.draw_bounding_boxes(images=batched, boxes=box)

    # the bbox_for_draw is another expression of [begin, begin+size],
    # not the bounding box in the new image.
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_convert), bounding_boxes=box, min_object_covered=0.1
    )
    distorted_image = tf.slice(img_convert, begin=begin, size=size)
    comparing_image = tf.image.draw_bounding_boxes(images=tf.expand_dims(img_convert, 0), boxes=bbox_for_draw)

    img_data, img_convert, distorted_image, comparing_image = \
        sess.run([img_data, img_rotate, distorted_image, comparing_image])

    img_list = [img_data, img_convert[0], distorted_image, comparing_image[0]]

    for i in range(len(img_list)):
        plt.subplot(2, 2, i+1)
        plt.imshow(img_list[i])
        plt.title('img {}'.format(i))
    plt.show()

    # img_data = tf.image.convert_image_dtype(resizd, dtype=tf.uint8)
    # encode_image = tf.image.encode_jpeg(img_data)
    # with tf.gfile.GFile('./ouwen', 'wb') as f:
    #     f.write(encode_image.eval())