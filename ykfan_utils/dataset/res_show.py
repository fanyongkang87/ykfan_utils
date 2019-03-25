import cv2
import os

img_dir = '/Users/yongkangfan/Downloads'
count = 0
for img_name in os.listdir(img_dir):
    if img_name.split('.')[1] != 'png':
        continue
    img_path = '{}/{}'.format(img_dir, img_name)
    print(img_path)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640*2, 640))
    cv2.imshow('img res', img)

    output_name = 'output_{}.png'.format(count)
    cv2.imwrite('{}/{}'.format(img_dir, output_name), img)
    count += 1
    cv2.waitKey(0)
