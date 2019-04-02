import cv2
import logging

image_wh = (1280, 720)


def record(record_name):
    cap = cv2.VideoCapture(0)
    assert cap.isOpened(), 'Please open the mac camera!'
    out = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'XVID'), 10, image_wh)

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, image_wh)
        out.write(frame)
        cv2.imshow('camera', frame)
        key = cv2.waitKey(30)
        if key & 0xFF == ord(' '):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def show(show_name):
    cap = cv2.VideoCapture(show_name)
    assert cap.isOpened(), 'Please chebck {} exists!'.format(video_name)
    while True:
        ret, frame = cap.read()
        if frame is None:
            logging.info('read video finished.')
            break
        cv2.imshow('camera', frame)
        key = cv2.waitKey(30)
        if key & 0xFF == ord('b'):
            break
        if key & 0xFF == ord(' '):
            cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    video_name = 'pose_angle.avi'
    # record(video_name)
    show(video_name)

    # img = cv2.imread('out_3044.jpg')
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # print cv2.__version__

