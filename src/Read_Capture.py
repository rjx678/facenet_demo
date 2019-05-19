import cv2
import os
from scipy import misc
import tensorflow as tf
import align.detect_face
import matplotlib.pyplot as plt
import numpy as np
import math
class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.prob = None
        self.landmarks = None
        self.ishumanface = None
# minsize = 20  # minimum size of face
# threshold = [0.6, 0.7, 0.7]  # three steps's threshold
# factor = 0.709  # scale factor
# gpu_memory_fraction = 0.25

# with tf.Graph().as_default():
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#     with sess.as_default():
#         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
# print('Creating networks and loading parameters')
def search_face(img):
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    # ###################仿射变换###########################
    # rows, cols, hn = img.shape
    # _new = np.transpose(_)  # (10,2)->(2,10)
    # dst = img
    # for i in range(len(_new)):
    #     # print("左眼的位置(%s,%s)" %(_new[i,0],_new[i,5]))
    #     # print("右眼的位置(%s,%s)" %(_new[i,1],_new[i,6]))
    #     eye_center_x = (_new[i, 0] + _new[i, 1]) * 0.5
    #     eye_center_y = (_new[i, 5] + _new[i, 6]) * 0.5
    #     dy = _new[i, 5] - _new[i, 6]
    #     dx = _new[i, 0] - _new[i, 1]
    #     angle = math.atan2(dy, dx) * 180.0 / math.pi + 180.0
    #     # print("旋转角度为%s" % angle)
    #     M = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), angle, 1)
    #     dst = cv2.warpAffine(img, M, (cols, rows))
    # dst = dst
    # ####################################################
    # bounding_boxes, _ = align.detect_face.detect_face(dst, minsize,
    #                                                   pnet, rnet, onet,
    #                                                   threshold, factor)
    nrof_faces = bounding_boxes.shape[0]  # 人脸数目
    #print('找到人脸数目为：{}'.format(nrof_faces))
    crop =None
    for face_position in bounding_boxes:
        random_key = np.random.randint(0, high=999)
        face_position = face_position.astype(int)
        #print(face_position[0:4])
        cv2.rectangle(img, (face_position[0]-16, face_position[1]-16), (face_position[2]+16, face_position[3]+16), (0, 255, 0), 2)
        crop = img[face_position[1]-16:face_position[3]+16,
               face_position[0]-16:face_position[2]+16,: ]
        crop = misc.imresize(crop, (160,160), interp='bilinear')
    crop =crop
        #crop = cv2.resize(crop, (160,160), interpolation=cv2.INTER_CUBIC)
        #misc.imsave(filepath + "\\" + os.path.split(filepath)[1] + "_" + str(random_key) + ".png", crop)

    return nrof_faces,crop
class Register:
    def __init__(self):
        self.detect = Detection()#人脸检测

    def register(self, image):
        faces = self.detect.find_faces(image)

        for i, face in enumerate(faces):
            if True:
                cv2.imshow("Face: " + str(i), face.image)
        return faces
class Detection:
    # face detection parameters
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    def __init__(self, face_crop_size=160, face_crop_margin=32):
        self.pnet, self.rnet, self.onet = self._setup_mtcnn()
        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin

    def _setup_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                return align.detect_face.create_mtcnn(sess, None)

    def find_faces(self, image):
        faces = []
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        # ####################仿射变换###########################
        # rows, cols, hn = image.shape
        # _new = np.transpose(_)  # (10,2)->(2,10)
        # dst = image
        # for i in range(len(_new)):
        #     # print("左眼的位置(%s,%s)" %(_new[i,0],_new[i,5]))
        #     # print("右眼的位置(%s,%s)" %(_new[i,1],_new[i,6]))
        #     eye_center_x = (_new[i, 0] + _new[i, 1]) * 0.5
        #     eye_center_y = (_new[i, 5] + _new[i, 6]) * 0.5
        #     dy = _new[i, 5] - _new[i, 6]
        #     dx = _new[i, 0] - _new[i, 1]
        #     angle = math.atan2(dy, dx) * 180.0 / math.pi + 180.0
        #     #print("旋转角度为%s" % angle)
        #     #print("_____")
        #     M = cv2.getRotationMatrix2D((eye_center_x, eye_center_y), angle, 1)
        #     dst = cv2.warpAffine(image, M, (cols, rows))
        # dst = dst
        #
        # bounding_boxes, _ = align.detect_face.detect_face(dst, self.minsize,
        #                                                   self.pnet, self.rnet, self.onet,
        #                                                   self.threshold, self.factor)

        face = Face()
        #5个关键点赋值到face对象
        _new = np.transpose(_)
        for l in _new:
            face.landmarks = np.zeros(10, dtype=np.int32)
            face.landmarks[0] = l[0]
            face.landmarks[1] = l[1]
            face.landmarks[2] = l[2]
            face.landmarks[3] = l[3]
            face.landmarks[4] = l[4]
            face.landmarks[5] = l[5]
            face.landmarks[6] = l[6]
            face.landmarks[7] = l[7]
            face.landmarks[8] = l[8]
            face.landmarks[9] = l[9]
        # 遍历每一个人脸框
        for bb in bounding_boxes:
            #face.container_image = dst
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)
            #img_size = np.asarray(dst.shape)[0:2]
            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            #cropped = dst[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            faces.append(face)
        return faces

def register_face(filepath):
    cap =cv2.VideoCapture(1)
    face_register = Register()
    count = 0
    while(True):
        random_key = np.random.randint(0, high=999)
        ret , frame =cap.read()
        #frame = cv2.flip(frame, 1, dst=None)
        faces = face_register.register(frame)
        # #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #num,crop= search_face(frame)
        cv2.putText(frame, "Find "+str(len(faces))+" faces", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        thickness=2, lineType=2)
        add_overlays(frame,faces)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            if faces is not None:
                for face in faces:
                    cv2.imwrite(filepath + "\\" + os.path.split(filepath)[1] + "_" + str(random_key) + ".png",face.image)
                    count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.putText(frame, "Press 'q' to quit " , (10, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)
        cv2.putText(frame, "Saved "+str(count)+" imgs", (10, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    thickness=2, lineType=2)

        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        #     misc.imsave(filepath + "\\" + os.path.split(filepath)[1] + "_" + str(random_key) + ".png", crop)
        #     count = count + 1
        # cv2.putText(frame, "Saving "+str(count)+" pics!", (10, 470),
        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
        #         thickness=2, lineType=2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.imshow('frame',frame)
    cap.release()
    cv2.destroyAllWindows()


def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame, ".",(face.landmarks[0], face.landmarks[5]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255),
                        thickness=2, lineType=2)
            cv2.putText(frame, ".",(face.landmarks[1], face.landmarks[6]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255),
                       thickness=2, lineType=2)
            cv2.putText(frame,".", (face.landmarks[2], face.landmarks[7]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255),
                       thickness=2, lineType=2)
            cv2.putText(frame,".", (face.landmarks[3], face.landmarks[8]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255),
                       thickness=2, lineType=2)
            cv2.putText(frame,".", (face.landmarks[4], face.landmarks[9]),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255),
                       thickness=2, lineType=2)
    cv2.putText(frame,"Press 's' to save screenshot", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def main():

    video_capture = cv2.VideoCapture(1)
    face_recognition = Recognition()
    start_time = time.time()
    # print(start_time)
    if args.debug:
        print("Debug enabled")
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        #frame = cv2.flip(frame, 1, dst=None)
        if (frame_count % frame_interval) == 0:
            faces = face_recognition.identify(frame)

            # Check our current fps
            end_time = time.time()
            # print(end_time)
            # print("_______________")
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                # print(frame_count)
                # print(end_time-start_time)
                # print(frame_rate)
                start_time = time.time()
                frame_count = 0

        add_overlays(frame, faces, frame_rate)

        frame_count += 1
        cv2.imshow('Video', frame)

        random_key = np.random.randint(0, high=99999)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\output\\img"+str(random_key)+".png",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
