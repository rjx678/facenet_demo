import facenet
import tensorflow as tf
import pickle
from scipy import misc
import align.detect_face
import argparse
import sys
import time
import cv2
import numpy as np

facenet_model_checkpoint =  "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\20180402-114759"
classifier_model = "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\sd160_n_aligned_poly.pkl"
debug = True
"""
定义人脸的一些属性
"""
class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.prob = None
        self.landmarks = None
        self.num = None

"""
人脸注册模块
"""
class Register:
    def __init__(self):
        self.detect = Detection()#人脸检测

    def register(self, image):
        faces = self.detect.find_faces(image)
        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)
        return faces

"""
人脸识别总模块
"""
class Recognition:
    def __init__(self):
        self.detect = Detection()#人脸检测
        self.encoder = Encoder()#特征提取
        self.identifier = Identifier()#人脸识别

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)
        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)
        for i, face in enumerate(faces):
            if debug:
                cv2.imshow("Face: " + str(i), face.image)

            face.embedding = self.encoder.generate_embedding(face)
            face.name,face.prob = self.identifier.identify(face)
            #print(i,face.name,face.prob)
        return faces

"""
人脸识别模块
"""
class Identifier:
    def __init__(self):
        self.best_class_probabilities = None
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            #print(type(face.embedding.shape))#(512,) tuple
            predictions = self.model.predict_proba([face.embedding])
            #print(predictions.shape)#(1,4)
            #print(predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            #print(best_class_indices.shape)#(1,)
            #print(best_class_indices)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            k = 0
            #for i in range(face.num):
            if best_class_probabilities<0.8:
                self.class_names.append("unknown")
                list1 = sorted(set(self.class_names), key=self.class_names.index) #去除list重复的unknown
                #print(list1)
                best_class_indices[0] = len(self.class_names)-1#本身有四个类,index分别对应0,1,2,3,然后增加一个index4,对应unknown
            #print(best_class_probabilities.shape)#(1,)
            #print(best_class_probabilities)
        return self.class_names[best_class_indices[0]],best_class_probabilities

"""
人脸特征提取模块
"""
class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]  # shape is 512

'''
人脸检测\人脸对齐模块
'''
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
        face = Face()
        bounding_boxes, _ = align.detect_face.detect_face(image, self.minsize,
                                                          self.pnet, self.rnet, self.onet,
                                                          self.threshold, self.factor)
        # ####################仿射变换###########################
        # rows, cols, hn = image.shape
        # _new = np.transpose(_)  # (10,2)->(2,10)
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
        #
        # bounding_boxes, _ = align.detect_face.detect_face(dst, self.minsize,
        #                                                   self.pnet, self.rnet, self.onet,
        #                                                   self.threshold, self.factor)

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

        nrof_faces = bounding_boxes.shape[0]
        # if nrof_faces > 0:
        #     det = bounding_boxes[:, 0:4]
        #     det_arr = []
        #     img_size = np.asarray(image.shape)[0:2]
        #     for i in range(nrof_faces):
        #         det_arr.append(np.squeeze(det[i]))
        #
        #     for i, det in enumerate(det_arr):
        #         # face.container_image = dst
        #         face.container_image = image
        #         det = np.squeeze(det)
        #         face.bounding_box = np.zeros(4, dtype=np.int32)
        #
        #         face.bounding_box[0] = np.maximum(det[0] - 32 / 2, 0)
        #         face.bounding_box[1] = np.maximum(det[1] - 32 / 2, 0)
        #         face.bounding_box[2] = np.minimum(det[2] + 32 / 2, img_size[1])
        #         face.bounding_box[3] = np.minimum(det[3] + 32 / 2, img_size[0])
        #         # cropped = dst[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #         cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #         face.image = misc.imresize(cropped, (160, 160), interp='bilinear')
        #         faces.append(face)

        for bb in bounding_boxes:
            face.container_image = image
            face.bounding_box = np.zeros(4, dtype=np.int32)

            img_size = np.asarray(image.shape)[0:2]
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin / 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin / 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin / 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin / 2, img_size[0])
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

            faces.append(face)


        # # 遍历每一个人脸框
        # #for bb in bounding_boxes:
        # for i in range(nrof_faces):
        #     #face.container_image = dst
        #     face.num = nrof_faces
        #     face.container_image = image
        #     face.bounding_box = np.zeros((nrof_faces,4), dtype=np.int32)
        #     #img_size = np.asarray(dst.shape)[0:2]
        #     img_size = np.asarray(image.shape)[0:2]
        #     face.bounding_box[i,0] = np.maximum(bounding_boxes[i,0] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[i,1] = np.maximum(bounding_boxes[i,1] - self.face_crop_margin / 2, 0)
        #     face.bounding_box[i,2] = np.minimum(bounding_boxes[i,2] + self.face_crop_margin / 2, img_size[1])
        #     face.bounding_box[i,3] = np.minimum(bounding_boxes[i,3] + self.face_crop_margin / 2, img_size[0])
        #     #cropped = dst[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        #     cropped = image[face.bounding_box[i,1]:face.bounding_box[i,3], face.bounding_box[i,0]:face.bounding_box[i,2], :]
        #     face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
        #     faces.append(face)
        return faces

def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
                cv2.putText(frame, '{:.3f}'.format(face.prob[0]), (face_bb[0], face_bb[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)
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
                # cv2.putText(frame, "Live?:{}".format(face.isLiveface), (100, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)
    cv2.putText(frame,"Press 's' to save screenshot", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def main(args):
    frame_interval = 3  # Number of frames after which to run face detection之后运行人脸检测的帧数
    fps_display_interval = 5  # seconds  fps显示间隔
    frame_rate = 0
    frame_count = 0

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
        cv2.imshow('frame', frame)
        random_key = np.random.randint(0, high=99999)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\output\\img"+str(random_key)+".png",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.',default=True)
    return parser.parse_args(argv)

#
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



