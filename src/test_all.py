import os
import numpy as np
import align.detect_face
from scipy import misc
import tensorflow as tf
import facenet
import matplotlib.pyplot as plt
img_path ="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\test\\test3.jpg"

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
print('Creating networks and loading parameters')

# with tf.Graph().as_default():
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#     with sess.as_default():
#         pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
# nrof_samples = len(img_path)
# img_list = []
# count_per_image = []
#
# img = misc.imread(os.path.expanduser(img_path))
# print(img.shape)
# img_size = np.asarray(img.shape)[0:2]
# print(img_size)
# bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
# count_per_image.append(len(bounding_boxes))#检测到的人脸数目
# print(count_per_image)
# print("_________________")
# for j in range(len(bounding_boxes)):
#     #print(bounding_boxes[j, 0:4])
#     det = np.squeeze(bounding_boxes[j, 0:4])
#     #print(det.shape)
#     print(det)
#
#     bb = np.zeros(4, dtype=np.int32)
#     bb[0] = np.maximum(det[0] - 32 / 2, 0)
#     bb[1] = np.maximum(det[1] - 32 / 2, 0)
#     bb[2] = np.minimum(det[2] + 32 / 2, img_size[1])
#     bb[3] = np.minimum(det[3] + 32 / 2, img_size[0])
#     #print(bb[0],bb[1],bb[2],bb[3])
#     cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
#
#     aligned = misc.imresize(cropped, (160, 160), interp='bilinear')
#     # plt.imshow(aligned)
#     # plt.show()
#     prewhitened = facenet.prewhiten(aligned)
#     # plt.imshow(prewhitened)
#     # plt.show()
#     img_list.append(prewhitened)
#
# print(type(img_list))
# images = np.stack(img_list)
# print(type(images))
# print(nrof_samples)
#return images, count_per_image, nrof_samples

# import  pickle
# f = open('C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\new_classifier.pkl','rb')
# data = pickle.load(f)
# print (data)

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i, :, :, :] = img
    return images

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

dataset= "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\lfw1_160"


if 1:
    dataset_tmp = facenet.get_dataset(dataset)
    print(dataset_tmp)
    train_set, test_set = split_dataset(dataset_tmp, 20, 10)
    print(train_set)
    print(test_set)