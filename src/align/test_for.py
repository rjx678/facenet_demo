import sys
import os
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep

def main():
    sleep(random.random())
    # 如果还没有输出文件夹，则创建
    # 设置对齐后的人脸图像存放的路径
    out ="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\lfw1_160"
    output_dir = os.path.expanduser(out)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 在日志目录的文本文件中存储一些Git修订信息
    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    # 在output_dir文件夹下创建revision_info.txt文件，里面存的是执行该命令时的参数信息
    # 当前使用的tensorflow版本，git hash,git diff
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    # 获取数据集下所有人名和其人名目录下是所有图片，
    # 放到ImageClass类中，再将类存到dataset列表里
    input ="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\lfw1"
    dataset = facenet.get_dataset(input)

    #print('Creating networks and loading parameters')
    '''2、建立MTCNN网络，并预训练(即使用训练好的网络初始化参数)'''
    with tf.Graph().as_default():
        # 设置Session的GPU参数，每条线程分配多少显存
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            # 获取P-Net，R-Net，O-Net网络
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20  # minimum size of face最小尺寸
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold阈值
    factor = 0.709  # scale factor 比例因子

    # Add a random key to the filename to allow alignment using multiple processes

    # 获取一个随机数，用于创建下面的文件名
    random_key = np.random.randint(0, high=99999)
    # 将图片和求得的相应的Bbox保存到bounding_boxes_XXXXX.txt文件里
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
    '''3、每个图片中人脸所在的边界框写入记录文件中'''
    with open(bounding_boxes_filename, "w") as text_file:
        # 处理图片的总数量
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        # 是否对所有图片进行洗牌
        if True:
            random.shuffle(dataset)
        # 获取每一个人，以及对应的所有图片的绝对路径
        for cls in dataset:
            # 每一个人对应的输出文件夹
            output_class_dir = os.path.join(output_dir, cls.name)
            print(output_class_dir)
            # 如果目的文件夹里还没有相应的人名的文件夹，则创建相应文件夹
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if True:
                    random.shuffle(cls.image_paths)
            # 遍历每一张图片
            print(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                # 对齐后的图片文件名
                print(nrof_images_total)
                print(os.path.split(image_path))
                print(os.path.split(image_path)[1])
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                print(filename)
                print("---5")
                output_filename = os.path.join(output_class_dir, filename + '.png')
                print(output_filename)

                print("________")
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        # 读取图片文件
                        img = misc.imread(image_path)
                        plt.imshow(img)

                        plt.show()
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim < 2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:, :, 0:3]
                        plt.imshow(img)
                        #plt.savefig("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\output\\output22.jpg")
                        plt.show()
                        # 检测人脸，bounding_boxes可能包含多张人脸框数据，
                        # 一张人脸框有5个数据，第一和第二个数据表示框左上角坐标，第三个第四个数据表示框右下角坐标，
                        # 最后一个数据应该是可信度

                        # 人脸检测 bounding_boxes：表示边界框 形状为[n,5] 5对应x1,y1,x2,y2,score
                        # _：人脸关键点坐标 形状为 [n,10]
                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                                          factor)
                        # 获得的人脸数量（#边界框个数）
                        nrof_faces = bounding_boxes.shape[0]
                        print("________________________________")
                        print("人脸数:")
                        print(nrof_faces)
                        if nrof_faces > 0:
                            # [n,4] 人脸框
                            det = bounding_boxes[:, 0:4]
                            print("人脸框:")
                            print(det)
                            # 保存所有人脸框
                            det_arr = []
                            # 原图片大小
                            img_size = np.asarray(img.shape)[0:2]
                            print("原图片大小:")
                            print(img_size)
                            if nrof_faces > 1:
                                # 一张图片中检测多个人脸
                                if False:
                                    # 如果要检测多张人脸的话
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    # 即使有多张人脸，也只要一张人脸就够了
                                    # 获取人脸框的大小
                                    print(det[:,0])
                                    print(det[:,1])
                                    print(det[:,2])
                                    print(det[:,3])
                                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                                    print(bounding_box_size)
                                    # 原图片中心坐标
                                    img_center = img_size / 2
                                    print("原图片重心坐标")
                                    print(img_center)
                                    # 求人脸框中心点相对于图片中心点的偏移，
                                    # (det[:,0]+det[:,2])/2和(det[:,1]+det[:,3])/2组成的坐标其实就是人脸框中心点
                                    offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    print("_____________")
                                    print([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                         (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                                    print("offsets的值:")
                                    print(offsets)
                                    # 求人脸框中心到图片中心偏移的平方和
                                    # 假设offsets=[[   4.20016056  145.02849352 -134.53862838] [ -22.14250919  -26.74770141  -30.76835772]]
                                    # 则offset_dist_squared=[  507.93206189 21748.70346425 19047.33436466]
                                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                                    print("offsets_dist的值:")
                                    print(offset_dist_squared)
                                    # 用人脸框像素大小减去偏移平方和的两倍，得到的结果哪个大就选哪个人脸框
                                    # 其实就是综合考虑了人脸框的位置和大小，优先选择框大，又靠近图片中心的人脸框
                                    index = np.argmax(
                                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                                    det_arr.append(det[index, :])
                                    print("det_arr的值")
                                    print(det_arr)
                            else:
                                # 只有一个人脸框的话，那就没得选了
                                det_arr.append(np.squeeze(det))
                            # 遍历每一个人脸框
                            for i, det in enumerate(det_arr):
                                # [4,]  边界框扩大margin区域，并进行裁切
                                print("进入遍历每一个人脸")
                                print(det)
                                print("____________")
                                det = np.squeeze(det)
                                print(det)
                                bb = np.zeros(4, dtype=np.int32)
                                print(bb)
                                # 边界框周围的裁剪边缘，就是我们这里要裁剪的人脸框要比MTCNN获取的人脸框大一点，
                                # 至于大多少，就由margin参数决定了
                                bb[0] = np.maximum(det[0] - 32 / 2, 0)
                                bb[1] = np.maximum(det[1] - 32 / 2, 0)
                                bb[2] = np.minimum(det[2] + 32 / 2, img_size[1])
                                bb[3] = np.minimum(det[3] + 32 / 2, img_size[0])
                                print(bb[0])
                                print(bb[1])
                                print(bb[2])
                                print(bb[3])
                                # 裁剪人脸框，再缩放
                                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                                plt.imshow(cropped)
                                plt.show()
                                # 缩放到指定大小，并保存图片，以及边界框位置信息
                                scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)  # 分离文件名和扩展名
                                if False:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                # 保存图片
                                misc.imsave(output_filename_n, scaled)
                                # 记录信息到bounding_boxes_XXXXX.txt文件里
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

                    print('Total number of images: %d' % nrof_images_total)
                    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
#main()
import cv2
import math
from scipy import misc
import matplotlib.pyplot as plt

with tf.Graph().as_default():
    # 设置Session的GPU参数，每条线程分配多少显存
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        # 获取P-Net，R-Net，O-Net网络
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

minsize = 20  # minimum size of face最小尺寸
threshold = [0.6, 0.7, 0.7]  # three steps's threshold阈值
factor = 0.709  # scale factor 比例因子

#fig = plt.figure(1)

image_path ="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\test\\test55.jpg"
#image_path ="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data\\ym\\ym_8.jpg"
image = misc.imread(image_path)
#plt.imshow(image)
#plt.figure(2)
#image = misc.imresize(image,0.8)
#plt.imshow(image)
#plt.imsave("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\test\\test55.jpg",image)
#print(image.shape)

rows,cols, hn = image.shape
bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
print(bounding_boxes)
print(bounding_boxes[0][4])
_new = np.transpose(_) #(10,2)->(2,10)
print(len(_new))
print(_new)
for i in range(len(_new)):
    print("左眼的位置(%s,%s)" %(_new[i,0],_new[i,5]))
    print("右眼的位置(%s,%s)" %(_new[i,1], _new[i,6]))
    eye_center_x =(_new[i,0]+_new[i,1])*0.5
    eye_center_y = (_new[i,5] + _new[i,6]) * 0.5
    dy = _new[i,5] - _new[i,6]
    dx = _new[i,0] - _new[i,1]
    plt.figure(3)
    plt.imshow(image)
    angle = math.atan2(dy, dx) * 180.0 / math.pi + 180.0  # (-180,180)->(0,360),<180,逆时针旋转 >180顺时针
    print("旋转角度为%s" % angle)
    M =cv2.getRotationMatrix2D((eye_center_x,eye_center_y),angle,1)
    dst =cv2.warpAffine(image,M,(cols,rows),borderValue=(255,255,255))
    #print(dst.shape)
    plt.figure(4)
    plt.imshow(dst)

bounding_boxes, _ = align.detect_face.detect_face(dst, minsize, pnet, rnet, onet, threshold, factor)
print(bounding_boxes)
print(bounding_boxes[0][4])
_new = np.transpose(_)  # (10,2)->(2,10)
print(len(_new))
print(_new)
for i in range(len(_new)):
    print("左眼的位置(%s,%s)" % (_new[i, 0], _new[i, 5]))
    print("右眼的位置(%s,%s)" % (_new[i, 1], _new[i, 6]))
    # if _new[i, 5]-_new[i,6]
    eye_center_x = (_new[i, 0] + _new[i, 1]) * 0.5
    eye_center_y = (_new[i, 5] + _new[i, 6]) * 0.5
    dy = _new[i, 5] - _new[i, 6]
    dx = _new[i, 0] - _new[i, 1]
    cv2.putText(dst, ".", (_new[i,0], _new[i,5]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),
                thickness=5, lineType=2)
    cv2.putText(dst, ".", (_new[i,1], _new[i,6]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),
                thickness=5, lineType=2)
    cv2.putText(dst, ".", (_new[i,2], _new[i,7]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),
                thickness=5, lineType=2)
    cv2.putText(dst, ".", (_new[i,3], _new[i,8]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),
                thickness=5, lineType=2)
    cv2.putText(dst, ".", (_new[i,4], _new[i,9]),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),
                thickness=5, lineType=2)
    plt.figure(5)
    plt.imshow(dst)
    #plt.show()

#获得的人脸数量（#边界框个数）
nrof_faces = bounding_boxes.shape[0]
if nrof_faces>0:
    # [n,4] 人脸框
    det = bounding_boxes[:,0:4]
    # 保存所有人脸框
    det_arr = []
    # 原图片大小
    img_size = np.asarray(dst.shape)[0:2]
    #img_size = np.asarray(img.shape)[0:2]
    if nrof_faces>1:
        # 一张图片中检测多个人脸
        if False:
            # 如果要检测多张人脸的话
            for i in range(nrof_faces):
                det_arr.append(np.squeeze(det[i]))
        else:
            # 即使有多张人脸，也只要一张人脸就够了
            # 获取人脸框的大小
            bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
            # 原图片中心坐标
            img_center = img_size / 2
            # 求人脸框中心点相对于图片中心点的偏移，
            # (det[:,0]+det[:,2])/2和(det[:,1]+det[:,3])/2组成的坐标其实就是人脸框中心点
            offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
            # 求人脸框中心到图片中心偏移的平方和
            # 假设offsets=[[   4.20016056  145.02849352 -134.53862838] [ -22.14250919  -26.74770141  -30.76835772]]
            # 则offset_dist_squared=[  507.93206189 21748.70346425 19047.33436466]
            offset_dist_squared = np.sum(np.power(offsets,2.0),0)
            # 用人脸框像素大小减去偏移平方和的两倍，得到的结果哪个大就选哪个人脸框
            # 其实就是综合考虑了人脸框的位置和大小，优先选择框大，又靠近图片中心的人脸框
            index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
            det_arr.append(det[index,:])
    else:
        # 只有一个人脸框的话，那就没得选了
        det_arr.append(np.squeeze(det))
    # 遍历每一个人脸框
    for i, det in enumerate(det_arr):
        # [4,]  边界框扩大margin区域，并进行裁切
        det = np.squeeze(det)
        bb = np.zeros(4, dtype=np.int32)
        # 边界框周围的裁剪边缘，就是我们这里要裁剪的人脸框要比MTCNN获取的人脸框大一点，
        # 至于大多少，就由margin参数决定了
        bb[0] = np.maximum(det[0]-32/2, 0)
        bb[1] = np.maximum(det[1]-32/2, 0)
        bb[2] = np.minimum(det[2]+32/2, img_size[1])
        bb[3] = np.minimum(det[3]+32/2, img_size[0])
        # 裁剪人脸框，再缩放
        cropped = dst[bb[1]:bb[3],bb[0]:bb[2],:]
        #cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        # 缩放到指定大小，并保存图片，以及边界框位置信息
        scaled = misc.imresize(cropped, (160,160), interp='bilinear')
        plt.figure(7)
        plt.imshow(scaled)
        plt.show()







# bounding_boxes, _ = align.detect_face.detect_face(dst, minsize, pnet, rnet, onet, threshold, factor)
# print(bounding_boxes.shape[0])
# print(bounding_boxes)