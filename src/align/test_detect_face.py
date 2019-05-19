from scipy import misc
import tensorflow as tf
import align.detect_face
import matplotlib.pyplot as plt
import numpy as np

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0

# function pick = nms(boxes,threshold,type)
# 非极大值抑制，去掉重复的检测框
def nms(boxes, threshold, method):
    if boxes.size==0:
        return np.empty((0,3))
    # 还原后的框的坐标
    print("进入nms非极大值抑制")
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    print(x1,y1,x2,y2)
    # 得分值，即是人脸的可信度
    s = boxes[:,4]
    print(s)
    area = (x2-x1+1) * (y2-y1+1)
    print(area)
    # 排序，从小到大，返回的是坐标
    I = np.argsort(s)
    #print(I)
    pick = np.zeros_like(s, dtype=np.int16)
    #print(pick)
    counter = 0
    s = 0
    while I.size>0:
        i = I[-1]
        s = s+1
        print("进入while%d"%s)
        print(i)
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        #print(idx)
        #print(type(idx))
        #x22= np.array([17.,18.,19.])
        #print(x22[idx])
        #print( x1[idx])
        #print( y1[idx])
        #print( x2[idx])
        #print( y2[idx])
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        #print(xx1)
        #print(yy1)
        #print(xx2)
        #print(yy2)
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        #print(inter)
        #print(area[idx])
        #print(area[i])
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        #print(o)
        #print(threshold)
        I = I[np.where(o<=threshold)]
        #print(I)
    pick = pick[0:counter]
    print(pick)
    print("_________________________")
    return pick

def generateBoundingBox(imap, reg, scale, t):
    """Use heatmap to generate bounding boxes"""
    stride = 2
    cellsize = 12
    # 获取x1,y1,x2,y2的坐标
    print("进入generate")
    #print(imap.shape)
    imap = np.transpose(imap)
    print(imap.shape)
    #print(type(imap))
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    print("进入reg")
    #print(reg[:, :, 0].shape)
    print(dx1)
    print(dy1)
    print(dx2)
    print(dy2)
    # 获取可信度大于阈值的人脸框的坐标
    print(imap)
    y, x = np.where(imap >= t)
    print(y)
    print(x)
    #print(type(y))
    #print(y.shape)
    #print(y.shape[0])
    # 只有一个符合的情况
    if y.shape[0] == 1:
        #print("进入if判断")
        dx1 = np.flipud(dx1)#翻转矩阵
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    # 筛选出符合条件的框
    print("_____________")
    # a= imap[(y,x)]
    # print(a)

    score = imap[(y, x)]
    print(score)
    print("_____________")
    #print(dx1[(y, x)].shape)
    print([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]])
    print((np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]])).shape)
    print("_____________")


    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    print(reg.shape)
    if reg.size == 0:
        #print("进入if")
        reg = np.empty((0, 3))
    # 还原尺度
    print("_____________")
    #print(np.vstack([y,x]))
    bb = np.transpose(np.vstack([y, x]))
    print(bb)
    print('进入计算部分')
    #print(stride * bb)
    print(scale)
    # #print((stride * bb + 1))
    #print((stride * bb + 1) / scale)
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    print(q1)
    print(q2)
    # shape(None, 9)
    #print(np.expand_dims(score, 0))
    #print(np.expand_dims(score, 1))
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    #print(boundingbox)
    return boundingbox, reg
    # boxes返回值中，前4个值是还原比例后的人脸框坐标，第5个值是该人脸框中是人脸的概率，后4个值的未还原的人脸框坐标
    # inter-scale nms
    # 非极大值抑制，去掉重复的检测框

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

image_path = 'C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\test\\test4.jpg'

img = misc.imread(image_path)
#print(img.shape)
bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
nrof_faces = bounding_boxes.shape[0]  # 人脸数目
#print('找到人脸数目为：{}'.format(nrof_faces))
print(_.shape)
print(bounding_boxes.shape)
#print(type(bounding_boxes))
print(bounding_boxes[:,:4])
det = bounding_boxes[:,0:4]
# 保存所有人脸框
det_arr = []
#print(type(det_arr))
# 原图片大小
img_size = np.asarray(img.shape)[0:2]
#print(img_size)

# for i in range(nrof_faces):
#     #print(det[i])
#     print(np.squeeze(det[i]))
#     det_arr.append(np.squeeze(det[i]))
# print(det_arr)

# 即使有多张人脸，也只要一张人脸就够了
# 获取人脸框的大小
bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
print(bounding_box_size)
# 原图片中心坐标
img_center = img_size / 2
#print(img_center)
# 求人脸框中心点相对于图片中心点的偏移，
# (det[:,0]+det[:,2])/2和(det[:,1]+det[:,3])/2组成的坐标其实就是人脸框中心点
offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
#print([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
#print(offsets)
# 求人脸框中心到图片中心偏移的平方和
# 假设offsets=[[   4.20016056  145.02849352 -134.53862838] [ -22.14250919  -26.74770141  -30.76835772]]
# 则offset_dist_squared=[  507.93206189 21748.70346425 19047.33436466]
offset_dist_squared = np.sum(np.power(offsets,2.0),0)
#print(offset_dist_squared)
# 用人脸框像素大小减去偏移平方和的两倍，得到的结果哪个大就选哪个人脸框
# 其实就是综合考虑了人脸框的位置和大小，优先选择框大，又靠近图片中心的人脸框
index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
#print(bounding_box_size-offset_dist_squared*2.0)
#print(index)
det_arr.append(det[index,:])
print("______________________________")
#print(det_arr)
#print(enumerate(det_arr))
for i, det in enumerate(det_arr):
    # [4,]  边界框扩大margin区域，并进行裁切
    det = np.squeeze(det)
    #print(i)
    #print(det)
    bb = np.zeros(4, dtype=np.int32)
    # 边界框周围的裁剪边缘，就是我们这里要裁剪的人脸框要比MTCNN获取的人脸框大一点，
    # 至于大多少，就由margin参数决定了
    # print(bb)
    bb[0] = np.maximum(det[0] - 32 / 2, 0)
    bb[1] = np.maximum(det[1] - 32 / 2, 0)
    bb[2] = np.minimum(det[2] + 32 / 2, img_size[1])
    bb[3] = np.minimum(det[3] + 32 / 2, img_size[0])
    # print(np.max(det[0] - 32 / 2, 0))
    # print(det[1] - 32 / 2)
    # print(det[2] + 32 / 2)
    # print(det[3] + 32 / 2)
    #print(bb)

    # 裁剪人脸框，再缩放
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    #print(cropped)
    # 缩放到指定大小，并保存图片，以及边界框位置信息
    scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
    #nrof_successfully_aligned += 1
    #filename_base, file_extension = os.path.splitext(output_filename)
    #if args.detect_multiple_faces:
    #    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
    #else:
    #    output_filename_n = "{}{}".format(filename_base, file_extension)
    # 保存图片
    #misc.imsave(output_filename_n, scaled)
    # 记录信息到bounding_boxes_XXXXX.txt文件里
    #text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))



###########################################################################################


factor_count=0
total_boxes=np.empty((0,9))
points=np.empty(0)
#print(type(total_boxes))
print(total_boxes)
print("显示total_boxes")
#print(points)
#print(type(points))

# 获取输入的图片的宽高
h=img.shape[0]
w=img.shape[1]
print(h)
print(w)
# 宽/高，谁小取谁 250*250
minl=np.amin([h, w])
#print(minl)
m=12.0/minsize#P Net 12*12 12/20=0.6
minl=minl*m#250*0.6=150
#print(minl)
# create scale pyramid
# 创建比例金字塔
scales=[]
while minl>=12:
    scales += [m*np.power(factor, factor_count)]
    minl = minl*factor
    #print(minl)
    factor_count += 1
    #print(factor_count)
print(scales)
# 将图片显示出来

plt.figure()
scale_img = img.copy()

# 第一步，首先将图像缩放到不同尺寸形成“图像金字塔”
# 然后，经过P-Net网络
# first stage
i=0
for scale in scales:
    # 宽高要取整
    hs = int(np.ceil(h * scale))
    ws = int(np.ceil(w * scale))
    print(hs)
    print(ws)
    # 使用opencv的方法对图片进行缩放
    im_data = align.detect_face.imresample(img, (hs, ws))
    print(im_data.shape)
    print("im_data设置完毕")
    #plt.imshow(scale_img)
    #plt.show()
    # 可视化的显示“图像金字塔”的效果
    # --韦访添加
    #plt.imshow(img)
    #plt.show()
    #plt.imshow(im_data)
    #plt.show()
    #scale_img[0:im_data.shape[0], 0:im_data.shape[1]] = 0
    #scale_img[0:im_data.shape[0], 0:im_data.shape[1]] = im_data[0:im_data.shape[0], 0:im_data.shape[1]]
    # plt.imshow(scale_img)
    # plt.show()
    # print('im_data.shape[0]', im_data.shape[0])
    # print('im_data.shape[1]', im_data.shape[1])
#     # 对图片数据进行归一化处理 [-1,1]
#     #print(im_data.shape)
    im_data = (im_data - 127.5) * 0.0078125
    print("---------------------")
    #print(im_data.shape)
    # 增加一个维度，即batch size，因为我们这里每次只处理一张图片，其实batch size就是1
    img_x = np.expand_dims(im_data, 0)
    #print(img_x.shape)
    img_y = np.transpose(img_x, (0, 2, 1, 3))
    #print(img_y.shape)
    # 送进P-Net网络
    # 假设img_y.shape=(1, 150, 150, 3)
    # 因为P-Net网络要经过3层核为3*3步长为1*1的卷积层，一层步长为2*2池化层
    # 所以conv4-2层输出形状为(1, 70, 70, 4)
    # 70是这么来的，(150-3+1)/1=148，经过池化层后为148/2=74，
    # 再经过一个卷积层(74-3+1)/1=72，再经过一个卷积层(72-3+1)/1=70
    # 计算方法参考博客：https://blog.csdn.net/rookie_wei/article/details/80146620
    # prob1层的输出形状为(1, 70, 70, 2)
    out = pnet(img_y)
    #print(type(out))
    #print(out[0].shape)
    #print(out[1].shape)
    # 又变回来
    # out0的形状是(1, 70, 70, 4)
    # 返回的是可能是人脸的框的坐标
    out0 = np.transpose(out[0], (0, 2, 1, 3))
    # out1的形状是(1, 70, 70, 2)
    # 返回的是对应与out0框中是人脸的可信度，第2个值为是人脸的概率
    out1 = np.transpose(out[1], (0, 2, 1, 3))
    print("out的shape")
    print(out0.shape)
    print(out1.shape)
    print("-----------------")
    #print(out0[:,:,:,:].shape)
    print(out0[0,:,:,:].shape)
    print("-----------------")
    #print(out1[:,:,:,1].shape)
    print(out1[0,:,:,1].shape)
    # out1[0,:,:,1]：表示框的可信度，只要一个值即可，因为这两个值相加严格等于1，这里只要获取“是”人脸框的概率
    # out0[0,:,:,:]：人脸框
    # scales:图片缩减比例
    # threshold:阈值，这里取0.6
    # boxes返回值中，前4个值是还原比例后的人脸框坐标，第5个值是该人脸框中是人脸的概率，后4个值的未还原的人脸框坐标
    boxes, _ = generateBoundingBox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, threshold[0])
#     # 人脸框坐标对应的可信度
#     print('处理之前：', out1[0, :, :, 1])
#     print('------------------')
#     s = boxes[:, 4]
#     print('处理之后：', s)
#
#     # # 显示人脸框
#     print('------------------')
#     x1 = boxes[:, 0]
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]
#     print(len(boxes))
#     print('------------------')
#     for i in range(len(boxes)):
#         print(x1[i])
#         print(y1[i])
#         print(x2[i])
#         print(y2[i])
#         print('------------------')
#         print(i)
#         plt.gca().add_patch(plt.Rectangle((x1[i], y1[i]), x2[i] - x1[i], y2[i] - y1[i], edgecolor='w',facecolor='none'))

    # --韦访添加
# plt.imshow(scale_img)
# plt.show()
# exit()

# inter-scale nms
# 非极大值抑制，去掉重复的检测框
    pick = nms(boxes.copy(), 0.5, 'Union')

    if boxes.size > 0 and pick.size > 0:
        boxes = boxes[pick, :]
        total_boxes = np.append(total_boxes, boxes, axis=0)

        # x1 = boxes[:, 0]
        # y1 = boxes[:, 1]
        # x2 = boxes[:, 2]
        # y2 = boxes[:, 3]
        # for i in range(len(boxes)):
        #     print(x1[i], y1[i], x2[i], y2[i])
        #     plt.gca().add_patch(
        #         plt.Rectangle((x1[i], y1[i]), x2[i] - x1[i], y2[i] - y1[i], edgecolor='w', facecolor='none'))

# --韦访添加
#plt.imshow(scale_img)
#plt.show()
#exit()
    # 图片按照所有scale走完一遍，会得到在原图上基于不同scale的所有的bb，然后对这些bb再进行一次NMS
    # 并且这次NMS的threshold要提高
numbox = total_boxes.shape[0]
if numbox > 0:
    # 再经过nms筛选掉一些可靠度更低的人脸框
    pick = nms(total_boxes.copy(), 0.7, 'Union')
    total_boxes = total_boxes[pick, :]
    # 获取每个人脸框的宽高
    regw = total_boxes[:, 2] - total_boxes[:, 0]
    regh = total_boxes[:, 3] - total_boxes[:, 1]
    # x1 = total_boxes[:, 0]
    # y1 = total_boxes[:, 1]
    # x2 = total_boxes[:, 2]
    # y2 = total_boxes[:, 3]
    # for i in range(len(total_boxes)):
    #     print(x1[i], y1[i], x2[i], y2[i])
    #     plt.gca().add_patch(
    #         plt.Rectangle((x1[i], y1[i]), x2[i] - x1[i], y2[i] - y1[i], edgecolor='w', facecolor='none'))

    # 对人脸框坐标做一些处理，使得人脸框更紧凑
    qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
    qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
    qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
    qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
    # x1 = qq1
    # y1 = qq2
    # x2 = qq3
    # y2 = qq4
    # for i in range(len(total_boxes)):
    #     print('lll', x1[i], y1[i], x2[i], y2[i])
    #     plt.gca().add_patch(
    #         plt.Rectangle((x1[i], y1[i]), x2[i] - x1[i], y2[i] - y1[i], edgecolor='r', facecolor='none'))
        # --韦访添加
# plt.imshow(scale_img)
# plt.show()
# exit()
    total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
    total_boxes = align.detect_face.rerec(total_boxes.copy())
    total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = align.detect_face.pad(total_boxes.copy(), w, h)

# R-Net
numbox = total_boxes.shape[0]
if numbox > 0:
    # second stage R-Net 对于P-Net输出的bb，缩放到24x24大小
    tempimg = np.zeros((24, 24, 3, numbox))
    for k in range(0, numbox):
        tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
        tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
        if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
            # R-Net输入大小为24*24，所以要进行缩放
            tempimg[:, :, :, k] = align.detect_face.imresample(tmp, (24, 24))
        #else:
        #    return np.empty()
    # 标准化[-1,1]
    tempimg = (tempimg - 127.5) * 0.0078125
    # 转置[n,24,24,3]
    tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
    # 经过R-Net网络
    out = rnet(tempimg1)
    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    score = out1[1, :]
    ipass = np.where(score > threshold[1])
    total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
    mv = out0[:, ipass[0]]
    if total_boxes.shape[0] > 0:
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        total_boxes = align.detect_face.bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
        total_boxes = align.detect_face.rerec(total_boxes.copy())

# 第三步，经过O-Net网络
numbox = total_boxes.shape[0]
if numbox > 0:
    # third stage
    total_boxes = np.fix(total_boxes).astype(np.int32)
    dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = align.detect_face.pad(total_boxes.copy(), w, h)
    tempimg = np.zeros((48, 48, 3, numbox))
    for k in range(0, numbox):
        tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
        tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
        if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
            # O-Net输入大小为48*48，所以要进行缩放
            tempimg[:, :, :, k] = align.detect_face.imresample(tmp, (48, 48))
        #else:
        #    return np.empty()
    tempimg = (tempimg - 127.5) * 0.0078125
    tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
    # 经过O-Net网络
    out = onet(tempimg1)
    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    out2 = np.transpose(out[2])
    score = out2[1, :]
    points = out1
    ipass = np.where(score > threshold[2])
    points = points[:, ipass[0]]
    total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
    mv = out0[:, ipass[0]]

    w = total_boxes[:, 2] - total_boxes[:, 0] + 1
    h = total_boxes[:, 3] - total_boxes[:, 1] + 1
    points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
    points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
    if total_boxes.shape[0] > 0:
        total_boxes = align.detect_face.bbreg(total_boxes.copy(), np.transpose(mv))
        pick = nms(total_boxes.copy(), 0.7, 'Min')
        total_boxes = total_boxes[pick, :]
        points = points[:, pick]

# 显示人脸框和关键点
for i in range(len(total_boxes)):
    x1 = total_boxes[:, 0]
    y1 = total_boxes[:, 1]
    x2 = total_boxes[:, 2]
    y2 = total_boxes[:, 3]
    print('lll', x1[i], y1[i], x2[i], y2[i])
    plt.gca().add_patch(
        plt.Rectangle((x1[i], y1[i]), x2[i] - x1[i], y2[i] - y1[i], edgecolor='r', facecolor='none'))

plt.scatter(points[0], points[5], c='red')
plt.scatter(points[1], points[6], c='red')
plt.scatter(points[2], points[7], c='red')
plt.scatter(points[3], points[8], c='red')
plt.scatter(points[4], points[9], c='red')

plt.imshow(scale_img)
plt.show()
exit()




