import os
#获取文件夹中的所有类别的图片路径
def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = [os.path.join(facedir,images) for images in os.listdir(facedir)]
        img_num = len(images)
        for i in range(img_num):
            image_paths += [os.path.join(images[i],img) for img in os.listdir(images[i])]
    return image_paths
img = get_image_paths(r'C:\Users\rjx\PycharmProjects\untitled1\facenet-master\data\self_data_1601')

from scipy import misc
from PIL import Image
#传入图片路径，将所有图片旋转一定角度后保存
def rotate(img,degree):
    for i in range(len(img)):
        image = Image.open(img[i])
        path,filename = os.path.split(img[i])
        l,r=filename.split('_',1)
        path11 = "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_160_align\\" + l
        l= l+"_"+str(degree)+"_"#旋转角度
        path1 =path11 + "\\" + l + r
        img_rote = image.rotate(degree)
        misc.imsave(path1,img_rote)


rotate(img,degree=90)
rotate(img,degree=-90)
rotate(img,degree=45)
rotate(img,degree=-45)
rotate(img,degree=30)
rotate(img,degree=-30)
rotate(img,degree=60)
rotate(img,degree=-60)
rotate(img,degree=75)
rotate(img,degree=-75)
rotate(img,degree=15)
rotate(img,degree=-15)




# 更名小脚本
# import cv2
# for images in os.listdir(r'C:\Users\rjx\PycharmProjects\untitled1\facenet-master\data\self_data\ty'):
#     img_path = "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_160_no\\ty\\"+images
#     #print(img_path)
#     l,r=images.split('_',1)
#     l ='ty'
#     r = '_'+r
#     images = "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data\\ty\\"+l+r
#     cv2.imwrite(images,cv2.imread(img_path))
#     #print(images)

