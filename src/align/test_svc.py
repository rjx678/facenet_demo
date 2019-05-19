import tensorflow as tf
import numpy as np
import facenet
import math
import pickle
from scipy import  misc

import sklearn.metrics as ms
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt  # 可视化绘图
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

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
    #print(nrof_samples)# 90
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    #print(images.shape) #90 160 160 3
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        #print(img.ndim)3
        #print(img.shape) 160 160 3
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
        #print(images.shape)# 90 160 160 3
        #print(images[i,:,:,:].shape) 160 160 3
    return images

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        #print("进入函数")
        #print(len(paths))
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            #print(cls.name)
            #print(nrof_train_images_per_class)
            nrof_train_images_per_class = int(0.8*len(paths))
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    #print(dataset)
    #print(len(dataset))
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        #print(dataset[i].image_paths)
        #print(len(dataset[i].image_paths))
        #print(image_paths_flat)
        #print([i])
        labels_flat += [i] * len(dataset[i].image_paths)#复制10个[0],[1],[2],[3]
        #print(labels_flat)
    #print(len(image_paths_flat))
    #print(len(labels_flat))
    return image_paths_flat, labels_flat

# dataset_tmp = facenet.get_dataset("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_1601")
# #print(dataset_tmp)
# train_set, test_set = split_dataset(dataset_tmp, 20,10)
# dataset = train_set
# paths, labels = get_image_paths_and_labels(dataset)
# dataset = test_set
# paths, labels = get_image_paths_and_labels(dataset)
def main1(args):
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            np.random.seed(seed=666)
            if True:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, 20,30)
                # if (args.mode == 'TRAIN'):
                #     dataset = train_set
                # elif (args.mode == 'CLASSIFY'):
                #     dataset = test_set
            else:
                dataset = facenet.get_dataset("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_160")
            dataset = train_set
            paths, labels = get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\20180402-114759")

            # Get input and output tensors
            # 输入的图片
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # 卷积网络最后输出的特征
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # 现在是不是训练阶段
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)  # 图片总数目
            # print(nrof_images)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 90))
            # print(nrof_batches_per_epoch)
            emb_array = np.zeros((nrof_images, embedding_size))
            #print(emb_array.shape)

            for i in range(nrof_batches_per_epoch):
                start_index = i * 90
                end_index = min((i + 1) * 90, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, 160)
                # 计算特征
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)


            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            #model = SVC(kernel='poly',C=1,gamma=1,degree=3,probability=True)
            #model = SVC(C=100,kernel='linear',probability=True)
            model =SVC(kernel='rbf',C=10,gamma=1,probability=True)
            #model = SVC(kernel='poly', C=10, gamma=1, degree=10, probability=True)
            # model = SVC(kernel='poly',C=1,gamma=1,degree=5,probability=True)
            # model = SVC(kernel='poly',C=1,gamma=1,degree=6,probability=True)
            # model = SVC(kernel='poly',C=1,gamma=1,degree=7,probability=True)



            model.fit(emb_array,labels)
            # param_grid = [#{'kernel': ['linear'], 'C': [0.001,0.01,0.1,1]},
            #               #{'kernel': ['rbf'], 'C': [0.001,0.01,0.1,1], 'gamma': [1, 0.1, 0.01, 0.001]},
            #               {
            #               'degree':[2,3,4,5,6,7,8,9,10]}
            #               ]
            # grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10,
            #                     scoring='accuracy',n_jobs=4)  # 针对每个参数对进行了10次交叉验证。scoring='accuracy'使用准确率为结果的度量指标。可以添加多个度量指标
            # grid.fit(emb_array, labels)
            print(emb_array.shape)
            #print(type(emb_array))
            #print(emb_array.shape[0])
            # print(emb_array[:,0].shape)
            #print(len(labels))
            #print(type(labels))
            print("_________________")
            # #print('网格搜索-度量记录：', grid.cv_results_)  # 包含每次训练的相关信息
            # print('网格搜索-最佳度量值:', grid.best_score_)  # 获取最佳度量值
            # print('网格搜索-最佳参数：', grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
            # print('网格搜索-最佳模型：', grid.best_estimator_)  # 获取最佳度量时的分类器模型


            dataset= test_set
            paths1, labels1 = get_image_paths_and_labels(dataset)
            #
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths1))
            # print(paths)
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\20180402-114759")

            # Get input and output tensors
            # 输入的图片
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # 卷积网络最后输出的特征
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # 现在是不是训练阶段
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths1)  # 图片总数目
            # print(nrof_images)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 1))
            # print(nrof_batches_per_epoch)
            emb_array1 = np.zeros((nrof_images, embedding_size))
            print(emb_array1.shape)

            for i in range(nrof_batches_per_epoch):
                start_index = i * 1
                end_index = min((i + 1) * 1, nrof_images)
                paths_batch = paths1[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, 160)
                # 计算特征
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array1[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            # 使用获取的最佳参数生成模型，预测数据
            #knn = SVC(n_neighbors=grid.best_params_['n_neighbors'],
            #                           weights=grid.best_params_['weights'])  # 取出最佳参数进行建模
            # model = SVC(kernel='poly',C=1,gamma=1,degree=grid.best_params_['degree'],
            #             probability=True)
            #,gamma=grid.best_params_['gamma'],
            # model = SVC(kernel='linear',probability=True)
            # model.fit(emb_array,labels)
            #print(model.predict(emb_array1))
            predictions = model.predict_proba(emb_array1)
            # #print(predictions)
            # #print(predictions.shape)
            best_class_indices = np.argmax(predictions, axis=1)
            #best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            # #print(best_class_indices)
            # #print(len(best_class_indices))
            #
            # #print(best_class_probabilities)
            # #print(len(best_class_probabilities))
            accuracy = np.mean(np.equal(best_class_indices, labels1))
            print('Accuracy: %.3f' % accuracy)



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv,
        n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='accuracy', verbose=0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training accuracy")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="validation accuracy")

    plt.legend(loc="best")
    return plt

# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
#                         n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
#     plt.figure()
#     plt.title(title)
#     if ylim is not None:
#         plt.ylim(*ylim)
#     plt.xlabel("Training examples")
#     plt.ylabel("Score")
#     train_sizes, train_scores, test_scores = validation_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std = np.std(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)
#     test_scores_std = np.std(test_scores, axis=1)
#     plt.grid()
#
#     plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, alpha=0.1,
#                      color="r")
#     plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
#              label="Cross-validation score")
#
#     plt.legend(loc="best")
#     return plt





def main():
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            np.random.seed(seed=666)

            dataset = facenet.get_dataset("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_160_not_aligned")

            paths, labels = get_image_paths_and_labels(dataset)
            #X_train, X_test, y_train, y_test = train_test_split(paths, labels, test_size=0.4, random_state=0)
            # print(X_train)
            # print(X_test)
            # print(y_train)
            # print(y_test)
            #
            # print(len(X_train))
            # print(len(X_test))
            # print(len(y_train))
            # print(len(y_test))


            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model("C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\20180402-114759")

            # Get input and output tensors
            # 输入的图片
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            # 卷积网络最后输出的特征
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            # 现在是不是训练阶段
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            #print(embeddings) #Tensor("embeddings:0", shape=(?, 512), dtype=float32)
            #print(embedding_size)#512
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)  # 图片总数目
            #print(nrof_images)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / 90))
            #print(nrof_batches_per_epoch)
            emb_array = np.zeros((nrof_images, embedding_size))
            #print(emb_array.shape)
            import os
            for i in range(nrof_batches_per_epoch):
                start_index = i * 90
                end_index = min((i + 1) * 90, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = load_data(paths_batch, False, False, 160)
                # 计算特征
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
            classifier_filename_exp = os.path.expanduser("C:/Users/rjx/PycharmProjects/untitled1/facenet-master/models/sd_82a.pkl")
            from sklearn.model_selection import learning_curve
            from sklearn.model_selection import ShuffleSplit
            from sklearn.model_selection import validation_curve

    ################1#############
            model = SVC(C=1,kernel='linear', probability=True)
            model.fit(emb_array, labels)
            ylim = (0.75, 1.05)
            title = r"Learning Curves(SVM) for self data(not aligned)"
            #title = r"Learning Curves(SVM) for not aligned data"
            cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
            plt.figure()
            plt.title(title)
            plt.ylim(*ylim)
            plt.xlabel("Number of training samples")
            plt.ylabel("Accuracy")
            train_sizes, train_scores, test_scores = learning_curve(
                model, emb_array, labels, cv=cv,
                n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10),
                scoring='accuracy', verbose=0)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.grid()

            plt.plot(train_sizes, test_scores_mean, 'o-', color="y",
                     label="SVM Linear(C=1) validation accuracy")

            model = SVC(C=10, kernel='linear', probability=True)
            model.fit(emb_array, labels)

            train_sizes, train_scores, test_scores = learning_curve(
                model, emb_array, labels, cv=cv,
                n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10),
                scoring='accuracy', verbose=0)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.plot(train_sizes, test_scores_mean, 'o-', color="r",
                     label="SVM Linear(C=10) validation accuracy")


            model = SVC(C=10,degree=5, gamma=1,kernel='poly', probability=True)
            model.fit(emb_array, labels)

            train_sizes, train_scores, test_scores = learning_curve(
                model, emb_array, labels, cv=cv,
                n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10),
                scoring='accuracy', verbose=0)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
                     label="SVM Poly(C=10,gamma=1,degree=5) validation accuracy")


            model = SVC(C=10,gamma=2.4, kernel='rbf', probability=True)
            model.fit(emb_array, labels)

            train_sizes, train_scores, test_scores = learning_curve(
                model, emb_array, labels, cv=cv,
                n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10),
                scoring='accuracy', verbose=0)
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            # title = r"Learning Curves(SVM) for not aligned data"
            plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="SVM RBF(C=10,gamma=2.4) validation accuracy")

            # model = SVC(C=10,degree=6, gamma=1, kernel='poly', probability=True)
            # model.fit(emb_array, labels)
            #
            # train_sizes, train_scores, test_scores = learning_curve(
            #     model, emb_array, labels, cv=cv,
            #     n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10),
            #     scoring='accuracy', verbose=0)
            # train_scores_mean = np.mean(train_scores, axis=1)
            # train_scores_std = np.std(train_scores, axis=1)
            # test_scores_mean = np.mean(test_scores, axis=1)
            # test_scores_std = np.std(test_scores, axis=1)
            # plt.plot(train_sizes, test_scores_mean, 'o-', color="y",
            #          label="SVM Poly(C=10,degree=6,gamma=1) validation accuracy")

            # model = SVC(C=10, gamma=1,degree=10, kernel='poly', probability=True)
            # model.fit(emb_array, labels)
            #
            # train_sizes, train_scores, test_scores = learning_curve(
            #     model, emb_array, labels, cv=cv,
            #     n_jobs=4, train_sizes=np.linspace(.1, 1.0, 10),
            #     scoring='accuracy', verbose=0)
            # train_scores_mean = np.mean(train_scores, axis=1)
            # train_scores_std = np.std(train_scores, axis=1)
            # test_scores_mean = np.mean(test_scores, axis=1)
            # test_scores_std = np.std(test_scores, axis=1)
            # # title = r"Learning Curves(SVM) for not aligned data"
            # plt.plot(train_sizes, test_scores_mean, 'o-', color="y",
            #          label="SVM Poly(C=10,degree=10,gamma=1) validation accuracy")


            plt.legend(loc="best")
            plt.show()


import argparse
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default="TRAIN")
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.', default=
                        "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_160_aligned")
    parser.add_argument('--model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\20180402-114759")
    parser.add_argument('--classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.',
                        default="C:/Users/rjx/PycharmProjects/untitled1/facenet-master/models/sdd_p_82na.pkl")
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.',
                        action='store_true',
                        default=True)
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    # parser.add_argument('--nrof_train_images_per_class', type=int,
    #     help='Use this number of images from each class for training and the rest for testing', default=50)

    return parser.parse_args(argv)



if __name__ == '__main__':
    main()