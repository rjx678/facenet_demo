from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

os.environ["CUDA_VISIBLE_DEVICES"] ="0"
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import math
import pickle
from sklearn.svm import SVC
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)

def main(args):
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            np.random.seed(seed=args.seed)
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class)
                if (args.mode=='TRAIN'):
                    print("进入train")
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # # Check that there are at least one training image per class
            # for cls in dataset:
            #     assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')

                 
            paths, labels = get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            #print(paths)
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            #输入的图片
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            #卷积网络最后输出的特征
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            #现在是不是训练阶段
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            #print(embeddings)
            #print("embedding size is ")
            #print(embedding_size)
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)#图片总数目
            #print(nrof_images)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            #print(nrof_batches_per_epoch)
            emb_array = np.zeros((nrof_images, embedding_size))
            print(emb_array.shape)

            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                #计算特征
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                # print(emb_array.shape)
                # print(sess.run(embeddings, feed_dict=feed_dict))
                # print(sess.run(embeddings, feed_dict=feed_dict).shape)
                # print(sess.run(embeddings, feed_dict=feed_dict)[0])
                # print(sess.run(embeddings, feed_dict=feed_dict)[0].shape)
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(C=1, kernel='linear', probability=True)
                #model = SVC(kernel="rbf",C=10,gamma=2.4,probability=True)
                #model = SVC(kernel='poly',degree=5,gamma=1,coef0=0,probability=True)
                #model = SVC(kernel='poly',gamma=2.4,C=10,degree=6,probability=True)
                #model.fit(emb_array, labels)
                # param_grid = {'C': [1e-4,1e-3, 1e-2, 1e-1, 1, 10, 100, 1000,10000],
                #               'gamma': [1,0.1,0.01,0.001],
                #               'kernel': ('linear', 'poly'),
                #               }
                # # model = SVC(kernel='linear', probability=True)
                # from sklearn.decomposition import PCA
                # pca = PCA(n_components=128)
                # new_array = pca.fit_transform(emb_array)
                # print(new_array)
                # print(new_array.shape)
                # print("______________________")


                model.fit(emb_array, labels)
                # model = SVC(kernel='poly',degree=2,gamma=1,coef0=0,probability=True)
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):#376
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)

def split_dataset(dataset, min_nrof_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            #print(cls.name)
            nrof_train_images_per_class=int(0.8*len(paths))
            #print(nrof_train_images_per_class)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set
# dataset1= "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_160"
# dataset_tmp = facenet.get_dataset(dataset1)
#
# train,test=split_dataset(dataset_tmp, 20)
# print(len(train))
# print(len(test))


#print("___________")
def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    #print(dataset)
    #print(len(dataset))
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    #print(len(image_paths_flat))
    #print(len(labels_flat))
    return image_paths_flat, labels_flat
#paths, labels = get_image_paths_and_labels(train)
#images = facenet.load_data(paths, False, False, 160)
#print(images.shape)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification',default="CLASSIFY")
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.',default=
                        "C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\data\\self_data_160_align")
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',
                        default="C:\\Users\\rjx\\PycharmProjects\\untitled1\\facenet-master\\models\\20180402-114759")
    parser.add_argument('--classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.',
                        default="C:/Users/rjx/PycharmProjects/untitled1/facenet-master/models/sd160_l_a.pkl")
    parser.add_argument('--use_split_dataset',
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true',
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
import sys
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
