import numpy as np
import tensorflow as tf
import os
import argparse
import resnet_model
import tensorlayer as tl
from random import shuffle
from scipy import misc
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Define parameters.')

parser.add_argument('--n_epoch', type=int, default=15)
parser.add_argument('--n_batch', type=int, default=64)
parser.add_argument('--n_img_row', type=int, default=100)
parser.add_argument('--n_img_col', type=int, default=100)
parser.add_argument('--n_img_channels', type=int, default=1)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.15)
parser.add_argument('--n_resid_units', type=int, default=5)
parser.add_argument('--lr_schedule', type=int, default=8)
parser.add_argument('--lr_factor', type=float, default=0.5)

args = parser.parse_args()

TRAIN_DIR = 'C:\\train\\train'
TEST_DIR  = 'C:\\test\\test'
IMG_SIZE = 100
LR = 1e-3

def label_img(img):
    word_label = img.split('.')[-3]

    if word_label == 'cat' : return [1, 0]

    elif word_label == 'dog' : return [0, 1]


def create_train_data():
    temp_data, training_data = [], []
    training_label = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        image = misc.imread(path, 'L')
        # image = image.resize((IMG_SIZE, IMG_SIZE), refcheck=False)
        temp_data.append([np.array(image), np.array(label)])

    shuffle(temp_data)
    for i in range(len(temp_data)):
        training_data.append(temp_data[i][0])
        training_label.append(temp_data[i][1])
    np.save('train_data.npy', training_data)
    np.save('train_label.npy', training_label)

    return training_data, training_label

def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        img_num = img.split('.')[0]
        path = os.path.join(TEST_DIR, img)
        image = misc.imread(path, 'L')
        image = misc.imresize((IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(image), img_num])
    shuffle(testing_data)

    return testing_data


train_data, train_label = create_train_data()
train_x = train_data[:-500]
train_y = train_label[:-500]
test_x = train_data[-500:]
test_y = train_label[-500:]
print(len(test_x))
print(test_x[0])
print(test_y[0])


# class CNNEnv:
#     def __init__(self):
#         self.x_train = train_x
#         self.y_train = train_y
#         self.x_test = test_x
#         self.y_test = test_y
#
#         self.mean = np.mean(self.x_train, axis= 0, keepdims= True)
#         self.std = np.std(self.x_train)
#         self.x_train = (self.x_train - self.mean) / self.std
#         self.x_test = (self.x_test - self.mean) / self.std
#
#         self.num_examples = self.x_train.shape[0]
#         self.index_in_epoch = 0
#         self.epochs_completed = 0
#
#         self.batch_num = args.n_batch
#         self.num_epoch = args.n_epoch
#         self.img_row = args.n_img_row
#         self.img_col = args.n_img_col
#         self.img_channels = args.n_img_channels
#         self.nb_classes = args.n_classes
#         self.num_iter = self.x_train.shape[0] // self.batch_num
#
#     def next_batch(self, test_or_train):
#         start = self.index_in_epoch
#         self.index_in_epoch += self.batch_num
#
#         if self.index_in_epoch > self.num_examples:
#             self.epochs_completed += 1
#             perm = np.arange(self.num_examples)
#             np.random.shuffle(perm)
#             if test_or_train == 'train':
#                 self.x_train = self.x_train[perm]
#                 self.y_train = self.y_train[perm]
#             else:
#                 self.x_test = self.x_test[perm]
#                 self.y_test = self.y_test[perm]
#             start = 0
#             self.index_in_epoch = self.batch_num
#             assert self.batch_num <= self.num_examples
#         end = self.index_in_epoch
#         if test_or_train == 'train':
#             return self.x_train[start:end], self.y_train[start:end]
#         else:
#             return self.x_test[start:end], self.y_test[start:end]
#
#
#
#     def train(self, hps):
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         sess= tf.InteractiveSession(config= config)
#
#         img = tf.placeholder(tf.float32, shape= [self.batch_num, IMG_SIZE, IMG_SIZE, 1])
#         labels = tf.placeholder(tf.int32, shape= [self.batch_num, ])
#
#         model = resnet_model.ResNet(hps, img, labels, 'train')
#         model.build_graph()
#
#         merged = model.summaries
#         train_writer = tf.summary.FileWriter("/tmp/train_log", sess.graph)
#
#         sess.run(tf.global_variables_initializer())
#         print('Done initializing variables')
#         print('Running model...')
#
#         lr = args.lr
#
#         for j in range(self.num_epoch):
#             print('Epoch {}'.format(j + 1))
#
#             if (j + 1) % args.lr_schedule == 0:
#                 print('reducing l_rate by factor', args.lr_factor)
#                 lr *= args.lr_factor
#             print('number of iteration', self.num_iter)
#
#             for i in range(self.num_iter):
#                 batch = self.next_batch('train')
#                 feed_dict = {img: batch[0], labels: batch[1], model.lrn_rate: lr}
#                 _, loss, accu, summary, lr = sess.run([model.train_op, model.cost,
#                                                        model.acc, merged, model.lrn_rate],
#                                                       feed_dict= feed_dict)
#
#                 # print('This is the predictions', y_pred)
#                 if i % 200 == 0:
#                     print('step', i)
#                     print('Training loss', loss)
#                     print('Training accuracy', accu)
#                     print('Learning rate', lr)
#
#             print('Running Evaluation...')
#             test_loss, test_acc, n_batch = 0, 0, 0
#             prediction = list()
#             for batch in tl.iterate.minibatches(inputs=self.x_test,
#                                                 targets=self.y_test,
#                                                 batch_size=self.batch_num,
#                                                 shuffle=False):
#                 feed_dict_eval = {img: batch[0], labels: batch[1]}
#
#                 loss, accu, summary, y_pred = sess.run([model.cost, model.acc, merged, model.predictions], feed_dict=feed_dict_eval)
#                 prediction.extend(y_pred)
#
#                 train_writer.add_summary(summary, n_batch)
#                 test_loss += loss
#                 test_acc += accu
#                 n_batch += 1
#
#             # print('This is the prediction', prediction)
#             tot_test_loss = test_loss / n_batch
#             tot_test_acc = test_acc / n_batch
#
#             print('   Test loss: {}'.format(tot_test_loss))
#             print('   Test accuracy: {}'.format(tot_test_acc))
#
# run = CNNEnv()
#
# hps = resnet_model.HParams(batch_size=run.batch_num,
#                            num_classes=run.nb_classes,
#                            min_lrn_rate=0.0001,
#                            lrn_rate=args.lr,
#                            num_residual_units=args.n_resid_units,
#                            use_bottleneck=False,
#                            weight_decay_rate=0.0002,
#                            relu_leakiness=0.1,
#                            optimizer='mom')
#
# run.train(hps)
