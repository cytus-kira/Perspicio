import glob
import pickle
from shutil import copyfile
from models.Head_pre.img2pose import img2poseModel
from models.Head_pre.model_loader import load_model
from tools.transform import gaze_transform_deal, get_test_transform
from PIL import Image
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import os
import numpy as np
from classification import Trainer
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from time import *
import pandas as pd
import matplotlib.pylab as plt

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
#
model_name = 'Head_posture'  # c Gaze360/resnet*/Head_posture/Face_cls
note = ''  # last/preimg
ImageNet_pretrained = True
# 机器学习分类参数
if model_name == "Gaze360":
    NB_features = 512  # CNN特征维度,根据模型调整
    dataset_name = "dataset/body"
if model_name == "Head_posture":
    NB_features = 1024  # CNN特征维度,根据模型调整
    dataset_name = "dataset/body"
# dataset_name = "dataset/head"  # c gaze:head other:body
classes_num = 2
input_img_size = (224, 224)  # [Height, Weight]
channel_num = 3  # 1 for grayscale and 3 for colored input image

train_path = dataset_name + '/train'
val_path = dataset_name + '/val'  # validation (best model is saved based on the validation data)
result_folder = model_name + note + "_trial0"

# head Parameters'
# -----------------------------------------------------------------------------
np.set_printoptions(suppress=True)
# -----------------------------------------------------------------------------

batch_size = 128
num_epochs = 20
learning_rate = 0.001
early_stopping_patience = 5  # parameter to contro early stopping
use_class_weight = False  # set True to use class weight (useful in case of imbalanced dataset)
mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])  # for ImageNet dataset

trainer = Trainer(model_name, dataset_name, classes_num, input_img_size, channel_num,
                  train_path, val_path, result_folder, batch_size, num_epochs,
                  None, None, mean, std, None, None)


printwrong = False  # 是否打印错误图像
TRAIN_LABEL_DIR = dataset_name + '/train.txt'
VAL_LABEL_DIR = dataset_name + '/val.txt'
result_folder_ = './trained_models/' + result_folder

cnn_model_path = os.path.join(result_folder_, model_name + '-bestmodel.pth')
if model_name == 'Gaze360':
    cnn_model_path = os.path.join(result_folder_, model_name + '_model.pth.tar')  # c Gaze360
# #构建保存特征的文件夹
feature_path = os.path.join(result_folder_, 'ml_features').replace('\\', '/')
WRONG_TEST_DIR = result_folder_ + '/wrong test_image/'
os.makedirs(feature_path, exist_ok=True)


def preprocess():
    train_labels = os.listdir(train_path)
    ##写train.txt文件
    for index, label in enumerate(train_labels):
        timglist = glob.glob(os.path.join(train_path, label, '*.jpg').replace('\\', '/'))
        print(len(timglist))


        with open(dataset_name + '/train.txt', 'a') as f:
            for img in timglist:
                img = img.replace('\\', '/')
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')

    test_labels = os.listdir(val_path)
    ##写test.txt文件
    for index, label in enumerate(test_labels):
        timglist = glob.glob(os.path.join(val_path, label, '*.jpg'))
        print(len(timglist))
        # print(timglist)
        with open(dataset_name + '/val.txt', 'a') as f:
            for img in timglist:
                img = img.replace('\\', '/')
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')
    f.close()
    print('finish txt!')


def save_feature(feature_path, label_path):
    '''
    提取特征，保存为pkl文件
    '''
    # 预训练imagenet模型
    if note == 'preimg':
        model = trainer.build_model(ImageNet_pretrained=True)
        model = model.cuda()
        model.eval()
    else:
        # 正常加载模型
        model = trainer.load_model(cnn_model_path)

    print('..... Finished loading model! ......')
    nb_features = NB_features
    features = np.empty((len(imgs), nb_features))
    labels = []
    start_predict_time = time()
    for i in tqdm(range(len(imgs))):
        img_path = imgs[i].strip().split(' ')[0]
        label = imgs[i].strip().split(' ')[1]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        if model_name == 'Gaze360':
            img = gaze_transform_deal(img)  # 得到gazelstm的输入，7张连续，此处7张一样
        elif model_name == 'Head_posture':
            transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
            img = transform(img)
        else:
            img = get_test_transform()(img).unsqueeze(0)
        if torch.cuda.is_available():
            if model_name not in ['Head_posture']:
                img = img.cuda()
        with torch.no_grad():
            if model_name == 'Gaze360':
                _, _, out = model(img)
                feature = out.view(out.size(1), -1).squeeze(1)
            elif model_name == 'Head_posture':
                _, head_dcit = model([img])
                feature = head_dcit['module.roi_heads.box_head.fc7'][0]
            else:
                model_feature = trainer.build_model_feature(model)
                out = model_feature(img)
                feature = out.view(out.size(1), -1).squeeze(1)
            # if i > 1000:
            #     print(model_name, (time() - start_predict_time)/1000)
            #     break
        features[i, :] = feature.cpu().numpy()
        labels.append(label)
    #
    pickle.dump(features, open(feature_path, 'wb'))
    pickle.dump(labels, open(label_path, 'wb'))
    print('CNN features obtained and saved.')


def classifier_training(feature_path, label_path, save_path):
    '''
    训练分类器
    '''
    print('Pre-extracted features and labels found. Loading them ...')
    features = pickle.load(open(feature_path, 'rb'))
    labels = pickle.load(open(label_path, 'rb'))
    # classifier = SVC(C=1,kernel='linear')
    classifier = MLPClassifier()
    # classifier = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
    # classifier = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
    # classifier = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
    # max_features=50, max_depth=40, min_samples_leaf=4)
    # classifier = GaussianNB()
    print(".... Start fitting this classifier ....")
    classifier.fit(features, labels)
    print("... Training process is down. Save the model ....")
    joblib.dump(classifier, save_path)
    print("... model is saved ...")


def classifier_pred(model_path, feature, label, printwrong=True):
    # ---------------------
    model_path = model_path
    # ---------------------
    '''
    得到测试集的预测结果
    '''
    features = pickle.load(open(feature, 'rb'))
    labels = pickle.load(open(label, 'rb'))
    print("... loading the model ...")
    classifier = joblib.load(model_path)
    print("... load model done and start predicting ...")
    predict = classifier.predict(features)
    # print(type(predict))
    # print(predict.shape)
    # print(ids)
    cor = 0
    if printwrong:
        with open(VAL_LABEL_DIR, 'r') as f:
            imgs = f.readlines()
        if not os.path.exists(WRONG_TEST_DIR):
            os.makedirs(WRONG_TEST_DIR)
        print('Wrong test has saved!')
    prediction = predict.tolist()
    y_pred = list(map(int, prediction))
    y_true = list(map(int, labels))
    # 打印错误
    if printwrong:
        for idx, data in enumerate(labels):
            if int(prediction[idx] == labels[idx]) == 0:
                subdir_fullpath = imgs[idx].strip().split(' ')[0].replace('\\', '/')
                image_fullname = subdir_fullpath.split('/')[-1]
                copyfile(subdir_fullpath, os.path.join(WRONG_TEST_DIR, image_fullname))
    classify_report = classification_report(y_true, y_pred, digits=3)
    print('classify_report : \n', classify_report)
    #
    # C = confusion_matrix(y_true, y_pred, labels=[0,1])
    # plt.matshow(C, cmap=plt.cm.Greens)
    # plt.colorbar()
    # for i in range(len(C)):
    #     for j in range(len(C)):
    #         plt.annotate(C[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    # plt.ylabel('Unconscious')
    # plt.xlabel('Normal')
    # plt.show()

    # ROC
    probas_ = classifier.predict_proba(features)
    print(probas_)
    fpr, tpr, thresholds = roc_curve(y_true, probas_[:, 1])
    roc_auc = auc(fpr, tpr)  # 计算auc的值
    # 绘制roc曲线
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='LR ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# model_name = ['Head_posture_trial0','Face_cls_trial0','Gaze360_trial0','resnet18_trial0','resnet18last_trial0','resnet18preimg_trial0',
#                      'resnet50last_trial0']


test_tag = True  # True:全部运行,包含保存分类集和测试集特征，False：仅测试机器学习分类器
if __name__ == "__main__":
    if not os.path.exists(dataset_name + '/train.txt') or not os.path.exists(dataset_name + '/val.txt'):
        preprocess()
    # if test_tag:
    #     #### 保存训练集特征
    #     with open(TRAIN_LABEL_DIR, 'r') as f:
    #         imgs = f.readlines()
    #     train_feature_path = feature_path + '/psdufeature.pkl'
    #     train_label_path = feature_path + '/psdulabel.pkl'
    #     save_feature(train_feature_path, train_label_path)

    # ## #训练并保存分类器
    # train_feature_path = feature_path + '/psdufeature.pkl'
    # train_label_path = feature_path + '/psdulabel.pkl'
    # save_path = feature_path + '/ml.m'
    # classifier_training(train_feature_path, train_label_path, save_path)
    #
    # begin_time = time()
    # ######################################################################
    # # #保存测试集特征
    if test_tag:
        with open(VAL_LABEL_DIR, 'r') as f:
            imgs = f.readlines()
        test_feature_path = feature_path + '/valfeature.pkl'
        test_label_path = feature_path + '/vallabel.pkl'
        save_feature(test_feature_path, test_label_path)

    # ## #预测结果
    # test_feature_path = feature_path + '/valfeature.pkl'
    # test_label_path = feature_path + '/vallabel.pkl'
    # save_path = feature_path + '/ml.m'
    # classifier_pred(save_path, test_feature_path, test_label_path, printwrong=printwrong)
    # end_time = time()
    # run_time = end_time - begin_time
    # print("test one image is {:05.5f} ms".format(float(run_time / 1844 * 1000)))
    #
