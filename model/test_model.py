import pickle
import numpy as np
import pandas as pd
from keras.models import load_model
from keras import Input, layers, Model
from keras.src.callbacks import EarlyStopping, Callback, TensorBoard, ModelCheckpoint
from keras.src.layers import Dense, Attention, StackedRNNCells, Dropout, Concatenate
from keras.src.optimizers import Adam, RMSprop
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm

from helper import get_eachpart_report, split_train_test, get_new_train_0, get_new_train_1
from prediction import get_feature, get_image_paths

# 假设每个预训练模型的数据结果的维度分别为 dim1, dim2, dim3, dim4

dim_list = [1024, 512, 1024, 512, 512]

# 运行该代码前提是保存好了相应特征文件
model_name = ['Head_posture_trial0', 'Gaze360_trial0', 'sp_model_5_trial0'
    , 'face_feature_trial0', 'openpose_trial0']
feature_path_list = ['./trained_models/' + i + '/ml_features' for i in model_name]
def muti_feature_fusion_cls(model_choice):
    train_feature_fu = []
    test_feature_fu = []
    for choice in model_choice:
        # 训练数据
        train_feature_path = feature_path_list[choice] + '/psdufeature.pkl'
        train_label_path = feature_path_list[choice] + '/psdulabel.pkl'
        train_features = np.array(pickle.load(open(train_feature_path, 'rb'))) * 1
        train_labels = np.array(pickle.load(open(train_label_path, 'rb')))
        train_feature_fu.append(train_features)
        test_feature_path = feature_path_list[choice] + '/valfeature.pkl'
        test_label_path = feature_path_list[choice] + '/vallabel.pkl'
        test_features = pickle.load(open(test_feature_path, 'rb'))
        test_labels = np.array(pickle.load(open(test_label_path, 'rb')))
        test_feature_fu.append(test_features)
    return train_feature_fu, train_labels, test_feature_fu, test_labels


def train():
    from keras import backend as K
    import tensorflow as tf
    K.clear_session()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    model_choice = [0, 1, 2, 3, 4]
    input_feature1 = Input(shape=(dim_list[model_choice[0]],))
    input_feature2 = Input(shape=(dim_list[model_choice[1]],))
    input_feature3 = Input(shape=(dim_list[model_choice[2]],))
    input_feature4 = Input(shape=(dim_list[model_choice[3]],))
    input_feature5 = Input(shape=(dim_list[model_choice[4]],))

    # 将特征通过全连接层映射到较低维度
    query1 = Dense(units=32)(input_feature1)
    query2 = Dense(units=32)(input_feature2)
    query3 = Dense(units=32)(input_feature3)
    query4 = Dense(units=32)(input_feature4)
    query5 = Dense(units=32)(input_feature5)

    key1 = Dense(units=32)(input_feature1)
    key2 = Dense(units=32)(input_feature2)
    key3 = Dense(units=32)(input_feature3)
    key4 = Dense(units=32)(input_feature4)
    key5 = Dense(units=32)(input_feature5)

    value1 = Dense(units=32)(input_feature1)
    value2 = Dense(units=32)(input_feature2)
    value3 = Dense(units=32)(input_feature3)
    value4 = Dense(units=32)(input_feature4)
    value5 = Dense(units=32)(input_feature5)
    # 注意力机制
    key = key1 + key2 + key3 + key4 + key5
    # key = key1 + key2
    attention_scores1 = Attention()([query1, key, value1])
    attention_scores2 = Attention()([query2, key, value2])
    attention_scores3 = Attention()([query3, key, value3])
    attention_scores4 = Attention()([query4, key, value4])
    attention_scores5 = Attention()([query5, key, value5])

    attention_scores1 = Dropout(0.5)(attention_scores1)
    attention_scores2 = Dropout(0.5)(attention_scores2)
    attention_scores3 = Dropout(0.5)(attention_scores3)
    attention_scores4 = Dropout(0.5)(attention_scores4)
    attention_scores5 = Dropout(0.5)(attention_scores5)

    # 分别应用注意力权重到对应的特征向量上
    weighted_feature1 = attention_scores1 * value1
    weighted_feature2 = attention_scores2 * value2
    weighted_feature3 = attention_scores3 * value3
    weighted_feature4 = attention_scores4 * value4
    weighted_feature5 = attention_scores5 * value5
    #
    # 合并加权后的特征
    merged_vector = Concatenate()(
        [weighted_feature1, weighted_feature2
            , weighted_feature3
            , weighted_feature4
            , weighted_feature5
         ])
    # 添加其他层，根据需要构建模型

    # ...
    dense_layer1 = Dense(units=128, activation='relu')(merged_vector)
    dense_layer1 = Dropout(0.5)(dense_layer1)
    # 输出层，根据任务需求调整激活函数和单元数
    output_layer = Dense(units=1, activation='sigmoid')(dense_layer1)
    # 创建模型

    model = Model(inputs=[input_feature1, input_feature2, input_feature3, input_feature4, input_feature5
                          ],
                  outputs=output_layer)
    train_feature_path = 'new_feature/final_data/train_feature.pkl'
    train_label_path = 'new_feature/final_data/train_label.pkl'
    test_feature_path = 'new_feature/final_data/test_feature.pkl'
    test_label_path = 'new_feature/final_data/test_label.pkl'

    train_feature_fu = pickle.load(open(train_feature_path, 'rb'))
    train_labels = pickle.load(open(train_label_path, 'rb'))
    test_feature_fu = pickle.load(open(test_feature_path, 'rb'))
    test_labels = pickle.load(open(test_label_path, 'rb'))
    train_labels = train_labels.astype(int)

    test_labels = test_labels.astype(int)
    train_feature_fu = [train_feature_fu[i] for i in model_choice]
    test_feature_fu = [test_feature_fu[i] for i in model_choice]

    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    checkpoint = ModelCheckpoint('muti_head_env_label_2.h5',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max',
                                 verbose=1)
    # 假设你有一个训练集 train_features 和相应的标签 train_labels，还有一个验证集 val_features 和相应的标签 val_labels
    model.fit(train_feature_fu, train_labels, epochs=300, validation_data=(test_feature_fu, test_labels), workers=20
              , callbacks=[early_stopping
            # , checkpoint
                           ]
              )
    predict = model.predict(test_feature_fu)
    y_pred = np.where(predict.squeeze() > 0.5, 1, 0).tolist()
    y_true = list(map(int, test_labels))
    classify_report = classification_report(y_true, y_pred, digits=3)
    print(classify_report)
    # a = get_eachpart_report(y_true, y_pred)
    # acc = np.sum(np.array(y_pred) == np.array(y_true)) / len(y_pred)


def main_test_model():
    muti_head = load_model('muti_head_env_label_5_best.h5')

    test_feature_path = 'new_feature/final_data/test_feature.pkl'
    test_label_path = 'new_feature/final_data/test_label.pkl'
    test_feature_fu = pickle.load(open(test_feature_path, 'rb'))
    test_labels = pickle.load(open(test_label_path, 'rb'))
    test_labels = test_labels.astype(int)
    predict = muti_head.predict(test_feature_fu)
    y_pred = np.where(predict.squeeze() > 0.5, 1, 0).tolist()
    y_true = list(map(int, test_labels))
    get_eachpart_report(y_true, y_pred)
    # existing_df = pd.read_excel("mlp_mh.xlsx")
    # existing_df.loc[len(existing_df)] = a

    # 保存 DataFrame 到 Excel 文件
    # existing_df.to_excel("mlp_mh.xlsx", index=False)


if __name__ == '__main__':
    # train()
    main_test_model()
