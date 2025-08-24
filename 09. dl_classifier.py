
import numpy as np
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.applications import ResNet50
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import random
import tensorflow as tf
import pickle
from collections import OrderedDict
import os

# CNN Modificada
def defineCNN(class_names):
    cnn = models.Sequential()
    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(GlobalAveragePooling2D())
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dense(len(class_names), activation='sigmoid'))
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['f1_score'])
    return cnn

##############################################################################
# CNN14
def defineCNN14(class_names):
    cnn14 = models.Sequential()
    cnn14.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)))
    cnn14.add(BatchNormalization())
    cnn14.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(MaxPooling2D((2, 2)))
    cnn14.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(MaxPooling2D((2, 2)))
    cnn14.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(MaxPooling2D((2, 2)))
    cnn14.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(MaxPooling2D((2, 2)))
    cnn14.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(Conv2D(2048, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(Conv2D(2048, (3, 3), activation='relu', padding='same'))
    cnn14.add(BatchNormalization())
    cnn14.add(GlobalAveragePooling2D())
    cnn14.add(Dense(2048, activation='relu'))
    cnn14.add(Dense(len(class_names), activation='sigmoid'))
    cnn14.compile(optimizer='adam', loss='binary_crossentropy', metrics=['f1_score'])
    return cnn14

##############################################################################
#ResNet50
def defineResNet50(class_names):
    input_layer = Input(shape = (128, 128, 1))
    x = layers.Concatenate()([input_layer, input_layer, input_layer])
    base_model = ResNet50(include_top = False, weights = 'imagenet', input_tensor = x)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation = 'relu')(x)
    output_layer = Dense(len(class_names), activation = 'sigmoid')(x)
    model = models.Model(inputs = input_layer, outputs = output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['f1_score'])
    return model
######################################################################################

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

###############################################################################

if __name__ == "__main__":
    
    cqt_path = "./data/cqt_image.pkl"    
    df = pd.read_pickle(cqt_path)
    
    folds_path = "./data/kfold_split.pkl"
    with open(folds_path, "rb") as f:
        folds = pickle.load(f)
    
    path = "./log/results/cnnLogs/"
    if not os.path.exists(path):
        os.makedirs(path)
        print("The new directory is created!")
    else:
        print("cnnLogs folder already exists!")
    
    all_results_dict = {}
    
    original_label_order = list(OrderedDict.fromkeys(label for sample in df["instrument"] for label in sample))
    mlb = MultiLabelBinarizer(classes=original_label_order)
    
    for seed in range(len(folds)):
        set_seed(seed)
        
        print(f"Rodando a fold {seed + 1} / {len(folds)}")


        train_idx = folds[seed]['train_ids']
        test_idx = folds[seed]['test_ids']
        
        df_train = df[df['id'].isin(train_idx)].copy()
        df_test = df[df['id'].isin(test_idx)].copy()

        print(f"  -> Usando {len(df_train)} amostras para treino e {len(df_test)} para teste.")
        
        X_train = np.stack(df_train['cqt_image'].values)
        X_test = np.stack(df_test['cqt_image'].values)
        y_train = mlb.fit_transform(df_train['instrument'])
        y_test = mlb.transform(df_test['instrument'])
        

        batch_size = 512
        validation_split_value = 0.2

        
        # CNN
        cnn = defineCNN(mlb.classes_)
        print(f"  -> Rodando a CNN para o fold {seed + 1}")
        early_stop = EarlyStopping(monitor='val_loss', mode="min", patience=20, restore_best_weights=True, verbose=0)
        csv_logger = CSVLogger(f"./log/results/cnnLogs/log_history_cnn_seed_{seed}.csv", separator=",", append=False)
        cnn.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=validation_split_value, callbacks=[early_stop, csv_logger], verbose=0)
        y_pred_cnn = cnn.predict(X_test)

        # CNN14
        cnn14 = defineCNN14(mlb.classes_)
        print(f"  -> Rodando a CNN14 para o fold {seed + 1}")
        early_stop = EarlyStopping(monitor='val_loss', mode="min", patience=20, restore_best_weights=True, verbose=0)
        csv_logger = CSVLogger(f"./log/results/cnnLogs/log_history_cnn14_seed_{seed}.csv", separator=",", append=False)
        cnn14.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=validation_split_value, callbacks=[early_stop, csv_logger], verbose=0)
        y_pred_cnn14 = cnn14.predict(X_test)

        # ResNet50
        resnet = defineResNet50(mlb.classes_)
        print(f"  -> Rodando a ResNet para o fold {seed + 1}")
        early_stop = EarlyStopping(monitor='val_loss', mode="min", patience=20, restore_best_weights=True, verbose=0)
        csv_logger = CSVLogger(f"./log/results/cnnLogs/log_history_resnet_seed_{seed}.csv", separator=",", append=False)
        resnet.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_split=validation_split_value, callbacks=[early_stop, csv_logger], verbose=0)
        y_pred_resnet = resnet.predict(X_test)


        results_df = pd.DataFrame({
            'id': df_test['id'].values,
            'y_pred_cnn': [row.tolist() for row in y_pred_cnn],
            'y_pred_cnn14': [row.tolist() for row in y_pred_cnn14],
            'y_pred_resnet': [row.tolist() for row in y_pred_resnet],
            'y_true': [row.tolist() for row in y_test]
        })
        
        all_results_dict[seed] = results_df

    print("\nEstrutura do dicionário de resultados:")
    for key, df_value in all_results_dict.items():
        print(f"  Fold {key}: DataFrame com {len(df_value)} linhas.")
    
    # Salva o dicionário final em um novo arquivo pickle
    output_path = os.path.join(path, "cnn_preds_full_10folds_dict.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(all_results_dict, f)

    print(f"\nDicionário com resultados dos 10 folds completos salvo em {output_path}")