import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import cv2
import datetime

# CSVファイルの読み込み
csv_path = 'train.csv'
df = pd.read_csv(csv_path)

# ラベルを数値に変換（例：A→0, B→1, ...）
labels = sorted(df['Label'].unique())
label_to_index = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

print(f"Unique labels: {labels}")
print(f"Number of unique labels: {num_classes}")

def preprocess_image(image):
    """
    画像の前処理を行う関数
    """
    # RGB画像をグレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # ノイズ除去
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 大津の二値化
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 反転（白背景に黒文字）
    inverted_binary = 255 - binary

    # 正規化（0-1の範囲に）
    normalized = inverted_binary.astype(np.float32) / 255.0

    # 画像を32x32にリサイズ
    resized = cv2.resize(normalized, (32, 32))

    # チャンネル次元を追加
    return np.expand_dims(resized, axis=-1)

# データ読み込みと整形
images = []
targets = []

# 前処理後の画像を保存するディレクトリを作成
preprocessed_save_dir = 'preprocessed_train_images'
if not os.path.exists(preprocessed_save_dir):
    os.makedirs(preprocessed_save_dir)

for i, (_, row) in enumerate(df.iterrows()):
    try:
        # ピクセル値を読み込んで正規化（0〜1）
        pixel_values = np.array(list(map(int, row['Pixel Values'].split())), dtype=np.uint8)
        size = int(np.sqrt(len(pixel_values)))
        img = pixel_values.reshape((size, size))
        
        # 前処理を適用
        img = preprocess_image(img)
        
        # 前処理後の画像を保存
        img_to_save = (img * 255).astype(np.uint8)
        save_path = os.path.join(preprocessed_save_dir, f"train_preprocessed_{i}.png")
        cv2.imwrite(save_path, img_to_save)
        
        images.append(img)
        targets.append(label_to_index[row['Label']])
    except Exception as e:
        print(f"Error processing row {i}: {str(e)}")
        continue

# データの準備
x = np.array(images)
y = to_categorical(targets, num_classes=num_classes)

print(f"Input data shape: {x.shape}")
print(f"Target data shape: {y.shape}")
print(f"Number of samples: {len(x)}")

def create_model():
    model = Sequential([
        # 第1層の畳み込みブロック
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 第2層の畳み込みブロック
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 第3層の畳み込みブロック
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # 全結合層
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # モデルのコンパイル
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model():
    # モデルの作成
    model = create_model()
    model.summary()

    # データ拡張の設定
    datagen = ImageDataGenerator(
        rotation_range=10,        # 回転範囲
        width_shift_range=0.1,    # 水平シフト
        height_shift_range=0.1,   # 垂直シフト
        zoom_range=0.1,           # ズーム範囲
        fill_mode='nearest'       # 空白部分の埋め方
    )

    # モデル学習
    history = model.fit(
        datagen.flow(x, y, batch_size=1),
        epochs=100,
        steps_per_epoch=len(x),
        verbose=1
    )

    # モデルの保存
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.save(os.path.join(save_dir, 'ocr_model.keras'))
    print(f"モデルを {save_dir}/ocr_model.keras に保存しました")

    return model, history

if __name__ == "__main__":
    model, history = train_model()