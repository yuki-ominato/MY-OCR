import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.optimizers import Adam
from PIL import Image
import os
import cv2
import datetime

# CSVファイルの読み込み
csv_path = 'train.csv'  # ←あなたのCSVファイル名に合わせて変更
df = pd.read_csv(csv_path)

# ラベルを数値に変換（例：A→0, B→1, ...）
labels = sorted(df['Label'].unique())
label_to_index = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

# データ読み込みと整形
images = []
targets = []

# 前処理後の画像を保存するディレクトリを作成
preprocessed_save_dir = 'preprocessed_train_images'
if not os.path.exists(preprocessed_save_dir):
    os.makedirs(preprocessed_save_dir)

def preprocess_image(image):
    """
    画像の前処理を行う関数
    """
    # RGB画像をグレースケールに変換
    if len(image.shape) == 3:
        # RGB画像の場合、グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        # すでにグレースケールの場合
        gray = image

    # ノイズ除去
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # 大津の二値化（白背景に黒文字を想定し、二値化）
    # THRESH_BINARY + THRESH_OTSU は、文字が0、背景が255になる
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # PIL.ImageOps.invert と同等の処理 (0-255の範囲で反転)
    inverted_binary = 255 - binary

    # 正規化（0-1の範囲に）
    normalized = inverted_binary.astype(np.float32) / 255.0

    return normalized

for i, (_, row) in enumerate(df.iterrows()):
    # ピクセル値を読み込んで正規化（0〜1）
    pixel_values = np.array(list(map(int, row['Pixel Values'].split())), dtype=np.uint8)
    size = int(np.sqrt(len(pixel_values)))  # 正方形画像を仮定
    img = pixel_values.reshape((size, size))
    if i == 0:
        print(img.shape)
    
    # 前処理を適用
    img = preprocess_image(img)
    if i == 0:
        print(img.shape)
    
    # 前処理後の画像を保存
    # normalized画像は0-1のfloat32なので、255を掛けてuint8に戻す
    img_to_save = (img * 255).astype(np.uint8)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(preprocessed_save_dir, f"train_preprocessed_{i}.png")
    # print(i)
    cv2.imwrite(save_path, img_to_save)
    
    # print(img)
    images.append(img)
    targets.append(label_to_index[row['Label']])

# 画像データを正しい形状に変換（バッチサイズ, 高さ, 幅, チャンネル数）
x = np.array(images).reshape(-1, size, size, 1)
print(x[0].shape)
y = to_categorical(targets, num_classes=num_classes)

# モデル構築
def model_training():
    model = Sequential([
        # 特徴抽出層
        Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(size, size, 1)),
        MaxPooling2D((2, 2)),
        
        # 分類層
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    # 学習率を下げてモデルをコンパイル
    optimizer = Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # モデル学習
    history = model.fit(
        x, y,
        epochs=100,  # エポック数を増やす
        batch_size=1,
        verbose=1
    )

    # 最終結果の表示
    final_loss, final_accuracy = model.evaluate(x, y, verbose=0)
    print(f"Final Accuracy on Training Data: {final_accuracy:.4f}")
    print(f"Final Loss on Training Data: {final_loss:.4f}")

    # モデルの保存
    # 保存先ディレクトリの作成
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # モデルの保存（新しい形式で保存）
    model.save(os.path.join(save_dir, 'font_recognition_model.keras'))
    print(f"モデルを {save_dir}/font_recognition_model.keras に保存しました")

    # 重みのみを保存する場合
    model.save_weights(os.path.join(save_dir, 'font_recognition.weights.h5'))
    print(f"モデルの重みを {save_dir}/font_recognition.weights.h5 に保存しました")

model_training()