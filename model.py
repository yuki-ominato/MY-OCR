import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# CSVファイルを読み込む
df = pd.read_csv('data.csv')

# 画像を読み込んで前処理する関数
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))  # 画像のリサイズ
    image = image.astype('float32') / 255  # ピクセル値の正規化
    return image

# 画像データとラベルのリストを作成
images = []
labels = []
for index, row in df.iterrows():
    img = load_image(row['Image Path'])
    images.append(img)
    labels.append(row['Label'])

# 画像とラベルをnumpy配列に変換
X = np.array(images)
y = np.array(labels)

# ラベルをOne-Hot Encoding
y = to_categorical(y)

# データセットを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの定義
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(np.unique(y)), activation='softmax'))

# モデルのコンパイル
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 学習の実行
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
