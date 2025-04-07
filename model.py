import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from PIL import Image

# CSVファイルの読み込み
csv_path = 'font_data.csv'  # ←あなたのCSVファイル名に合わせて変更
df = pd.read_csv(csv_path)

# ラベルを数値に変換（例：A→0, B→1, ...）
labels = sorted(df['Label'].unique())
label_to_index = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

# データ読み込みと整形
images = []
targets = []

for _, row in df.iterrows():
    # ピクセル値を読み込んで正規化（0〜1）
    pixel_values = np.array(list(map(int, row['Pixel Values'].split())), dtype=np.uint8)
    size = int(np.sqrt(len(pixel_values)))  # 正方形画像を仮定
    img = pixel_values.reshape((size, size)) / 255.0
    images.append(img)
    targets.append(label_to_index[row['Label']])

x = np.array(images).reshape(-1, size, size, 1)
y = to_categorical(targets, num_classes=num_classes)

# モデル構築
model = Sequential([
    Flatten(input_shape=(size, size, 1)),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# モデルコンパイル
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# モデル学習（少量データのためエポック少なめ）
history = model.fit(x, y, epochs=50, batch_size=1, verbose=0)

# 最終結果の表示
final_loss, final_accuracy = model.evaluate(x, y, verbose=0)
print(f"Final Accuracy on Training Data: {final_accuracy:.4f}")
print(f"Final Loss on Training Data: {final_loss:.4f}")
