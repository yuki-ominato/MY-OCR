import os
import csv
from PIL import Image
import numpy as np
import cv2

# 入力画像
threshold = 256  # ←ここを好きな値に（例：100〜200など）
img_rgb = Image.open("ascii.png").convert("RGBA")
img = np.array(img_rgb)
print(img.shape)

img_np = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
print(img_np.shape)
for i in range(img_np.shape[0]):
    for j in range(img_np.shape[1]):
        img_np[i][j] = (img[i][j][0] * 0.2126 + img[i][j][1] * 0.7152 + img[i][j][2] * 0.0722) * img[i][j][3] / 255.0

# 二値化処理を追加
_, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 出力フォルダ
output_folder = "char_images"
os.makedirs(output_folder, exist_ok=True)

# グリッド情報
rows, cols = 6, 16
cell_h = img_np.shape[0] // rows
cell_w = img_np.shape[1] // cols

# グリッド上の文字配置
char_grid = [
    ' !"#$%&\'()*+,-./',
    '0123456789:;<=>?',
    '@ABCDEFGHIJKLMNO',
    'PQRSTUVWXYZ[\\]^_',
    '`abcdefghijklmno',
    'pqrstuvwxyz{|}~ ',
    '                ',
]

# 対象文字
target_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz?"

# 文字位置マッピング
char_positions = {}
for row in range(len(char_grid)):
    for col in range(len(char_grid[row])):
        ch = char_grid[row][col]
        if ch in target_chars:
            char_positions[ch] = (row, col)
            # print(ch, row, col)

# CSV出力準備
csv_rows = []
csv_path = "train.csv"

# 38行38列のchar_imgを作成
char_img = np.zeros((cell_h, cell_w+3), dtype=np.uint8)
offset_x = 6

# 切り出しと保存ループ
for ch, (row, col) in char_positions.items():
    y1, y2 = row * cell_h, (row + 1) * cell_h
    x1, x2 = max(col * cell_w - offset_x, 0), min((col + 1) * cell_w - offset_x, 566)
    diff_min, diff_max = col * cell_w - offset_x, ((col + 1) * cell_w - offset_x) - 566
    for i in range(y1, y2):
        for j in range(x1, x2):
            if(x1==0):
                char_img[i-y1][j-x1-diff_min] = img_np[i][j]
            elif(x2==566):
                char_img[i-y1][j-x1+diff_max] = img_np[i][j]
            else:
                char_img[i - y1][j - x1] = img_np[i][j]
    
    # char_img = img_np[y1:y2, x1:x2]
    print(char_img.shape)

    # ファイル名"
    if ch.isupper():
        file_name = f"upper_{ch}.png"
    elif ch.islower():
        file_name = f"lower_{ch}.png"
    elif ch.isdigit():
        file_name = f"num_{ch}.png"
    else:
        file_name = f"sym_{ord(ch)}.png"
    image_path = os.path.join(output_folder, file_name)
    # print(file_name)

    # 画像保存
    Image.fromarray(char_img).save(image_path)

    # ピクセルデータをflattenしてCSV用に保存
    pixels_flat = " ".join(str(val) for val in char_img.flatten())
    csv_rows.append([file_name, pixels_flat, ch])

# CSVに書き出し
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Image Path", "Pixel Values", "Label"])
    writer.writerows(csv_rows)

print("CSVと画像の保存が完了しました。")
