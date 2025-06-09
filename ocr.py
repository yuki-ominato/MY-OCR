import os
import sys
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import ctypes
import keyboard
import threading
import pyperclip
import numpy as np
from keras.models import load_model
import cv2
from scipy import ndimage
import datetime

# 自作OCRモデルの読み込み
model = load_model("saved_models/font_recognition_model.keras")

# クラスラベル
LABELS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz?")

# DPIスケーリング無効化
ctypes.windll.user32.SetProcessDPIAware()

def preprocess_image(image):
    """
    画像の前処理を行う関数
    """
    img_rgb = image.convert("RGBA")
    img = np.array(img_rgb)

    img_np = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            img_np[i][j] = (img[i][j][0] * 0.2126 + img[i][j][1] * 0.7152 + img[i][j][2] * 0.0722) * img[i][j][3] / 255.0
    
    # 二値化処理を追加 (以前の大津の二値化に戻す)
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def detect_characters(image):
    """
    文字領域を検出する関数
    """
    # 輪郭検出
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # print(f"[DEBUG] Detected {len(contours)} contours.") # デバッグ情報
    
    # 文字領域の抽出
    char_regions = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # print(f"[DEBUG] Contour {i}: x={x}, y={y}, w={w}, h={h}") # デバッグ情報
        if w > 10 and h > 10:  # 小さすぎる領域は除外
            char_regions.append((x, y, w, h))
    
    return char_regions

def postprocess_text(text):
    """
    認識結果の後処理を行う関数
    """
    # 空白文字の削除
    text = text.strip()
    
    # 連続する同じ文字の削除（例：'AA' → 'A'）
    result = ''
    for i in range(len(text)):
        if i == 0 or text[i] != text[i-1]:
            result += text[i]
    
    return result

def Check_Binary(img, check_name):
    # print(img.shape)
    # print(img)
    flag = False
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            num = img[h][w]
            if num != 0 and num != 255:
                print("f{check_name} is Not Binary")
                flag = True
                break
        if flag:
            break

class ScreenOCRApp:
    def __init__(self):
        self.start_x = None
        self.start_y = None
        self.rect = None

        self.root = tk.Tk()
        self.root.attributes("-alpha", 0.3)
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}+0+0")

        self.canvas = tk.Canvas(self.root, cursor="cross", bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        self.root.mainloop()

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )

    def on_move_press(self, event):
        cur_x, cur_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        self.root.destroy()

        x1 = int(min(self.start_x, end_x))
        y1 = int(min(self.start_y, end_y))
        x2 = int(max(self.start_x, end_x))
        y2 = int(max(self.start_y, end_y))

        # 画像の取得
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        
        # 前処理
        processed_img = preprocess_image(img)
        Check_Binary(processed_img, "processed_img")
        save_img = Image.fromarray(processed_img)
        save_img.save("debug_images/processed_img.png")
        
        # 文字検出
        char_regions = detect_characters(processed_img)
        
        # 文字認識
        recognized_text = ""
        # 保存用のディレクトリを作成
        save_dir = "debug_images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        for i, (x, y, w, h) in enumerate(char_regions):
            char_img = processed_img[y:y+h, x:x+w]
            char_img = Image.fromarray(char_img)
            char_img = char_img.resize((38, 38))
            img_array = np.array(char_img) / 255.0
            img_array = img_array.reshape(1, 38, 38, 1)

            # img_arrayを画像として保存
            save_array = (img_array[0, :, :, 0] * 255).astype(np.uint8)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"char_array_{timestamp}_{i}.png")
            cv2.imwrite(save_path, save_array)

            prediction = model.predict(img_array, verbose=0)
            label_index = np.argmax(prediction)
            print(label_index)
            char = LABELS[label_index]
            recognized_text += char
        
        # 後処理
        # final_text = postprocess_text(recognized_text)

        print("===== OCR結果（自作モデル） =====")
        # print(final_text)
        print(recognized_text)
        print("===============================")
        # pyperclip.copy(final_text)
        pyperclip.copy(recognized_text)
        print("▶ 結果をクリップボードにコピーしました！")

def run_ocr():
    threading.Thread(target=ScreenOCRApp).start()

def quit_app():
    print("🔚 プログラムを終了します")
    os._exit(0)

keyboard.add_hotkey('ctrl+shift+f', run_ocr)
keyboard.add_hotkey("ctrl+shift+q", quit_app)

print("🔍 Ctrl + Shift + F を押すとOCRモードが起動します（自作モデル）")
print("❌ Ctrl + Shift + Q で終了")

keyboard.wait()

