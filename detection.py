import sys
import face_recognition
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import glob
import config # 設定ファイル

threshold = config.threshold
emp_info = config.emp_info
mode = config.mode

#顔情報の初期化
fece_locations = []
face_encodings = []

# 登録画像の読み込み
image_paths = glob.glob("Image/*")
image_paths.sort()
known_face_encodings = []
known_face_names = []
checked_face = []

delimiter = "/" #Mac用 windows は ¥¥

for image_path in image_paths:
  im_name = image_path.split(delimiter)[-1].split(".")[0]
  image = face_recognition.load_image_file(image_path)
  face_encoding = face_recognition.face_encodings(image)[0]
  known_face_encodings.append(face_encoding)
  known_face_names.append(im_name)

video_capture = cv2.VideoCapture(0) #何番目のカメラを選ぶか

def main():
  # 処理フラグの初期化
  process_this_frame = True

  while True:
    # ビデオの単一フレームを取得
    _, frame = video_capture.read()

    # 時間を節約するために、フレームごとに処理をスキップ
    if process_this_frame:
      # 画像を縦横 1/4 に圧縮
      small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)

      # 顔の位置情報を検索
      face_locations = face_recognition.face_locations(small_frame)

      # 顔画像の符号化
      face_encodings = face_recognition.face_encodings(small_frame, face_locations)

      # 名前配列の初期化
      face_names = []

      for face_encoding in face_encodings:
        # 顔画像が登録画像と一致しているか検証
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, threshold) #threshold より小さければ True、大きければ False を返す
        name = "Unknown"

        # 顔画像と最も近い登録画像を候補とする
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
          name = known_face_names[best_match_index]

        face_names.append(name)
    
    # 処理フラグの切り替え
    process_this_frame = not process_this_frame

    # 位置情報の表示
    for (top, right, bottom, left), name in zip(face_locations, face_names):

      # 圧縮した画像の座標を復元
      top *= 4
      right *= 4
      bottom *= 4
      left *= 4

      # 顔領域に枠を描画
      cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2) #最後は枠の太さ

      # 顔領域の下に枠を表示
      cv2.rectangle(frame, (left, bottom-40), (right, bottom), (255, 0, 0), cv2.FILLED)
      font = cv2.FONT_HERSHEY_PLAIN
      cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

      # 本人確認
      if mode == 1 and name != "Unknown":
        check_passward(name)

    # 結果をビデオに表示
    cv2.imshow("Video", frame)

    # ESCキーで終了
    if cv2.waitKey(1) == 27: #1ミリ秒で動作させる 27がESCキー
      break

def check_passward(name):
  if name in checked_face:
    return 
  
  emp_pw = input(name + "さんのパスワードを入力してください")

  if emp_info[name] == emp_pw:
    print("出勤しました")
    checked_face.append(name)
  else:
    print("パスワードが不正です")


main()

video_capture.release()
cv2.destroyAllWindows()