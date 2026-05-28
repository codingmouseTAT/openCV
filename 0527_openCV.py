import os
import cv2
from deepface import DeepFace

input_folder = "face_data"
output_folder = "face_data_ok"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"已自動建立輸出資料夾：{output_folder}")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if not os.path.exists(input_folder):
    print(f"❌ 錯誤：找不到來源資料夾 [{input_folder}]！")
    print("請先建立該資料夾，並將 Sad、Angry、Happy 的圖片移入其中。")
    exit()

file_list = os.listdir(input_folder)

image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
image_files = [f for f in file_list if f.endswith(image_extensions)]

print(f"在 [{input_folder}] 中共找到 {len(image_files)} 張圖片，開始處理")

success_count = 0

for filename in image_files:

    img_path = os.path.join(input_folder, filename)
    save_path = os.path.join(output_folder, filename)

    print(f"正在處理: {filename} ... ", end="")

    img = cv2.imread(img_path)
    if img is None:
        print("讀取失敗")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:

        try:
            result = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                enforce_detection=False
            )
            emotion = result[0]["dominant_emotion"]

            cv2.putText(img, f"Face not framed ({emotion})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        except Exception:
            cv2.putText(img, "No Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            if face_img.size == 0:
                continue

            try:
                result = DeepFace.analyze(
                    img_path=face_img,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="skip"
                )
                emotion = result[0]["dominant_emotion"]

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(img, emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                pass

    cv2.imwrite(save_path, img)
    print("完成已導出。")
    success_count += 1

print(f"任務完成！ {success_count}/{len(image_files)} 張圖片。")
print(f" [{output_folder}] 查結果。")