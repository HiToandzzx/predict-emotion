import cv2
import joblib
import matplotlib.pyplot as plt
from skimage.feature import hog

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # if face_cascade_path is not None:
    #     dog_face_cascade = cv2.CascadeClassifier(face_cascade_path)
    #     faces = dog_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    #     if len(faces) > 0:
    #         x, y, w, h = faces[0]
    #         gray = gray[y:y + h, x:x + w]

    gray = cv2.resize(gray, (128, 128))
    return gray


def extract_features(image):
    features, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features


def predict_emotion(image_path, model):

    img_processed = preprocess_image(image_path) # Tiền xử lí ảnh

    if img_processed is None:
        print(f"Không thể đọc hoặc xử lý ảnh: {image_path}")
        return None, None

    features = extract_features(img_processed) # Trích xuất đặc trưng
    # Lấy xác suất cho từng lớp
    prob = model.predict_proba([features])[0]
    # Lấy thứ tự các lớp dự đoán từ mô hình
    classes = model.classes_

    # Chuyển xác suất sang phần trăm làm tròn
    prob_percent = {cls: round(p * 100, 2) for cls, p in zip(classes, prob)}
    return prob_percent, img_processed


def main():
    # Đường dẫn tới mô hình đã lưu
    model_path = 'svm_dog_emotion_model.pkl'
    model = joblib.load(model_path)
    print("Mô hình đã được tải thành công.")

    # Đường dẫn tới cascade để phát hiện khuôn mặt chó (nếu có)
    dog_face_cascade_path = None  # Hoặc 'dog_face.xml'

    # Ảnh cần dự đoán
    test_image_path = 'test/img_1.png'  # Thay đường dẫn theo ảnh bạn muốn kiểm tra
    prob_percent, cropped_img = predict_emotion(test_image_path, model)

    if cropped_img is not None and prob_percent is not None:
        # Tạo chuỗi hiển thị xác suất cho từng emotion
        prob_text = "\n".join([f"{cls}: {pct}%" for cls, pct in prob_percent.items()])

        # Hiển thị ảnh và xác suất
        plt.imshow(cropped_img, cmap='gray')
        plt.title(prob_text, fontsize=8)
        plt.axis('off')
        plt.show()
    else:
        print("Không thể hiển thị ảnh hoặc dự đoán.")


if __name__ == "__main__":
    main()
