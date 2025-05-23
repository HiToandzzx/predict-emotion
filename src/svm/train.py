import os
import cv2
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skimage.feature import hog

def preprocess_image(image_path, face_cascade_path=None):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Chuyển ảnh sang grayscale để giảm độ phức tạp và tập trung vào các đặc trưng cấu trúc.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt chó nếu có file cascade huấn luyện riêng
    # if face_cascade_path is not None:
    #     dog_face_cascade = cv2.CascadeClassifier(face_cascade_path)
    #     faces = dog_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    #     if len(faces) > 0:
    #         x, y, w, h = faces[0]
    #         gray = gray[y:y + h, x:x + w]

    # Resize về kích thước 128x128
    gray = cv2.resize(gray, (128, 128))
    return gray


# Trích xuất đặc trưng HOG (Histogram of Oriented Gradients) từ ảnh grayscale đã được resize.
def extract_features(image):
    features, _ = hog(
        image,
        orientations=9, # Số lượng hướng gradient (20/1o)3
        pixels_per_cell=(8, 8), # Kích thước của mỗi ô (cell)
        cells_per_block=(2, 2), # Số lượng cell trong một khối (block)
        block_norm='L2-Hys', # Phương pháp chuẩn hóa các khối đặc trưng (cân bằng độ sáng)
        visualize=True
    )
    return features


def load_data(csv_path, data_dir, face_cascade_path=None):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for idx, row in df.iterrows():
        filename = row['filename']
        label = row['label']

        image_path = os.path.join(data_dir, label, filename) # Đường dẫn đến các ảnh data

        img_processed = preprocess_image(image_path, face_cascade_path) # Gọi hàm để tiền xử lí các ảnh

        if img_processed is None:
            print(f"Warning: Không đọc được ảnh {image_path}.")
            continue

        features = extract_features(img_processed) # Gọi hàm để trích xuất đặc trưng của những ảnh data
        X.append(features) # vector đặc trưng
        y.append(label) # nhãn0

    return np.array(X), np.array(y) # Convert thành mảng numpy


def main():
    # Đường dẫn tới file CSV và thư mục data
    csv_path = 'data/labels.csv'  # File CSV của bạn
    data_dir = 'data'  # Thư mục chứa subfolder angry/happy/relaxed/sad

    # (Tuỳ chọn) Đường dẫn tới cascade để phát hiện khuôn mặt chó (nếu có)
    # dog_face_cascade_path = None  # Hoặc 'dog_face.xml' nếu bạn có file cascade

    # 1. Đọc dữ liệu và trích xuất đặc trưng
    X, y = load_data(csv_path, data_dir)
    print("Kích thước tập dữ liệu:", X.shape, y.shape)

    if len(X) < 2:
        print("Dữ liệu quá ít để huấn luyện.")
        return

    # 2. Chia dữ liệu train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Thiết lập mô hình SVM với probability=True và Grid Search
    param_grid = {
        'C': [0.1, 1, 10], # Điều chỉnh mức độ phạt cho các mẫu bị phân loại sai.
        'gamma': ['scale', 'auto'], # Hệ số của kernel
        'kernel': ['rbf', 'linear'] # Phân chia phi tuyến tính (Radial Basis Function)
    }

    svm = SVC(probability=True) # Khởi tạo mô hình SVM (xác suất)
    # Tạo đối tượng GridSearchCV
    # Truyền mô hình SVM cùng với grid siêu tham số
    # Thiết lập số lượng fold cho cross-validation
    # Tìm kiếm bộ tham số tốt nhất cho mô hình
    grid = GridSearchCV(svm, param_grid, refit=True, cv=2) # Chia dữ liệu thành 2 phần, lặp lại 2 lần và lấy điểm TB (kiểm tra chéo)

    # 4. Huấn luyện mô hình
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)

    # 5. Đánh giá mô hình
    y_pred = grid.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # 6. Tính và in độ chính xác của mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # 7. Lưu mô hình đã huấn luyện
    model_path = 'svm_dog_emotion_model.pkl'
    joblib.dump(grid.best_estimator_, model_path)
    print(f"Mô hình đã được lưu tại: {model_path}")


if __name__ == "__main__":
    main()
