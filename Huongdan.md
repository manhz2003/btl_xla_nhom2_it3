## 1. Tạo và kích hoạt môi trường ảo

### Trên macOS/Linux:

python3 -m venv myenv
source myenv/bin/activate

### Trên Windows:

python -m venv myenv
myenv\Scripts\activate

## 2. Cài đặt các thư viện cần thiết

Sau khi kích hoạt môi trường ảo, bạn cần cài đặt các thư viện từ tệp `requirements.txt`:

pip install -r requirements.txt

## 3. Tải file (`yolov3.weights`)

https://drive.google.com/drive/folders/1iZ0LN5RDbZWBmr4E8DexHTJouF4PTo3O

## 4. Chạy chương trình

python main.py

thư mục bg_icon chứa icon
thư mục data chứa ảnh data test

file coco.names
Danh sách lớp đối tượng: Tệp này liệt kê tên của các lớp đối tượng mà mô hình có thể nhận diện. Ví dụ, các lớp có thể bao gồm: "người", "xe hơi", "chó", "mèo", "ghế", "laptop", v.v.

file yolov3.cfg
mô hình YOLO (You Only Look Once) được sử dụng trong việc huấn luyện một mạng thần kinh sâu (deep neural network) cho bài toán nhận dạng đối tượng (object detection). Đây là một phần trong việc huấn luyện mạng neural với các lớp convolutional (lọc), downsampling, upsampling và các lớp YOLO cụ thể để phân loại và nhận dạng các đối tượng trong ảnh.
