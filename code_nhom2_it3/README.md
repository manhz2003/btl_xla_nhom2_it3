ĐỀ TÀI SỐ 14: XÂY DỰNG HỆ THỐNG XÁC NHẬN ĐỐI TƯỢNG VÀ ĐẾM ĐỐI TƯỢNG TRONG ẢNH

Hướng dẫn build

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
