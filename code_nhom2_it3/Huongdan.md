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

Hướng dẫn yolo

1. File yolo3.cfg (Cấu hình mô hình)
   File này định nghĩa kiến trúc của mạng nơ-ron YOLO. Trong trường hợp của YOLOv3, nó thường là yolov3.cfg.

Cấu trúc chính của file:
Convolutional Layers: YOLOv3 sử dụng các lớp tích chập để trích xuất đặc trưng từ hình ảnh.
Residual Blocks: Đây là các khối giúp cải thiện khả năng học và giảm mất mát thông tin qua các lớp.
Anchor Boxes: Các hộp dự đoán trước được định nghĩa trong cấu hình để giúp mô hình nhận diện đối tượng có kích thước và hình dạng khác nhau.
Detection Layers: YOLOv3 sử dụng 3 lớp phát hiện để xử lý các đối tượng ở các kích thước khác nhau (lớp 13x13, 26x26, 52x52).
Có thể chỉnh sửa file .cfg để:

Thay đổi số lớp (num_classes).
Điều chỉnh các tham số mạng như kích thước đầu vào (width, height).

2. File .weights (Trọng số đã huấn luyện)
   File yolov3.weights chứa trọng số đã được huấn luyện sẵn, tức là giá trị của các tham số trong mạng nơ-ron.

Trọng số này được:
Huấn luyện trên COCO Dataset, một bộ dữ liệu với 80 danh mục đối tượng phổ biến (người, xe cộ, động vật, v.v.).
Sử dụng để tăng tốc phát triển vì không cần huấn luyện từ đầu.
Có thể thay thế file trọng số này bằng các trọng số khác, ví dụ:

Trọng số tùy chỉnh nếu bạn huấn luyện mô hình trên dữ liệu riêng.
Trọng số nhẹ hơn nếu muốn tốc độ nhanh hơn.

3. File .names (Tên danh mục đối tượng)
   File coco.names chứa danh sách các danh mục đối tượng mà YOLO có thể nhận diện. Ví dụ:
   python
   person
   bicycle
   car
   motorbike
   aeroplane
   bus
   ...
   Mỗi dòng đại diện cho một danh mục.
   Thứ tự các dòng tương ứng với chỉ số danh mục mà YOLO dự đoán. Ví dụ:
   0 -> person
   1 -> bicycle
   2 -> car
   Nếu bạn huấn luyện mô hình trên một tập dữ liệu tùy chỉnh, bạn cần tạo một file .names mới chứa danh mục đối tượng trong dữ liệu.

YOLO
Mô hình YOLO (You Only Look Once) là một phương pháp phổ biến trong nhận diện đối tượng (object detection) dùng trong thị giác máy tính (computer vision). YOLO đặc biệt vì tính thời gian thực và khả năng nhận diện nhiều đối tượng trong một bức ảnh chỉ trong một lần duy nhất. Cơ chế hoạt động của YOLO có thể được tóm tắt qua các điểm chính dưới đây:

1. Cách YOLO hoạt động:
   YOLO tiếp cận vấn đề nhận diện đối tượng một cách toàn diện và hiệu quả, phân loại và dự đoán tất cả các đối tượng trong một bức ảnh trong một lần chạy duy nhất. Điều này có nghĩa là YOLO không cần phải thực hiện nhiều lần quét (như các phương pháp truyền thống như R-CNN), mà chỉ cần một lần chạy qua toàn bộ ảnh.

2. Cấu trúc mô hình YOLO:
   Input (Dữ liệu đầu vào): Mô hình nhận vào một ảnh có kích thước cố định, ví dụ: 416x416 hoặc 608x608.
   Grid Cells (Các ô lưới): YOLO chia ảnh thành một lưới. Mỗi ô lưới sẽ chịu trách nhiệm phát hiện các đối tượng trong khu vực đó. Ví dụ, nếu ảnh là 416x416, và bạn chia nó thành một lưới 13x13, thì mỗi ô trong lưới này sẽ dự đoán các đối tượng trong phần ảnh tương ứng với ô đó.
   Bounding Box (Hộp bao quanh): Mỗi ô lưới không chỉ dự đoán một đối tượng, mà dự đoán n hộp bao quanh (bounding boxes). Mỗi hộp này sẽ có:
   X, Y: Vị trí tâm của hộp so với ô lưới.
   Width (W), Height (H): Kích thước của hộp.
   Confidence: Độ tin cậy rằng hộp chứa một đối tượng (theo mô hình YOLO).
   Class Probability (Xác suất lớp): Mỗi ô lưới cũng sẽ dự đoán xác suất mà đối tượng thuộc về mỗi lớp trong danh mục (ví dụ: người, xe, chó,...).

3. Cách YOLO dự đoán:
   YOLO không chỉ dự đoán bounding boxes cho các đối tượng mà còn xác định lớp của đối tượng đó, chẳng hạn như xe, người, chó, v.v. YOLO đưa ra ba thành phần quan trọng:

   Bounding box prediction: Dự đoán vị trí và kích thước của hộp bao quanh đối tượng.
   Objectness score: Một chỉ số thể hiện khả năng ô lưới đó chứa một đối tượng.
   Class probabilities: Dự đoán xác suất thuộc về từng lớp đối tượng.

4. Các cải tiến chính của YOLO (YOLOv3):
   Dự đoán ở nhiều cấp độ: YOLOv3 sử dụng ba kích thước lưới khác nhau để nhận diện các đối tượng nhỏ, vừa và lớn. Điều này giúp mô hình nhận diện các đối tượng có kích thước khác nhau trong cùng một ảnh.
   Residual Blocks: Các khối residual giúp giảm hiện tượng mất mát thông tin qua các lớp của mạng, đồng thời giúp mô hình học tốt hơn.
   Anchor Boxes: Thay vì chỉ dự đoán một hộp bao quanh, YOLOv3 sử dụng Anchor boxes để dự đoán các hộp bao quanh ở nhiều tỉ lệ khác nhau, giúp tăng khả năng dự đoán chính xác đối tượng.

5. Ưu điểm của YOLO:
   Nhanh: Một trong những điểm mạnh của YOLO là khả năng dự đoán đối tượng trong thời gian thực. YOLO có thể xử lý hàng trăm khung hình mỗi giây, điều này làm cho nó phù hợp với các ứng dụng như giám sát video trực tiếp.
   Đơn giản và hiệu quả: YOLO chỉ cần một lần chạy duy nhất để hoàn thành việc nhận diện đối tượng, điều này giúp giảm độ phức tạp so với các mô hình như R-CNN, Fast R-CNN, hay Faster R-CNN.
   Dễ triển khai: YOLO có thể được triển khai nhanh chóng trong các ứng dụng nhận diện đối tượng với bộ dữ liệu tùy chỉnh.

6. Nhược điểm của YOLO:
   Khó nhận diện đối tượng nhỏ: Mặc dù YOLO rất nhanh, nhưng đôi khi nó không nhận diện tốt các đối tượng rất nhỏ trong ảnh, đặc biệt là khi chúng ở gần rìa của các ô lưới.
   Độ chính xác thấp hơn so với các mô hình khác: Trong một số trường hợp, độ chính xác của YOLO có thể không cao bằng các mô hình khác như Faster R-CNN, đặc biệt khi đối tượng có hình dạng phức tạp hoặc không có đặc điểm rõ ràng.
7. Cách YOLOv3 được huấn luyện:
   Dataset: Để huấn luyện YOLO, bạn cần có một bộ dữ liệu lớn với các đối tượng được gắn nhãn. Bộ dữ liệu phổ biến cho YOLO là COCO (Common Objects in Context), trong đó có hàng nghìn đối tượng thuộc nhiều lớp khác nhau.
   Loss Function: YOLO sử dụng một hàm mất mát kết hợp giữa các thành phần:
   Mất mát vị trí: Đo lường sai số giữa vị trí thật của bounding box và vị trí dự đoán.
   Mất mát xác suất đối tượng: Đo lường độ tin cậy dự đoán rằng một ô chứa đối tượng.
   Mất mát lớp: Đo lường sai lệch giữa lớp thực tế và lớp dự đoán.
