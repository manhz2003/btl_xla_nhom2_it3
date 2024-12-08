# cv2: Thư viện OpenCV để xử lý hình ảnh và nhận diện đối tượng.
# numpy: Dùng để xử lý mảng số liệu.
# tkinter: Dùng để xây dựng giao diện người dùng.
# PIL (Pillow): Hỗ trợ xử lý ảnh, chuyển đổi giữa định dạng OpenCV và Tkinter.
# os: Dùng để làm việc với hệ thống tệp.
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Load Yolo
# cv2.dnn.readNet: Tải mô hình YOLO với các file cấu hình và trọng số.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# classes: Danh sách các lớp đối tượng được mô hình nhận diện (ví dụ: người, xe, động vật, v.v.).
# coco.names: File chứa danh sách tên các lớp.
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()


# file_path: Đường dẫn tệp ảnh được chọn.
# img: Ảnh gốc đã đọc từ đường dẫn.
# detected_img: Ảnh sau khi đã qua nhận diện.
# photo và detected_photo: Ảnh được chuyển sang định dạng hiển thị trên Tkinter.
# object_count: Số lượng đối tượng nhận diện được.
file_path = None
img = None
detected_img = None
photo = None
detected_photo = None
object_count = 0

# Hàm chọn ảnh
# filedialog.askopenfilename: Mở hộp thoại để người dùng chọn ảnh.
# Kiểm tra xem tệp được chọn có đúng định dạng không (JPG, JPEG, PNG).
# Nếu hợp lệ, ảnh được tải bằng cv2.imread và chuyển đổi sang định dạng RGB để hiển thị trên giao diện bằng ImageTk.PhotoImage.
def select_image():
    global file_path, img, photo
    file_path = filedialog.askopenfilename(filetypes=[
        ("JPG files", "*.jpg"),
        ("JPEG files", "*.jpeg"),
        ("PNG files", "*.png"),
    ])
    if not file_path:
        return  # Không có tập tin nào được chọn

    if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        messagebox.showerror("Lỗi định dạng", "Chỉ chấp nhận các file ảnh định dạng JPG, JPEG, hoặc PNG.")
        file_path = None  # Đặt lại file_path nếu không hợp lệ
        return

    # Tải và xử lý hình ảnh ở đây
    print("File hợp lệ:", file_path)
    # Tải và hiển thị hình ảnh đã chọn
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((300, 300), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img_pil)

    original_image_label.config(image=photo)
    original_image_label.image = photo

# Nhận dạng đối tượng:
def show_result():
    global file_path, img, detected_img, detected_photo, object_count
    if not file_path:
        messagebox.showerror("Lỗi", "Vui lòng chọn ảnh trước khi nhấn 'Xác nhận và Đếm'.")
        return

    # logic nhận dạng đối tượng
    # Sử lý ảnh trước khu đưa vào mô hình
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    
    # hàm thực hiện nhận dạng getUnconnectedOutLayersNames
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes, confidences, class_ids = [], [], []
    detected_objects = []
    
    #Duyệt qua các đầu ra từ mô hình chọn ra lớp có độ tin cậy cao nhất
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Loại bỏ các hộp bao quanh trùng lặp
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    
    if isinstance(indexes, tuple):
        indexes = indexes[0]
        
    #Tạo một bản sao của ảnh gốc để vẽ các bounding boxes lên.
    detected_img = img.copy()
    
    #Duyệt qua tất cả các hộp giới hạn được chọn và vẽ chúng lên ảnh. Cũng hiển thị nhãn và độ tin cậy cho mỗi đối tượng phát hiện
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = round(confidences[i], 2)
        detected_objects.append(f"{label}: {confidence:.2f}")
        color = (0, 255, 0)
        cv2.rectangle(detected_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(detected_img, f'{label}: {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # hàm len để đếm số phần tử trong mảng 1 chiều đã được làm phẳng bằng flatten
    object_count = len(indexes.flatten())

    # Cập nhật hình ảnh đã phát hiện
    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    detected_img_pil = Image.fromarray(detected_img_rgb)
    detected_img_pil = detected_img_pil.resize((300, 300), Image.LANCZOS)
    detected_photo = ImageTk.PhotoImage(detected_img_pil)
    detected_image_label.config(image=detected_photo)
    detected_image_label.image = detected_photo

    # Cập nhật danh sách đối tượng được phát hiện và số lượng đối tượng
    object_list_frame.delete("1.0", tk.END)
    for obj_info in detected_objects:
        object_list_frame.insert(tk.END, f"{obj_info}\n")
    object_count_label.config(text=f"Tổng số đối tượng: {object_count}")
    
# Điều chỉnh bố cục cho thiết lập GUI Tkinter
root = tk.Tk()
root.title("Hệ thống xác nhận và đếm đối tượng trong ảnh")
root.geometry("900x650")

# Tải hình nền
bg_image = Image.open("bg_icon/Ul.png")
bg_image = bg_image.resize((900, 650), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Đặt nhãn nền
background_label = tk.Label(root, image=bg_photo)
background_label.place(relwidth=1, relheight=1)

#nhãn tiêu đề
title_label = tk.Label(root, text="Hệ thống xác nhận và đếm đối tượng trong ảnh", font=("Arial", 18, "bold"), bg="#111F69", fg="white")
title_label.grid(row=0, column=0, columnspan=3, pady=10)

# Khung bên trái cho hình ảnh gốc
left_frame = tk.Frame(root, width=300, height=400, bg="white")
left_frame.grid(row=1, column=0, padx=10, pady=10)
left_frame.grid_propagate(False)
original_image_label = tk.Label(left_frame)
original_image_label.place(relx=0.5, rely=0.5, anchor="center")

# Khung trung tâm cho hình ảnh được phát hiện
center_frame = tk.Frame(root, width=300, height=400, bg="white")
center_frame.grid(row=1, column=1, padx=10, pady=10)
center_frame.grid_propagate(False)
detected_image_label = tk.Label(center_frame)
detected_image_label.place(relx=0.5, rely=0.5, anchor="center")

# Khung bên phải cho danh sách và số lượng đối tượng được phát hiện
right_frame = tk.Frame(root, width=300, height=400, bg="white")
right_frame.grid(row=1, column=2, padx=10, pady=10, sticky="n")
right_frame.grid_propagate(False)

# Nhãn đếm đối tượng
object_count_label = tk.Label(right_frame, text="Tổng số đối tượng: 0", font=("Arial", 12, "bold"), fg="red", bg="white")
object_count_label.pack(pady=10)

# Khung danh sách các đối tượng nhỏ được phát hiện
object_list_frame = tk.Text(right_frame, wrap="word", font=("Arial", 10), bg="white", fg="black", height=10, width=30)
object_list_frame.pack(pady=5)

# Load icons
select_icon = Image.open("bg_icon/download.png")
select_icon = select_icon.resize((30, 30), Image.LANCZOS)
select_icon_photo = ImageTk.PhotoImage(select_icon)

# Đường dẫn tới icon "Xác nhận và Đếm" của bạn
confirm_icon = Image.open("bg_icon/verified.png")
confirm_icon = confirm_icon.resize((30, 30), Image.LANCZOS)
confirm_icon_photo = ImageTk.PhotoImage(confirm_icon)

button_select = tk.Button(
    root, text="Chọn ảnh", command=select_image, font=("Arial", 12),
    image=select_icon_photo, compound="left", bg="#4CAF50", fg="white"
)
button_select.grid(row=2, column=1, pady=(40, 10), padx=10, sticky="ew")

button_show = tk.Button(
    root, text="Xác nhận và Đếm", command=show_result, font=("Arial", 12),
    image=confirm_icon_photo, compound="left", bg="#2196F3", fg="white"
)
button_show.grid(row=3, column=1, pady=(0, 10), padx=10, sticky="ew")

root.mainloop()