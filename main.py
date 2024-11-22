import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().splitlines()

# Variables for storing file path, images, and detection count
file_path = None
img = None
detected_img = None
photo = None
detected_photo = None
object_count = 0

# Function to select an image
def select_image():
    global file_path, img, photo
    file_path = filedialog.askopenfilename(filetypes=[
        ("JPG files", "*.jpg"),
        ("JPEG files", "*.jpeg"),
        ("PNG files", "*.png"),
    ])
    if not file_path:
        return  # No file selected

    if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        messagebox.showerror("Lỗi định dạng", "Chỉ chấp nhận các file ảnh định dạng JPG, JPEG, hoặc PNG.")
        file_path = None  # Reset file_path if invalid
        return

    # Load and process the image here
    print("File hợp lệ:", file_path)
    # Load and display the selected image
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((300, 300), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img_pil)

    original_image_label.config(image=photo)
    original_image_label.image = photo

# Function to perform object detection and display results
def show_result():
    global file_path, img, detected_img, detected_photo, object_count
    if not file_path:
        messagebox.showerror("Lỗi", "Vui lòng chọn ảnh trước khi nhấn 'Xác nhận và Đếm'.")
        return

    # Object detection logic
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes, confidences, class_ids = [], [], []
    detected_objects = []

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

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if isinstance(indexes, tuple):
        indexes = indexes[0]

    detected_img = img.copy()
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = round(confidences[i], 2)
        detected_objects.append(f"{label}: {confidence:.2f}")
        color = (0, 255, 0)
        cv2.rectangle(detected_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(detected_img, f'{label}: {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    object_count = len(indexes.flatten())

    # Update the detected image
    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    detected_img_pil = Image.fromarray(detected_img_rgb)
    detected_img_pil = detected_img_pil.resize((300, 300), Image.LANCZOS)
    detected_photo = ImageTk.PhotoImage(detected_img_pil)

    detected_image_label.config(image=detected_photo)
    detected_image_label.image = detected_photo

    # Update detected objects list and object count
    object_list_frame.delete("1.0", tk.END)
    for obj_info in detected_objects:
        object_list_frame.insert(tk.END, f"{obj_info}\n")
    object_count_label.config(text=f"Tổng số đối tượng: {object_count}")
# Adjust the layout for Tkinter GUI setup
root = tk.Tk()
root.title("Hệ thống xác nhận và đếm đối tượng trong ảnh")
root.geometry("900x650")

# Load background image
bg_image = Image.open("bg_icon/Ul.png")  # Path to your background image
bg_image = bg_image.resize((900, 650), Image.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Set background label
background_label = tk.Label(root, image=bg_photo)
background_label.place(relwidth=1, relheight=1)

# Title label
title_label = tk.Label(root, text="Hệ thống xác nhận và đếm đối tượng trong ảnh", font=("Arial", 18, "bold"), bg="#111F69", fg="white")
title_label.grid(row=0, column=0, columnspan=3, pady=10)

# Left Frame for original image
left_frame = tk.Frame(root, width=300, height=400, bg="white")
left_frame.grid(row=1, column=0, padx=10, pady=10)
left_frame.grid_propagate(False)
original_image_label = tk.Label(left_frame)
original_image_label.place(relx=0.5, rely=0.5, anchor="center")

# Center Frame for detected image
center_frame = tk.Frame(root, width=300, height=400, bg="white")
center_frame.grid(row=1, column=1, padx=10, pady=10)
center_frame.grid_propagate(False)
detected_image_label = tk.Label(center_frame)
detected_image_label.place(relx=0.5, rely=0.5, anchor="center")

# Right Frame for detected object list and count
right_frame = tk.Frame(root, width=300, height=400, bg="white")
right_frame.grid(row=1, column=2, padx=10, pady=10, sticky="n")
right_frame.grid_propagate(False)

# Object count label
object_count_label = tk.Label(right_frame, text="Tổng số đối tượng: 0", font=("Arial", 12, "bold"), fg="red", bg="white")
object_count_label.pack(pady=10)

# Small detected objects list frame
object_list_frame = tk.Text(right_frame, wrap="word", font=("Arial", 10), bg="white", fg="black", height=10, width=30)
object_list_frame.pack(pady=5)

# Load icons
select_icon = Image.open("bg_icon/download.png")  # Path to your "Chọn ảnh" icon
select_icon = select_icon.resize((30, 30), Image.LANCZOS)
select_icon_photo = ImageTk.PhotoImage(select_icon)

confirm_icon = Image.open("bg_icon/verified.png")  # Path to your "Xác nhận và Đếm" icon
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
