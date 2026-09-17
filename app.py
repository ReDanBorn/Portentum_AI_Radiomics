import os
import streamlit as st
from PIL import Image
import pydicom
import numpy as np
import cv2
from ultralytics import YOLO

# Загрузка обученных моделей
model_fracture = YOLO('Fractures.pt')  # Замени на путь к модели переломов
model_medimp = YOLO('MedImp.pt')  # Замени на путь к модели инородных тел

# Функции для обработки DICOM
def process_dicom(file_path):
    dicom_data = pydicom.dcmread(file_path)
    processed_image = apply_advanced_windowing(dicom_data)
    enhanced_image = enhance_image_contrast(processed_image)
    filtered_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
    return Image.fromarray(filtered_image)

def apply_advanced_windowing(dicom_data):
    intercept = dicom_data.RescaleIntercept if "RescaleIntercept" in dicom_data else 0
    slope = dicom_data.RescaleSlope if "RescaleSlope" in dicom_data else 1
    image = dicom_data.pixel_array * slope + intercept
    if "WindowCenter" in dicom_data and "WindowWidth" in dicom_data:
        center = dicom_data.WindowCenter
        width = dicom_data.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]
        min_val = center - width / 2
        max_val = center + width / 2
        image = np.clip(image, min_val, max_val)
    else:
        min_val, max_val = np.min(image), np.max(image)
        image = np.clip(image, min_val, max_val)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def enhance_image_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Интерфейс Streamlit
st.title("Медицинская обработка изображений")
st.header("Загрузите файл DICOM, PNG, JPG, JPEG или WEBP")

uploaded_file = st.file_uploader("Выберите файл", type=["dcm", "png", "jpg", "jpeg", "webp"])

if uploaded_file:
    # Определяем тип файла
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
    
    if file_extension == ".dcm":
        st.write("Обрабатывается DICOM-файл...")
        processed_image = process_dicom(uploaded_file)
    else:
        st.write("Обрабатывается изображение...")
        processed_image = Image.open(uploaded_file)

    # Сохранение обработанного изображения во временный файл
    temp_image_path = "temp_image.png"
    processed_image.save(temp_image_path)

    # Отображаем изображение
    st.image(processed_image, caption="Обработанное изображение", use_container_width=True)

    # Выполняем предсказания
    st.write("Выполняется предсказание модели...")
    fracture_prediction = model_fracture.predict(temp_image_path)[0].probs.top1
    medimp_prediction = model_medimp.predict(temp_image_path)[0].probs.top1

    # Вывод результатов
    fracture_result = "Перелом: Обнаружен" if fracture_prediction == 1 else "Перелом: Не обнаружен"
    medimp_result = "Инородное тело: Обнаружено" if medimp_prediction == 1 else "Инородное тело: Не обнаружено"

    st.success(fracture_result)
    st.success(medimp_result)
