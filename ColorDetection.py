import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from colorthief import ColorThief
from tkinter import Tk, filedialog

def color_segmentation(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red
    lower_red1 = np.array([0, 50, 70])
    upper_red1 = np.array([9, 255, 255])
    lower_red2 = np.array([159, 50, 70])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_count = np.sum(red_mask > 0)

    # Black
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    black_mask = cv2.inRange(hsv_image, lower_black, upper_black)
    black_count = np.sum(black_mask > 0)

    # Yellow
    lower_yellow = np.array([25, 50, 70])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    yellow_count = np.sum(yellow_mask > 0)

    # Green
    lower_green = np.array([36, 50, 70])
    upper_green = np.array([89, 255, 255])
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    green_count = np.sum(green_mask > 0)

    # Blue
    lower_blue = np.array([90, 50, 70])
    upper_blue = np.array([128, 255, 255])
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_count = np.sum(blue_mask > 0)

    # Gray
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 18, 230])
    gray_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)
    
    # White
    lower_white = np.array([0, 0, 231])  # [0, 0, 30]
    upper_white = np.array([180, 18, 255])  # [180, 30, 255]
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    gray_white_mask = cv2.bitwise_or(gray_mask, white_mask)
    whitegray_count = np.sum(gray_white_mask > 0)

    """
    cv2.imshow('Red ', cv2.bitwise_and(image, image, mask=red_mask))
    cv2.imshow('Black', cv2.bitwise_and(image, image, mask=black_mask))
    cv2.imshow('Yellow', cv2.bitwise_and(image, image, mask=yellow_mask))
    cv2.imshow('Green', cv2.bitwise_and(image, image, mask=green_mask))
    cv2.imshow('Blue', cv2.bitwise_and(image, image, mask=blue_mask))
    cv2.imshow('Gray-White', cv2.bitwise_and(image, image, mask=gray_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    color_count = {
        'Kirmizi': red_count,
        'Siyah': black_count,
        'Sari': yellow_count,
        'Yesil': green_count,
        'Mavi': blue_count,
        'Gri-Beyaz': whitegray_count
    }

    if not color_count or all(count == 0 for count in color_count.values()):
        return None

    dominant_color = max(color_count, key=color_count.get)
    return dominant_color

def dominant_color_detection(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return None

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h_histogram = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])

    dominant_hue_bin = np.argmax(h_histogram)

    # Baskın rengin HSV aralığını ayarları (biraz genişletilmiş)
    lower_hue = max(0, dominant_hue_bin - 10)
    upper_hue = min(180, dominant_hue_bin + 10)

    lower_saturation = 50  # max(0, dominant_saturation_bin - 10)
    upper_saturation = 255  # min(255, dominant_saturation_bin + 10)

    lower_value = 50  # max(0, dominant_value_bin - 10)
    upper_value = 255  # min(255, dominant_value_bin + 10)

    lower_dominant = np.array([lower_hue, lower_saturation, lower_value])
    upper_dominant = np.array([upper_hue, upper_saturation, upper_value])

    mask = cv2.inRange(hsv_frame, lower_dominant, upper_dominant)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    #color_segment = color_segmentation(result)

    # Görüntüleri göster
    plt.plot(h_histogram)
    plt.title('Ton Histogramı')
    plt.xlabel('Ton Değeri')
    plt.ylabel('Piksel Sayısı')
    plt.show()

    frame = cv2.resize(frame,(640,640))
    mask = cv2.resize(mask,(640,640))
    result = cv2.resize(result,(640,640))

    cv2.imshow('Orijinal', frame)
    cv2.imshow('Maske', mask)
    cv2.imshow('Result', result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #return color_segment

def ycrcb_color_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    gri_lower = np.array([100, 100, 100])
    gri_upper = np.array([150, 150, 150])

    siyah_lower = np.array([[0, 128, 128]])
    siyah_upper = np.array([20, 130, 130])

    beyaz_lower = np.array([200, 0, 0])
    beyaz_upper = np.array([255, 20, 20])

    gri_mask = cv2.inRange(ycrcb, gri_lower, gri_upper)
    siyah_mask = cv2.inRange(ycrcb, siyah_lower, siyah_upper)
    beyaz_mask = cv2.inRange(ycrcb, beyaz_lower, beyaz_upper)
    gri_beyaz_mask = cv2.bitwise_or(gri_mask, beyaz_mask)

    # Maskelerin tersini alma (renk olmayan alanlar beyaz olacak)
    #gri_beyaz_mask = cv2.bitwise_not(gri_beyaz_mask)
    #siyah_mask = cv2.bitwise_not(siyah_mask)

    gri_beyaz_masked_img = cv2.bitwise_and(img, img, mask=gri_beyaz_mask)
    siyah_masked_img = cv2.bitwise_and(img, img, mask=siyah_mask)

    img = cv2.resize(img, (640, 640))
    gri_beyaz_masked_img = cv2.resize(gri_beyaz_masked_img, (640, 640))
    siyah_masked_img = cv2.resize(siyah_masked_img, (640, 640))

    cv2.imshow('Orijinal Image', img)
    cv2.imshow('YCRCB Masked Image/GriBeyaz', gri_beyaz_masked_img)
    cv2.imshow('YCRCB Masked Image/Siyah', siyah_masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def lab_color_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    gri_lower = np.array([0, -20, -20])
    gri_upper = np.array([100, 20, 20])

    siyah_lower = np.array([0, -128, -128])
    siyah_upper = np.array([50, 127, 127])

    beyaz_lower = np.array([60, -20, -20])
    beyaz_upper = np.array([100, 20, 20])

    gri_mask = cv2.inRange(lab, gri_lower, gri_upper)
    siyah_mask = cv2.inRange(lab, siyah_lower, siyah_upper)
    beyaz_mask = cv2.inRange(lab, beyaz_lower, beyaz_upper)

    gri_beyaz_mask = cv2.bitwise_or(gri_mask, beyaz_mask)
    gri_beyaz_masked = cv2.bitwise_and(img, img, mask=gri_beyaz_mask)
    siyah_masked = cv2.bitwise_and(img, img, mask=siyah_mask)

    img = cv2.resize(img, (640, 640))
    gri_beyaz_masked_img = cv2.resize(gri_beyaz_masked, (640, 640))
    siyah_masked_img = cv2.resize(siyah_masked, (640, 640))

    cv2.imshow('Orijinal Image', img)
    cv2.imshow('LAB Masked Image/GriBeyaz', gri_beyaz_masked_img)
    cv2.imshow('LAB Masked Image/Siyah', siyah_masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rgb_color_detection(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return

    siyah_lower = np.array([0, 0, 0])  # Siyah için RGB değerleri (yaklaşık: 0, 0, 0)
    siyah_upper = np.array([20, 20, 20])  # Siyah için üst sınır

    beyaz_lower = np.array([200, 200, 200])  # Beyaz için RGB değerleri (yaklaşık: 255, 255, 255)
    beyaz_upper = np.array([255, 255, 255])  # Beyaz için üst sınır

    gri_lower = np.array([100, 100, 100])  # Gri için RGB değerleri (yaklaşık: 128, 128, 128)
    gri_upper = np.array([200, 200, 200])  # Gri için üst sınır

    siyah_mask = cv2.inRange(img, siyah_lower, siyah_upper)
    beyaz_mask = cv2.inRange(img, beyaz_lower, beyaz_upper)
    gri_mask = cv2.inRange(img, gri_lower, gri_upper)

    gri_beyaz_mask = cv2.bitwise_or(gri_mask, beyaz_mask)

    siyah_masked_img = cv2.bitwise_and(img, img, mask=siyah_mask)
    beyaz_masked_img = cv2.bitwise_and(img, img, mask=beyaz_mask)
    gri_masked_img = cv2.bitwise_and(img, img, mask=gri_mask)
    gri_beyaz_masked_img = cv2.bitwise_and(img, img, mask=gri_beyaz_mask)

    img_resized = cv2.resize(img, (640, 640))
    siyah_resized = cv2.resize(siyah_masked_img, (640, 640))
    beyaz_resized = cv2.resize(beyaz_masked_img, (640, 640))
    gri_resized = cv2.resize(gri_masked_img, (640, 640))
    gri_beyaz_resized = cv2.resize(gri_beyaz_masked_img, (640, 640))

    # Görüntüleri gösterme
    cv2.imshow('Orijinal Image', img_resized)
    cv2.imshow('Siyah Renk Maskesi', siyah_resized)
    cv2.imshow('Beyaz Renk Maskesi', beyaz_resized)
    cv2.imshow('Gri Renk Maskesi', gri_resized)
    cv2.imshow('Gri/Beyaz Renk Maskesi', gri_beyaz_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize_channel(channel):
    min_val = np.min(channel)
    max_val = np.max(channel)
    if max_val - min_val == 0:
        return channel  # Bölme sıfır hatasını önle
    normalized_channel = (channel - min_val) * 255 / (max_val - min_val)
    return np.uint8(normalized_channel)

def combined_color_detection(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return

    maskEdge = frame;

    # Görüntü işleme adımları
    #img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(frame, (11, 11), 0) #img_gray
    img_blur2 = cv2.medianBlur(img_blur, 5)
    img_blur3 = cv2.bilateralFilter(img_blur2, 9, 75, 75)

    """
    _, th = cv2.threshold(img_blur3, 0, 255, cv2.THRESH_OTSU)
    img_canny = cv2.Canny(th, 0, 130, 3)
    img_dilated = cv2.dilate(img_canny, (5, 5), iterations=2)
    contours, hierarchy = cv2.findContours(img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    maskEdge = np.zeros_like(img_gray)
    cv2.drawContours(maskEdge, contours, -1, 255, cv2.FILLED)

    # Maskeyi tersine çevirme
    mask_inv = cv2.bitwise_not(maskEdge)

    # Maskeyi orijinal görüntüye uygulama
    masked_image = cv2.bitwise_and(frame, frame, mask=mask_inv)
    """
    #colorized = cv2.cvtColor(img_blur3,cv2.COLOR_GRAY2BGR)
    hsv_frame = cv2.cvtColor(img_blur3, cv2.COLOR_BGR2HSV) #masked_image cv2.COLOR_BGR2HSV
    h, s, v = cv2.split(hsv_frame)

    # Histogram tabanlı baskın renk aralığı
    h_histogram = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
    s_histogram = cv2.calcHist([s], [0], None, [256], [0, 256])
    v_histogram = cv2.calcHist([v], [0], None, [256], [0, 256])
    """
    s_normalized = normalize_channel(s)
    v_normalized = normalize_channel(v)

    hsv_normalized = cv2.merge([h, s_normalized, v_normalized])
    """
    s_equized = cv2.equalizeHist(s)
    v_equized = cv2.equalizeHist(v)
    hsv_frame = cv2.merge([h, s_equized, v_equized])

    dominant_hue_bin = np.argmax(h_histogram)
    lower_hue = max(0, dominant_hue_bin - 10)
    upper_hue = min(180, dominant_hue_bin + 10)
    lower_dominant = np.array([lower_hue, 50, 70])
    upper_dominant = np.array([upper_hue, 255, 255])
    dominant_mask = cv2.inRange(hsv_frame, lower_dominant, upper_dominant)
    dominant_result = cv2.bitwise_and(frame, frame, mask=dominant_mask)

    # Önceden tanımlanmış renk aralıkları
    color_masks = {
        'Kirmizi': (
            (np.array([0, 50, 70]), np.array([9, 255, 255])),
            (np.array([159, 50, 70]), np.array([180, 255, 255]))
        ),
        'Siyah': ((np.array([0, 0, 0]), np.array([180, 255, 50])),),#180,255,20
        'Sari': ((np.array([25, 50, 70]), np.array([35, 255, 255])),),
        'Yesil': ((np.array([36, 50, 70]), np.array([89, 255, 255])),),
        'Mavi': ((np.array([90, 50, 70]), np.array([128, 255, 255])),),
        'Gri': ((np.array([0, 0, 40]), np.array([180, 18, 230])),),
        'Beyaz': ((np.array([0, 0, 200]), np.array([180, 20, 255])),) #(np.array([0, 0, 200]), np.array([180, 20, 200])
    }
    """
    color_pixel_counts = {}
    for color_name, ranges in color_masks.items():
        mask = np.zeros(h.shape, dtype=np.uint8)  # Tek kanallı maske oluştur
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
        color_pixel_counts[color_name] = np.count_nonzero(mask)

    dominant_color = max(color_pixel_counts, key=color_pixel_counts.get) if color_pixel_counts else None

    """
    color_counts = {}
    for color_name, ranges in color_masks.items():
        mask = np.zeros_like(dominant_mask)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
        color_counts[color_name] = np.count_nonzero(cv2.bitwise_and(dominant_result, dominant_result, mask=mask))

    dominant_color = max(color_counts, key=color_counts.get) if color_counts else None

    # Görselleştirme
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(h_histogram)
    plt.title('Ton Histogramı')
    plt.xlabel('Ton Değeri')
    plt.ylabel('Piksel Sayısı')

    plt.subplot(1, 3, 2)
    plt.plot(s_histogram)
    plt.title('Doygunluk Histogramı')
    plt.xlabel('Doygunluk Değeri')
    plt.ylabel('Piksel Sayısı')

    plt.subplot(1, 3, 3)
    plt.plot(h_histogram)
    plt.title('Value Histogramı')
    plt.xlabel('Value Değeri')
    plt.ylabel('Piksel Sayısı')

    plt.tight_layout()
    plt.show()

    frame = cv2.resize(frame, (640, 640))
    dominant_mask = cv2.resize(dominant_mask, (640, 640))
    dominant_result = cv2.resize(dominant_result, (640, 640))
    hsv_frame = cv2.resize(hsv_frame, (640, 640))
    #hsv_normalized = cv2.resize(hsv_normalized, (640, 640))

    """
    mask_resized = cv2.resize(maskEdge, (640, 640))
    mask_inv_resized = cv2.resize(mask_inv, (640, 640))
    masked_image_resized = cv2.resize(masked_image, (640, 640))

    cv2.imshow('Maske', mask_resized)
    cv2.imshow('Ters Maske', mask_inv_resized)
    cv2.imshow('Maskelenmis Goruntu', masked_image_resized)
    """

    cv2.imshow('Orijinal', frame)
    cv2.imshow("Histogram Equized", hsv_frame)
    #cv2.imshow('Normalized After Equization', hsv_normalized)
    cv2.imshow('Baskin Maske', dominant_mask)
    cv2.imshow('Baskin Sonuc', dominant_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dominant_color

def combined_rgb_hsv_detection(image_path):
    # Görüntüyü yükle
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return

    # ---------------------------------------------
    # RGB Uzayında Siyah, Beyaz ve Gri Tespiti
    # ---------------------------------------------
    siyah_lower = np.array([0, 0, 0])  # Siyah için RGB değerleri (yaklaşık: 0, 0, 0)
    siyah_upper = np.array([20, 20, 20])  # Siyah için üst sınır

    beyaz_lower = np.array([200, 200, 200])  # Beyaz için RGB değerleri (yaklaşık: 255, 255, 255)
    beyaz_upper = np.array([255, 255, 255])  # Beyaz için üst sınır

    gri_lower = np.array([100, 100, 100])  # Gri için RGB değerleri (yaklaşık: 128, 128, 128)
    gri_upper = np.array([200, 200, 200])  # Gri için üst sınır

    # Maskeleri oluştur
    siyah_mask = cv2.inRange(frame, siyah_lower, siyah_upper)
    beyaz_mask = cv2.inRange(frame, beyaz_lower, beyaz_upper)
    gri_mask = cv2.inRange(frame, gri_lower, gri_upper)

    # Maskeleri birleştir
    gri_beyaz_mask = cv2.bitwise_or(gri_mask, beyaz_mask)

    # Maskeleri uygulama
    siyah_masked_img = cv2.bitwise_and(frame, frame, mask=siyah_mask)
    beyaz_masked_img = cv2.bitwise_and(frame, frame, mask=beyaz_mask)
    gri_masked_img = cv2.bitwise_and(frame, frame, mask=gri_mask)
    gri_beyaz_masked_img = cv2.bitwise_and(frame, frame, mask=gri_beyaz_mask)

    # ---------------------------------------------
    # HSV Uzayında Renkli Renk Tespiti
    # ---------------------------------------------
    # Görüntüyü HSV'ye çevir
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv_frame)

    # Histogram tabanlı baskın renk aralığı
    h_histogram = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
    s_histogram = cv2.calcHist([s], [0], None, [256], [0, 256])
    v_histogram = cv2.calcHist([v], [0], None, [256], [0, 256])

    s_equized = cv2.equalizeHist(s)
    v_equized = cv2.equalizeHist(v)
    hsv_frame = cv2.merge([h, s_equized, v_equized])

    # Dominant rengi bulmak için ton histogramı üzerinde analiz yapıyoruz
    dominant_hue_bin = np.argmax(h_histogram)
    lower_hue = max(0, dominant_hue_bin - 10)
    upper_hue = min(180, dominant_hue_bin + 10)
    lower_dominant = np.array([lower_hue, 50, 70])
    upper_dominant = np.array([upper_hue, 255, 255])
    dominant_mask = cv2.inRange(hsv_frame, lower_dominant, upper_dominant)
    dominant_result = cv2.bitwise_and(frame, frame, mask=dominant_mask)

    # Önceden tanımlanmış renk aralıkları
    color_masks = {
        'Kirmizi': (
            (np.array([0, 50, 70]), np.array([9, 255, 255])),
            (np.array([159, 50, 70]), np.array([180, 255, 255]))
        ),
        'Siyah': ((np.array([0, 0, 0]), np.array([180, 255, 20])),),
        'Sari': ((np.array([25, 50, 70]), np.array([35, 255, 255])),),
        'Yesil': ((np.array([36, 50, 70]), np.array([89, 255, 255])),),
        'Mavi': ((np.array([90, 50, 70]), np.array([128, 255, 255])),),
        'Gri': ((np.array([0, 0, 40]), np.array([180, 18, 230])),),
        'Beyaz': ((np.array([0, 0, 0]), np.array([0, 0, 255])),)
    }#(np.array([0, 0, 200]), np.array([180, 20, 200])

    # Renk tespiti
    color_counts = {}
    for color_name, ranges in color_masks.items():
        mask = np.zeros_like(dominant_mask)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv_frame, lower, upper))
        color_counts[color_name] = np.count_nonzero(cv2.bitwise_and(dominant_result, dominant_result, mask=mask))

    dominant_color = max(color_counts, key=color_counts.get) if color_counts else None

    # Görselleştirme
    plt.plot(h_histogram)
    plt.title('Ton Histogramı')
    plt.xlabel('Ton Değeri')
    plt.ylabel('Piksel Sayısı')
    plt.show()

    # Görüntüleri boyutlandırma
    frame_resized = cv2.resize(frame, (640, 640))
    siyah_resized = cv2.resize(siyah_masked_img, (640, 640))
    beyaz_resized = cv2.resize(beyaz_masked_img, (640, 640))
    gri_resized = cv2.resize(gri_masked_img, (640, 640))
    gri_beyaz_resized = cv2.resize(gri_beyaz_masked_img, (640, 640))
    dominant_mask_resized = cv2.resize(dominant_mask, (640, 640))
    dominant_result_resized = cv2.resize(dominant_result, (640, 640))

    # Görüntüleri gösterme
    cv2.imshow('Orijinal Image', frame_resized)
    cv2.imshow('Siyah Renk Maskesi', siyah_resized)
    cv2.imshow('Beyaz Renk Maskesi', beyaz_resized)
    cv2.imshow('Gri Renk Maskesi', gri_resized)
    cv2.imshow('Gri/Beyaz Renk Maskesi', gri_beyaz_resized)
    cv2.imshow('Dominant Mask', dominant_mask_resized)
    cv2.imshow('Dominant Renk Sonuc', dominant_result_resized)

    # Sonuç olarak baskın renk
    print(f"Baskın Renk: {dominant_color}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return dominant_color

def goster_rgb_histogramlari(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return

    b, g, r = cv2.split(img)

    # Histogramları hesapla
    r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
    g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
    b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])


    # Aynı anda histogramları görselleştir
    plt.figure(figsize=(18, 5))

    # Kırmızı Histogramı
    plt.subplot(1, 4, 1)
    plt.plot(r_hist, color='red')
    plt.title('Kırmızı (R) Histogramı')
    plt.xlabel('Kırmızı Değeri (0-255)')
    plt.ylabel('Piksel Sayısı')
    plt.grid(True)

    # Yeşil Histogramı
    plt.subplot(1, 4, 2)
    plt.plot(g_hist, color='green')
    plt.title('Yeşil (G) Histogramı')
    plt.xlabel('Yeşil Değeri (0-255)')
    plt.ylabel('Piksel Sayısı')
    plt.grid(True)

    # Mavi Histogramı
    plt.subplot(1, 4, 3)
    plt.plot(b_hist, color='blue')
    plt.title('Mavi (B) Histogramı')
    plt.xlabel('Mavi Değeri (0-255)')
    plt.ylabel('Piksel Sayısı')
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Fotoğraf')
    plt.xticks([])  # Eksen işaretlerini gizle
    plt.yticks([])  # Eksen işaretlerini gizle

    plt.tight_layout()
    plt.show()

def goster_ycrcb_histogramlari(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} dosyasından görüntü okunamadı.")
        return

    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)

    # Histogramları hesapla
    y_hist = cv2.calcHist([y], [0], None, [256], [0, 256])
    cr_hist = cv2.calcHist([cr], [0], None, [256], [0, 256])
    cb_hist = cv2.calcHist([cb], [0], None, [256], [0, 256])

    # Aynı anda histogramları görselleştir
    plt.figure(figsize=(18, 5))

    # Y (Luma) Histogramı
    plt.subplot(1, 4, 1)
    plt.plot(y_hist, color='gray')
    plt.title('Y (Luma) Histogramı')
    plt.xlabel('Parlaklık Değeri (0-255)')
    plt.ylabel('Piksel Sayısı')
    plt.grid(True)

    # Cr (Kırmızı farkı) Histogramı
    plt.subplot(1, 4, 2)
    plt.plot(cr_hist, color='red')
    plt.title('Cr (Kırmızı farkı) Histogramı')
    plt.xlabel('Cr Değeri (0-255)')
    plt.ylabel('Piksel Sayısı')
    plt.grid(True)

    # Cb (Mavi farkı) Histogramı
    plt.subplot(1, 4, 3)
    plt.plot(cb_hist, color='blue')
    plt.title('Cb (Mavi farkı) Histogramı')
    plt.xlabel('Cb Değeri (0-255)')
    plt.ylabel('Piksel Sayısı')
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Fotoğraf')
    plt.xticks([])  # Eksen işaretlerini gizle
    plt.yticks([])  # Eksen işaretlerini gizle

    plt.tight_layout()
    plt.show()

def goster_lab_histogrami(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} okunamadı.")
        return

    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab_img)
    histogramlar = [L, a, b]
    kanallar = ['L (Parlaklık)', 'a (Yeşil-Kırmızı)', 'b (Mavi-Sarı)']
    x_etiketleri = ['Değer (0-255)', 'Değer (0-255)', 'Değer (0-255)']

    plt.figure(figsize=(18, 5))
    for i in range(3):
        hist = cv2.calcHist([histogramlar[i]], [0], None, [256], [0, 256])
        plt.subplot(1, 4, i + 1)
        plt.plot(hist)
        plt.title(f'{kanallar[i]} Histogramı (Lab)')
        plt.xlabel(x_etiketleri[i])
        plt.ylabel('Piksel Sayısı')
        plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Fotoğraf')
    plt.xticks([])  # Eksen işaretlerini gizle
    plt.yticks([])  # Eksen işaretlerini gizle

    plt.tight_layout()
    plt.show()

def goster_luv_histogrami(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: {image_path} okunamadı.")
        return

    luv_img = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    L, u, v = cv2.split(luv_img)
    histogramlar = [L, u, v]
    kanallar = ['L (Parlaklık)', 'u (Yeşil-Kırmızı)', 'v (Mavi-Sarı)']
    x_etiketleri = ['Değer (0-255)', 'Değer (0-255)', 'Değer (0-255)']

    plt.figure(figsize=(18, 5))
    for i in range(3):
        hist = cv2.calcHist([histogramlar[i]], [0], None, [256], [0, 256])
        plt.subplot(1, 4, i + 1)
        plt.plot(hist)
        plt.title(f'{kanallar[i]} Histogramı (Luv)')
        plt.xlabel(x_etiketleri[i])
        plt.ylabel('Piksel Sayısı')
        plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Orijinal Fotoğraf')
    plt.xticks([])  # Eksen işaretlerini gizle
    plt.yticks([])  # Eksen işaretlerini gizle

    plt.tight_layout()
    plt.show()

def detect_black_white_gray_hsv(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Hata: Görüntü okunamadı: {image_path}")
        return None, None, None

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)

    # Siyah Tespiti: Düşük parlaklık (V) değerleri
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])  # Parlaklık (V) düşük eşikte
    black_mask = cv2.inRange(hsv_img, lower_black, upper_black)

    # Beyaz Tespiti: Yüksek parlaklık (V) ve düşük doygunluk (S) değerleri
    lower_white = np.array([0, 0, 200])  # Yüksek parlaklık
    upper_white = np.array([180, 30, 255])  # Düşük doygunluk
    white_mask = cv2.inRange(hsv_img, lower_white, upper_white)

    # Gri Tespiti: Orta parlaklık (V) ve düşük doygunluk (S) değerleri
    lower_gray = np.array([0, 0, 31])    # Orta parlaklık alt sınırı
    upper_gray = np.array([180, 50, 199])   # Orta parlaklık üst sınırı, düşük doygunluk üst sınırı
    gray_mask = cv2.inRange(hsv_img, lower_gray, upper_gray)

    return black_mask, white_mask, gray_mask

def visualize_masks(original_img_path, black_mask, white_mask, gray_mask):
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Hata: Orijinal görüntü okunamadı: {original_img_path}")
        return

    def draw_contours(image, mask, contour_color, contour_thickness):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, contour_color, contour_thickness)
        return image

    black_contour_img = draw_contours(original_img.copy(), black_mask, (255,0,0), 5)
    white_contour_img = draw_contours(original_img.copy(), white_mask, (255,0,0), 5)
    gray_contour_img = draw_contours(original_img.copy(), gray_mask, (255,0,0), 5)

    #black_detected = cv2.bitwise_and(original_img, original_img, mask=black_mask)
    #white_detected = cv2.bitwise_and(original_img, original_img, mask=white_mask)
    #gray_detected = cv2.bitwise_and(original_img, original_img, mask=gray_mask)

    cv2.imshow("Original Image", cv2.resize(original_img, (0, 0), fx=0.5, fy=0.5))
    cv2.imshow("Black Detection (HSV)", cv2.resize(black_contour_img, (0, 0), fx=0.5, fy=0.5))#black_detected
    cv2.imshow("White Detection (HSV)", cv2.resize(white_contour_img, (0, 0), fx=0.5, fy=0.5))#white_detected
    cv2.imshow("Gray Detection (HSV)", cv2.resize(gray_contour_img, (0, 0), fx=0.5, fy=0.5))#gray_detected
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def histogram_equazitation_rgb(img_path):
    try:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Hata: Görüntü okunamadı: {img_path}")
            return None

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(img_hsv)
        equized_v = cv2.equalizeHist(v)
        equized_hsv = cv2.merge((h, s, equized_v))

        equized_bgr = cv2.cvtColor(equized_hsv, cv2.COLOR_HSV2BGR)

        equized_rgb = cv2.cvtColor(equized_bgr, cv2.COLOR_BGR2RGB)

        return equized_rgb

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return None

if __name__ == "__main__":
    root = Tk()
    root.withdraw()

    print("Lütfen işlenecek görüntülerin bulunduğu klasörü seçin.")

    folder_path = filedialog.askdirectory(title="Görüntü Klasörünü Seçin")
    if not folder_path:
        print("Klasör seçilmedi. Program sonlandırılıyor.")
        exit()

    print(f"Seçilen Klasör: {folder_path}")

    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not file_list:
        print("Seçilen klasörde işlenecek hiçbir görüntü dosyası bulunamadı.")
        exit()

    output_eq_dir = os.path.join(folder_path, "Equized_Images")
    os.makedirs(output_eq_dir, exist_ok=True)

    print("\nKlasördeki tüm görüntüler üzerinde işlemler başlatılıyor...")

    for filename in file_list:
        image_path = os.path.join(folder_path, filename)
        print(f"\n--- '{filename}' işleniyor ---")

        output_eq_path = os.path.join(output_eq_dir, "Equized_" + os.path.basename(image_path))
        equized_img_rgb = histogram_equazitation_rgb(image_path)

        if equized_img_rgb is not None:
            try:
                cv2.imwrite(output_eq_path, cv2.cvtColor(equized_img_rgb, cv2.COLOR_RGB2BGR))
                print(f"  Histogram eşitlemesi uygulanmış görüntü kaydedildi: {output_eq_path}")

                color_thief = ColorThief(output_eq_path)
                baskin_renk = color_thief.get_color(quality=1)
                renk_paleti = color_thief.get_palette(color_count=6)

                print(f"  Baskın Renk (ColorThief): {baskin_renk}")
                print(f"  Renk Paleti (ColorThief): {renk_paleti}")

            except Exception as e:
                print(f"  ColorThief veya eşitlenmiş görüntüyü kaydetme sırasında bir hata oluştu: {e}")
        else:
            print(f"  Histogram eşitleme başarısız oldu veya görüntü okunamadı: {image_path}")

        black_mask, white_mask, gray_mask = detect_black_white_gray_hsv(image_path)
        if black_mask is not None:
            visualize_masks(image_path, black_mask, white_mask, gray_mask)

        dominant_color_combined = combined_color_detection(image_path)
        print(f"  '{filename}' için ana renk tespiti (combined_color_detection): {dominant_color_combined}")

    print("\nTüm işlemler tamamlandı.")