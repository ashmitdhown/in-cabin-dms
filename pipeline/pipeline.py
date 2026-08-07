from ultralytics import YOLO
import cv2
import numpy as np

# =====================================================
# LOAD MODELS
# =====================================================
gaze_model = YOLO(r"D:\In Cabin Ai Monitoring System - Ashmit & Aarav\In Cabin Monitoring System (Combined)\Phase 1\Dataset\runs\detect\final-train\weights\best.pt") 
drowsy_model = YOLO(r"D:\In Cabin Ai Monitoring System - Ashmit & Aarav\In Cabin Monitoring System (Combined)\Phase 3\runs\detect\train\weights\best.pt")
object_model = YOLO(r"D:\In Cabin Ai Monitoring System - Ashmit & Aarav\In Cabin Monitoring System (Combined)\Phase 2\Dataset V8\runs\detect\train2\weights\best.pt") 

# ===================================================== # IMAGE # ===================================================== 

IMAGE_PATH = r"D:\In Cabin Ai Monitoring System - Ashmit & Aarav\In Cabin Monitoring System (Combined)\Phase 3\image2.png"
img = cv2.imread(IMAGE_PATH)

annotated = img.copy()

# =====================================================
# STATUS TRACKING (BEST CONFIDENCE ONLY)
# =====================================================
gaze_status = ("No Detection", 0)
drowsy_status = ("No Detection", 0)
object_status = ("No Detection", 0)

# =====================================================
# DRAW FUNCTION
# =====================================================
def draw_results(results, image, color, prefix=""):
    global gaze_status, drowsy_status, object_status

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = r.names[cls]

            label = f"{prefix}{class_name} {conf:.2f}"

            # =========================
            # UPDATE BEST STATUS
            # =========================
            if prefix == "Gaze-":
                if conf > gaze_status[1]:
                    gaze_status = (f"{class_name} ({conf:.2f})", conf)

            elif prefix == "Drowsy-":
                if conf > drowsy_status[1]:
                    drowsy_status = (f"{class_name} ({conf:.2f})", conf)

            elif prefix == "Object-":
                if conf > object_status[1]:
                    object_status = (f"{class_name} ({conf:.2f})", conf)

            # =========================
            # DRAW BOX
            # =========================
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(image, (x1, y1 - 25), (x1 + w + 6, y1), color, -1)

            cv2.putText(
                image,
                label,
                (x1 + 3, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

# =====================================================
# RUN MODELS
# =====================================================
draw_results(gaze_model(img), annotated, (0,255,0), "Gaze-")
draw_results(drowsy_model(img), annotated, (255,0,0), "Drowsy-")
draw_results(object_model(img), annotated, (0,0,255), "Object-")

# =====================================================
# RESIZE FOR BETTER VIEWING
# =====================================================
h, w = annotated.shape[:2]
scale = 900 / w
new_size = (int(w * scale), int(h * scale))
display_img = cv2.resize(annotated, new_size)

# =====================================================
# MODERN STATUS PANEL (TRANSPARENT STYLE)
# =====================================================
overlay = display_img.copy()

cv2.rectangle(overlay, (10, 10), (350, 110), (30, 30, 30), -1)

alpha = 0.6
cv2.addWeighted(overlay, alpha, display_img, 1 - alpha, 0, display_img)

cv2.putText(display_img, f"Gaze: {gaze_status[0]}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

cv2.putText(display_img, f"Drowsy: {drowsy_status[0]}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

cv2.putText(display_img, f"Object: {object_status[0]}", (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# =====================================================
# DISPLAY WINDOW (RESIZABLE)
# =====================================================
cv2.namedWindow("Combined Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Combined Detection", display_img)

cv2.waitKey(0)
cv2.destroyAllWindows()