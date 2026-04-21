import cv2, os
from utils import log

def run_extractor(image_path):
    """提取地标轮廓（OpenCV 边缘检测）"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    out_path = os.path.join("outputs", "outline.png")
    cv2.imwrite(out_path, edges)
    log("OutlineExtractor", f"Saved outline: {out_path}")
    return out_path
