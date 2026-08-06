import cv2
import numpy as np
import requests

def load_cv2_image(image_source: str) -> np.ndarray:
    if image_source.startswith("http://") or image_source.startswith("https://"):
        response = requests.get(image_source, timeout=10)
        response.raise_for_status()
        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(image_source)
        
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_source}")
    return img

def check_boundary_cutoff(gray_img: np.ndarray, margin: int = 5) -> bool:
    """
    배터리 객체가 이미지 경계(가장자리)에 닿아 잘려 나갔는지 확인합니다.
    """
    # 가우시안 블러 후 오츠(Otsu) 이진화로 객체와 배경 분리
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 외곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
        
    # 가장 큰 외곽선을 배터리 본체로 간주
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    height, width = gray_img.shape
    
    # 바운딩 박스가 상하좌우 경계선(margin 픽셀 이내)에 닿았는지 판별
    is_cut_off = (x <= margin) or (y <= margin) or (x + w >= width - margin) or (y + h >= height - margin)
    return is_cut_off

def check_physical_quality(
    image_source: str, 
    image_type: str = "RGB",
    blur_threshold: float = 18.0, 
    under_threshold: float = 70.0, 
    over_threshold: float = 240.0
) -> tuple[str | None, str]:
    try:
        img = load_cv2_image(image_source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_type == "RGB":
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < blur_threshold:
                return "rgb_focus_failure", f"선명도 점수({laplacian_var:.1f}) 미달로 초점 불량."
            
            h, w = img.shape[:2]
            roi = gray[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
            mean_brightness = float(np.mean(roi))

            if mean_brightness < under_threshold:
                return "rgb_underexposure", f"평균 밝기({mean_brightness:.1f}) 미달로 노출 부족."
            elif mean_brightness > over_threshold:
                return "rgb_overexposure", f"평균 밝기({mean_brightness:.1f}) 초과로 노출 과다."

        return None, "OpenCV 사전 검사 통과"
    except Exception as e:
        return None, f"OpenCV 검사 오류: {str(e)}"