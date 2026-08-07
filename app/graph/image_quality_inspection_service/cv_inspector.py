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


def check_physical_quality(
    image_source: str, 
    image_type: str = "RGB",
    blur_threshold: float = 18.0, 
    under_threshold: float = 70.0, 
    over_threshold: float = 190.0
) -> tuple[str | None, str]:
    try:
        img = load_cv2_image(image_source)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if image_type == "RGB":
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < blur_threshold:
                return "rgb_focus_failure", f"선명도 점수({laplacian_var:.1f}) 미달로 초점 불량."
            
            h, w = img.shape[:2]
            roi = gray[int(h*0.25):int(h*0.75), int(w*0.40):int(w*0.60)]
            mean_brightness = float(np.mean(roi))

            if mean_brightness < under_threshold:
                return "rgb_underexposure", f"평균 밝기({mean_brightness:.1f}) 미달로 노출 부족."
            elif mean_brightness > over_threshold:
                return "rgb_overexposure", f"평균 밝기({mean_brightness:.1f}) 초과로 노출 과다."

        return None, "OpenCV 사전 검사 통과"
    except Exception as e:
        return None, f"OpenCV 검사 오류: {str(e)}"