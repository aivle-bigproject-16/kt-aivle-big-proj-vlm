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
    over_threshold: float = 190.0,
    noise_threshold: float = 4.0,
    margin_threshold: float = 5.0,
    cupping_diff_threshold: float = 30.0,
    directional_ratio_threshold: float = 4.5
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
            
        elif image_type == "CT":
            h, w = gray.shape[:2]
            roi = gray[int(h*0.3):int(h*0.7), int(w*0.4):int(w*0.6)]
            blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            noise_residual = cv2.absdiff(roi, blurred)
            noise_level = float(np.mean(noise_residual))

            if noise_level > noise_threshold:
                return "ct_low_signal_noise", f"평균 노이즈 수치({noise_level:.1f}) 초과로 화질 저하."

            col_mean = np.mean(gray, axis=0)
            max_mean = np.max(col_mean)
            
            # 최고 평균 밝기의 50%를 넘는 X 좌표 추출 (배터리 영역)
            battery_cols = np.where(col_mean > max_mean * 0.5)[0]
            if len(battery_cols) == 0:
                return "error", "배터리 객체를 찾을 수 없습니다."

            bx = battery_cols[0]
            bw = battery_cols[-1] - bx
            by, bh = 0, h  # Y축 전체 사용

            # =================================================================
            # [검사 1] 정렬 실패 (ct_cell_alignment_failure)
            # =================================================================
            margin = 5
            if bx <= margin or (bx + bw) >= (w - margin):
                return "ct_cell_alignment_failure", f"배터리 좌우가 화면 경계에 닿음. (X 시작: {bx}, 끝: {bx+bw})"

            battery_roi = gray[by:by+bh, bx:bx+bw]
            roi_h, roi_w = battery_roi.shape

            # =================================================================
            # [검사 3] 빔 하드닝 (ct_beam_hardening_metal_streak)
            # =================================================================
            left_edge_roi = battery_roi[:, int(roi_w*0.05):int(roi_w*0.15)]
            right_edge_roi = battery_roi[:, int(roi_w*0.85):int(roi_w*0.95)]

            mean_center = float(np.mean(center_roi))
            mean_edge = float((np.mean(left_edge_roi) + np.mean(right_edge_roi)) / 2)
            cupping_diff = mean_edge - mean_center

            if cupping_diff > 35.0:  # 양 끝단이 중심부보다 압도적으로 밝음
                return "ct_beam_hardening_metal_streak", f"외곽과 중심부의 밝기 편차({cupping_diff:.1f})로 Cupping 감지됨."

            # =================================================================
            # [검사 4] 투영 부족/알리어싱 (ct_insufficient_projection_sampling)
            # =================================================================
            # 배터리 바깥 배경 픽셀만 모아서 옅은 원형 띠(Halo)나 줄무늬가 있는지 검사
            bg_left = gray[:, :max(bx, 1)]
            bg_right = gray[:, min(bx+bw, w-1):]
            
            bg_pixels = np.concatenate((bg_left.flatten(), bg_right.flatten()))
            bg_stddev = float(np.std(bg_pixels)) if len(bg_pixels) > 0 else 0.0

            if bg_stddev > 10.0: # 배경이 단일 흑색이 아니면 표준편차가 치솟음
                return "ct_insufficient_projection_sampling", f"배경 영역 표준편차({bg_stddev:.1f})가 높아 Halo/Streak 감지됨."

            # =================================================================
            # [검사 5] 모션 블러 (ct_acquisition_motion)
            # =================================================================
            sobel_x = cv2.Sobel(battery_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(battery_roi, cv2.CV_64F, 0, 1, ksize=3)
            
            var_x = np.var(sobel_x)
            var_y = np.var(sobel_y)

            var_min = max(min(var_x, var_y), 1e-6)
            var_max = max(var_x, var_y)
            directional_ratio = var_max / var_min

            if directional_ratio > 3.0: 
                return "ct_acquisition_motion", f"X/Y 방향 엣지 편차 비율({directional_ratio:.1f})이 커 흔들림 감지됨."

            return "ct_NONE", "모든 수치 검사를 무사히 통과한 정상 CT 영상입니다."

        return None, "OpenCV 사전 검사 통과"
    except Exception as e:
        return None, f"OpenCV 검사 오류: {str(e)}"