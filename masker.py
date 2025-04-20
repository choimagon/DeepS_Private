import cv2
import numpy as np
import random

class FaceMasker:
    def __init__(self, scaleFactor=1.1, minNeighbors=5, angle_range=(-30, 30), size_ratio=(1/3.5, 1/2)):
        """
        얼굴 검출 및 마스킹 클래스
        - scaleFactor: Haar cascade 스케일 팩터
        - minNeighbors: 얼굴 검출 민감도
        - angle_range: 마스킹 회전 각도 범위 (도 단위)
        - size_ratio: 마스킹 박스 크기 비율 (min, max)
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.angle_range = angle_range
        self.size_ratio = size_ratio

    def mask_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=self.scaleFactor, minNeighbors=self.minNeighbors)
        masked_image = image.copy()
        mask_only = np.zeros_like(image)  # 마스크만 그릴 빈 이미지

        if len(faces) == 0:
            # 얼굴이 없을 경우 전체 이미지 기준으로 랜덤 박스 생성
            h_img, w_img = image.shape[:2]
            rw = int(random.uniform(w_img * self.size_ratio[0], w_img * self.size_ratio[1]))
            rh = int(random.uniform(h_img * self.size_ratio[0], h_img * self.size_ratio[1]))

            rx = random.randint(0, w_img - rw)
            ry = random.randint(0, h_img - rh)
            angle = random.uniform(*self.angle_range)

            center = (rx + rw // 2, ry + rh // 2)
            size = (rw, rh)
            rot_rect = (center, size, angle)
            box = cv2.boxPoints(rot_rect)
            box = np.intp(box)

            cv2.fillConvexPoly(masked_image, box, (0, 0, 0))
            cv2.fillConvexPoly(mask_only, box, (255, 255, 255))  # 마스크는 흰색 영역
        else:
            for (x, y, w, h) in faces:
                rw = int(random.uniform(w * self.size_ratio[0], w * self.size_ratio[1]))
                rh = int(random.uniform(h * self.size_ratio[0], h * self.size_ratio[1]))

                rx = random.randint(x, x + w - rw)
                ry = random.randint(y, y + h - rh)

                angle = random.uniform(*self.angle_range)
                center = (rx + rw // 2, ry + rh // 2)
                size = (rw, rh)

                rot_rect = (center, size, angle)
                box = cv2.boxPoints(rot_rect)
                box = np.intp(box)

                cv2.fillConvexPoly(masked_image, box, (0, 0, 0))
                cv2.fillConvexPoly(mask_only, box, (255, 255, 255))

        return masked_image, mask_only



