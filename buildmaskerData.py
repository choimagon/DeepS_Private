import os
import cv2
import time
from masker import FaceMasker

# 처리할 입력 폴더 목록
input_base_folder = 'data/image'
target_folders = ['00000', '01000', '02000', '03000']  # 필요 시 추가

# 출력 폴더 (모든 결과를 이곳에 저장)
output_folder_masked = 'all_half_masked'
output_folder_mask = 'all_half_mask_only'
os.makedirs(output_folder_masked, exist_ok=True)
os.makedirs(output_folder_mask, exist_ok=True)

# FaceMasker 인스턴스 생성
masker = FaceMasker()

for folder in target_folders:
    input_folder = os.path.join(input_base_folder, folder)
    print(f"[시작] {folder} 폴더 처리 중...")

    # 이미지 파일 리스트
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    s = time.time()
    for filename in image_files:
        input_path = os.path.join(input_folder, filename)

        # 출력 파일 경로 (폴더명 prefix로 구분)
        output_filename = f"{folder}_{filename}"
        output_path_masked = os.path.join(output_folder_masked, output_filename)
        output_path_mask = os.path.join(output_folder_mask, output_filename)

        image = cv2.imread(input_path)
        if image is None:
            print(f"[경고] 이미지 로드 실패: {input_path}")
            continue

        # 마스크 처리
        masked_image, mask_only = masker.mask_face(image)

        # 저장
        cv2.imwrite(output_path_masked, masked_image)
        cv2.imwrite(output_path_mask, mask_only)
        print(f"[처리 완료] {output_filename}")

    print(f"[완료] {folder} 폴더 마스킹 완료. {(time.time() - s):.1f}초\n")

