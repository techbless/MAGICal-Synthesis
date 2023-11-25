from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_color_similarity(image, n):
    # 이미지를 NumPy 배열로 변환합니다.
    image_array = np.array(image)

    # 이미지의 높이와 너비를 가져옵니다.
    height, width = image_array.shape[:2]

    # 유사성 맵을 초기화합니다.
    similarity_map = np.zeros((height, width))

    # 이미지의 각 픽셀을 순회합니다.
    for y in range(height):
        for x in range(width):
            # 주변 n x n 픽셀을 추출합니다.
            neighborhood = image_array[max(y - n//2, 0):min(y + n//2 + 1, height),
                                       max(x - n//2, 0):min(x + n//2 + 1, width)]

            # 현재 픽셀의 색상과 주변 픽셀의 색상 차이를 계산합니다.
            current_pixel = image_array[y, x]
            color_diff = np.sqrt(np.sum((neighborhood - current_pixel)**2, axis=-1))

            # 색상 유사성을 계산합니다.
            similarity_map[y, x] = np.mean(color_diff)

    return similarity_map

# 이미지 폴더 경로를 지정합니다.
folder_path = 'dataset/celeba/img_align_celeba'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

# 이미지 갯수 제한 설정 (예: 10개)
n_images = 20

# 유사성 맵들을 저장할 리스트
similarity_maps = []

# 폴더 내의 이미지를 불러와서 유사성 맵을 계산합니다.
for i, file in enumerate(image_files):
    if i >= n_images:
        break
    image_path = os.path.join(folder_path, file)
    image = Image.open(image_path)
    similarity_map = calculate_color_similarity(image, 2)
    similarity_maps.append(similarity_map)

# 유사성 맵들의 평균을 계산합니다.
# ... [이전 코드와 동일한 부분] ...

# 유사성 맵들의 평균을 계산합니다.
average_similarity_map = np.mean(similarity_maps, axis=0)

# 최소값과 최대값을 사용하여 0-1 사이로 스케일링합니다.
min_val = np.min(average_similarity_map)
max_val = np.max(average_similarity_map)
scaled_similarity_map = (average_similarity_map - min_val) / (max_val - min_val)

# 1에서 스케일링된 값을 빼서 새로운 맵을 생성합니다.
inverted_similarity_map = 1 - scaled_similarity_map

# 수정된 유사성 맵을 시각화합니다.
plt.imshow(inverted_similarity_map, cmap='cubehelix')
plt.colorbar()  # 컬러바를 추가하여 값의 범위를 표시합니다.
plt.title('Adjacent Pixel Color Similarity Map')
plt.axis('off')
plt.show()
