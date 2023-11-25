from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

# 이미지를 불러옵니다.
image = Image.open('dataset/celeba/img_align_celeba/000001.jpg')

# 색상 유사성 맵을 계산합니다.
similarity_map = calculate_color_similarity(image, 3)

# 원본 이미지와 색상 유사성 맵을 시각화합니다.
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(similarity_map, cmap='hot')
plt.title('Color Similarity Map')
plt.axis('off')

plt.show()