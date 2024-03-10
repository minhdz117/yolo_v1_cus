import numpy as np
import cv2
from random import randint,choice
def generate_random_shape_parameters(image_height, image_width):
    shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
    size = np.random.randint(30, 50)

    if shape_type == 'circle':
        position = (
            np.random.randint(size, image_height - size),
            np.random.randint(size, image_width - size)
        )
    elif shape_type == 'rectangle':
        half_size = size // 2
        position = (
            np.random.randint(half_size, image_height - half_size),
            np.random.randint(half_size, image_width - half_size)
        )
    elif shape_type == 'triangle':
        half_size = size // 2
        position = (
            np.random.randint(half_size, image_height - half_size),
            np.random.randint(half_size, image_width - half_size)
        )

    return shape_type, position, size

def check_overlap(shapes, new_shape):
    for _, position, size in shapes:
        if (
            abs(position[0] - new_shape[1][0]) < (size + new_shape[2]) / 2 and
            abs(position[1] - new_shape[1][1]) < (size + new_shape[2]) / 2
        ):
            return True
    return False
def generate_random_color():
    #B R G

    colors = [(255,0,0),(0,255,0),(0,0,255)]
    color_name = ['Blue','Red','Green']
    i = randint(0,2)
    return color_name[i],colors[i]

def draw_random_shape(image, shape_type, position, size, shapes):
    clsColor,color = generate_random_color()
    if shape_type == 'circle':
        cv2.circle(image, position, size, color, -1)
    elif shape_type == 'rectangle':
        half_size = size // 2
        top_left = (position[0] - half_size, position[1] - half_size)
        bottom_right = (position[0] + half_size, position[1] + half_size)
        cv2.rectangle(image, top_left, bottom_right, color, -1)
    elif shape_type == 'triangle':
        pts = np.array([
            [position[0], position[1] - size // 2],
            [position[0] - size // 2, position[1] + size // 2],
            [position[0] + size // 2, position[1] + size // 2]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color)
    return clsColor

# Tạo ảnh màu xám nhạt
for i in range(7000):
    height, width = 448, 448
    gray_image = np.ones((height, width), dtype=np.uint8) * 150
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    # Tạo và vẽ các hình tròn, vuông và tam giác ngẫu nhiên
    num_shapes = randint(3,8)
    shapes = []

    for _ in range(num_shapes):
        shape_type, position, size = generate_random_shape_parameters(height, width)
        new_shape = (shape_type, position, size)

        # Kiểm tra xem hình có vượt ra khỏi biên và không chèn lên nhau không, điều chỉnh nếu cần
        while (
            position[0] - size // 2 < 0 or position[1] - size // 2 < 0 or
            position[0] + size // 2 > height or position[1] + size // 2 > width or
            check_overlap(shapes, new_shape)
        ):
            shape_type, position, size = generate_random_shape_parameters(height, width)
            new_shape = (shape_type, position, size)

        clsColor = draw_random_shape(gray_image, shape_type, position, size, shapes)
        shapes.append(new_shape)

        # In ra thông tin về vị trí và kích thước của vật thể

        print(f"{shape_type}: x={position[0]}, y={position[1]}, w={size}, h={size}")
        with open(f'Data/label/{i}.txt','a',newline='') as f:
            f.write(f"{shape_type} {clsColor} {position[0]} {position[1]} {size} {size}\n")

    # Hiển thị ảnh
    # cv2.imshow('Random Shapes on Gray Image', gray_image)
    cv2.imwrite(f'Data/image/{i}.png',gray_image)
    if i<5000:
        with open(f'Data/content/train.csv','a',newline='') as f:
            f.write(f"{i}.png,{i}.txt\n")
    elif i < 5500:
        with open(f'Data/content/val.csv','a',newline='') as f:
            f.write(f"{i}.png,{i}.txt\n")
    else:
        with open(f'Data/content/test.csv','a',newline='') as f:
            f.write(f"{i}.png,{i}.txt\n")

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
