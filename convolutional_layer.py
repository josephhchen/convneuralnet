import numpy as np

def conv2d(image: np.ndarray, kernel: np.ndarray, stride: int=1, padding: int=0) -> np.ndarray:

    if padding > 0:
        image = np.pad(image,( (padding, padding), (padding, padding) ), mode='constant', constant_values=0)

    kernel_height, kernel_width = kernel.shape # current kernal that is sliding over image
    image_height, image_width = image.shape

    # determines number of valid positions where kernel can be placed on the input image
    output_height = (image_height - kernel_height + 2 * padding) // stride + 1 
    output_width = (image_width - kernel_width + 2 * padding) // stride + 1

    output_matrix = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
               
            start_y = y * stride
            end_y = start_y + kernel_height

            start_x = x * stride
            end_x = start_x + kernel_width

            region = image[start_y:end_y, start_x:end_x]

            output_matrix[y, x] = np.sum(region * kernel)

    return output_matrix

input_image = np.array([
    [7,2,3,3,8],
    [4,5,3,8,4],
    [3,3,2,8,4],
    [2,8,7,2,7],
    [5,4,4,5,4]
])

test_kernel = np.array([
    [1,0,-1],
    [1,0,-1],
    [1,0,-1]
])

output = conv2d(input_image, test_kernel, stride=1, padding=0)

print(f"input image:\n {input_image}")
print(f"kernel: {test_kernel}")
print(f"output: {output}")




