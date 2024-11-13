from PIL import Image
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import utils

def readImage(path: str) -> np.ndarray:
    image = Image.open(path)
    image = image.convert("RGB")
    return np.array(image)


def showImage(image: np.ndarray, boxes=None) -> None:
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(image, aspect="equal")
    if boxes is not None:
        for bounding_box in boxes:
            bbox_min_x = bounding_box[0]
            bbox_min_y = bounding_box[1]
            bbox_max_x = bounding_box[2]
            bbox_max_y = bounding_box[3]

            bbox_xy = (bbox_min_x, bbox_min_y)
            bbox_width = bbox_max_x - bbox_min_x
            bbox_height = bbox_max_y - bbox_min_y
            rect = Rectangle(
                bbox_xy,
                bbox_width,
                bbox_height,
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            axs.add_patch(rect)
    pyplot.axis("off")
    pyplot.tight_layout()
    pyplot.imshow(image, cmap="gray", aspect="equal")
    pyplot.show()


def createCanvas(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


def rgbToGreyscale(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    greyscale = createCanvas(height, width)
    for h in range(height):
        for w in range(width):
            greyscale[h, w] = image[h, w, 0] * 0.2989 + \
                image[h, w, 1] * 0.5870 + image[h, w, 2] * 0.1140
    return greyscale
    # return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])


def stretchContrast(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    stretched = createCanvas(height, width)
    min_value = np.min(image)
    max_value = np.max(image)
    for h in range(height):
        for w in range(width):
            stretched[h, w] = (image[h, w] - min_value) * \
                255 / (max_value - min_value)
    return stretched


def meanFilter(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    filtered = createCanvas(height, width)

    WINDOW_SIZE = 3

    filtered = createCanvas(height, width)
    window_half = WINDOW_SIZE // 2

    for row in range(window_half, height - window_half):
        for col in range(window_half, width - window_half):
            result = 0
            for i in range(-window_half, window_half+1):
                for j in range(-window_half, window_half+1):
                    result += image[row + i, col + j]
            filtered[row, col] = abs(float(result / 25))

    return filtered


def simpleThreshold(image: np.ndarray, theta: int) -> np.ndarray:
    height, width = image.shape[:2]
    thresh = createCanvas(height, width)

    for row in range(height):
        for col in range(width):
            if image[row, col] < theta:
                thresh[row, col] = 255
            else:
                thresh[row, col] = 0

    return thresh


def adaptiveThreshold(image: np.ndarray) -> np.ndarray:
    # Get the image dimensions
    height, width = image.shape

    # Create the histogram (using numpy)
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 256))

    # Calculate N and theta0
    N = hist.cumsum()[-1]  # Total number of pixels
    theta = np.sum(np.arange(256) * hist) / N  # Initial theta

    # Calculate Nob and Nbg
    Nob = np.sum(hist[:round(theta)])
    Nbg = np.sum(hist[round(theta):])

    # Calculate uob and ubg
    uob = np.sum(np.arange(round(theta)) *
                 hist[:round(theta)]) / Nob if Nob > 0 else 0
    ubg = np.sum(np.arange(round(theta), 256) *
                 hist[round(theta):]) / Nbg if Nbg > 0 else 0

    # Find theta j+1
    thetaNext = (uob + ubg) / 2

    # Iteration loop until convergence
    while abs(theta - thetaNext) > 1e-5:  # Set a small tolerance for convergence
        theta = thetaNext

        Nob = np.sum(hist[:round(theta)])
        Nbg = np.sum(hist[round(theta):])

        uob = np.sum(np.arange(round(theta)) *
                     hist[:round(theta)]) / Nob if Nob > 0 else 0
        ubg = np.sum(np.arange(round(theta), 256) *
                     hist[round(theta):]) / Nbg if Nbg > 0 else 0

        thetaNext = (uob + ubg) / 2

    # Apply the threshold using the calculated theta
    thresh = simpleThreshold(image, theta)

    return thresh


def dilation(image: np.ndarray, kernel: list) -> np.ndarray:
    height, width = image.shape
    kernel = np.array(kernel)
    kernel_half = len(kernel) // 2

    # Create a padded version of the image to handle borders
    padded_image = np.pad(image, kernel_half,
                          mode='constant', constant_values=0)
    result = np.zeros_like(image)

    # Apply the kernel on each pixel
    for i in range(-kernel_half, kernel_half + 1):
        for j in range(-kernel_half, kernel_half + 1):
            if kernel[i + kernel_half, j + kernel_half] == 1:
                # Shift the padded image and apply maximum where kernel is 1
                result = np.maximum(
                    result, padded_image[kernel_half + i:kernel_half + i + height, kernel_half + j:kernel_half + j + width])

    return result


def erosion(image: np.ndarray, kernel: list) -> np.ndarray:
    height, width = image.shape
    kernel = np.array(kernel)
    kernel_half = len(kernel) // 2

    # Create a padded version of the image to handle borders
    padded_image = np.pad(image, kernel_half,
                          mode='constant', constant_values=255)
    result = np.full_like(image, 255)

    # Apply the kernel on each pixel
    for i in range(-kernel_half, kernel_half + 1):
        for j in range(-kernel_half, kernel_half + 1):
            if kernel[i + kernel_half, j + kernel_half] == 1:
                # Shift the padded image and apply minimum where kernel is 1
                result = np.minimum(
                    result, padded_image[kernel_half + i:kernel_half + i + height, kernel_half + j:kernel_half + j + width])

    return result


def connectedComponents(image: np.ndarray) -> list:
    components = []
    height, width = image.shape[:2]
    visited = createCanvas(height, width)
    queue = []

    for row in range(height):
        for col in range(width):
            if not visited[row, col] and image[row, col] != 0:
                queue.append((row, col))
                visited[row, col] = 1
                min_x, min_y, max_x, max_y = col, row, 0, 0

                # BFS, enqueue if pixel not 0 and not visited
                while queue:
                    row, col = queue.pop(0)
                    min_x = min(min_x, col)
                    min_y = min(min_y, row)
                    max_x = max(max_x, col)
                    max_y = max(max_y, row)

                    # Left
                    try:
                        if visited[row, col-1] == 0 and image[row, col-1] != 0:
                            queue.append((row, col-1))
                            visited[row, col-1] = 1
                    except IndexError:
                        pass

                    # Right
                    try:
                        if visited[row, col+1] == 0 and image[row, col+1] != 0:
                            queue.append((row, col+1))
                            visited[row, col+1] = 1
                    except IndexError:
                        pass

                    # Up
                    try:
                        if visited[row-1, col] == 0 and image[row-1, col] != 0:
                            queue.append((row-1, col))
                            visited[row-1, col] = 1
                    except IndexError:
                        pass

                    # Down
                    try:
                        if visited[row+1, col] == 0 and image[row+1, col] != 0:
                            queue.append((row+1, col))
                            visited[row+1, col] = 1
                    except IndexError:
                        pass

                components.append((min_x, min_y, max_x, max_y))

                diameter_x = max_x-min_x
                diameter_y = max_y-min_y
                # Minimum size threshold to filter out little noise
                threshold_size = 8
                if diameter_x < threshold_size and diameter_y < threshold_size:
                    pass
                else:
                    components.append((min_x, min_y, max_x, max_y))

    return components


def getComponents(image: np.ndarray, boxes: list) -> list:
    components = []
    for box in boxes:
        min_x, min_y, max_x, max_y = box
        component = image[min_y:max_y, min_x:max_x]
        components.append(component)
    return components

def shrinkImage(image:np.ndarray, factor:int) -> np.ndarray:
    height, width = image.shape[:2]
    new_height = height // factor
    new_width = width // factor
    new_image = createCanvas(new_height, new_width)

    for h in range(new_height):
        for w in range(new_width):
            window = image[h*factor:(h+1)*factor, w*factor:(w+1)*factor]
            if np.any(window == 255):
                new_image[h, w] = 255
            else:
                new_image[h, w] = 0

    return new_image

kernel = [
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0]
]

image = readImage("test.jpg")
image = rgbToGreyscale(image)
image = stretchContrast(image)
image = meanFilter(image)
image = adaptiveThreshold(image)
image = dilation(image, kernel)
image = shrinkImage(image, 2)
image = erosion(image, kernel)
boxes = connectedComponents(image)

boxes.sort(key=lambda x: x[0])  # Order boxes by min_x

components = getComponents(image, boxes)
templates = utils.TEMPLATES
showImage(image, boxes)
# for component in components:
#     showImage(component)

# TODO: enlarge the template to match the size of the component, if the template completely captures the component, then it is a match