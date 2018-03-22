import time
import numpy as np
import cv2
import os


def show_image(name, image):
    max_width = 1980

    cv2.namedWindow(name)
    cv2.moveWindow(name, show_image.x, show_image.y)
    show_image.x += show_image.width + 82
    if show_image.x > max_width:
        show_image.x = 0
        show_image.y += show_image.height + 81
    cv2.imshow(name, image)

    path = os.path.split(input_file)
    output_dir = path[0] + os.sep + 'out' + os.sep + os.path.splitext(path[1])[0] + os.sep
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(output_dir + str(show_image.step) + '. ' + name + '.png', image)
    show_image.step += 1


show_image.x = 0
show_image.y = 200
show_image.step = 1
show_image.width = 240
show_image.height = 240


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        diff = (time2 - time1) * 1000.0
        timing.total += diff
        print('%s: %0.3f ms' % (f.__name__, diff))
        return ret
    return wrap


timing.total = 0

@timing
def resize(image):
    width = 320
    scale = width / image.shape[1]
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


@timing
def blur(image):
    kernel_size = 3
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


@timing
def edges(image):
    threshold1 = 150
    threshold2 = 200
    return cv2.Canny(image, threshold1, threshold2)


@timing
def dilate(image):
    kernel_size = 4
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


@timing
def threshold(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image_gray, 90, 255, 0)
    return thresh


@timing
def get_contours(image):
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


@timing
def filter_contours_by_area(contours):
    min_area = 88
    max_area = 5500
    ret = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            ret.append(c)
    return ret


@timing
def approximate_contours(contours):
    ret = []
    for c in contours:
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        ret.append(approx)
    return ret


@timing
def filter_non_convex_contours(contours):
    ret = []
    for c in contours:
        if cv2.isContourConvex(c):
            ret.append(c)
    return ret

@timing
def find_face_pieces(contours):
    if len(contours) == 9:
        return contours

    # TODO: improve
    threshold = 0.155
    matched = []
    found = False
    for c1 in contours:
        matched = []
        for i, c2 in enumerate(contours):
            m = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0)
            if m < threshold:
                matched.append(i)
        if len(matched) == 9:
            found = True
            break

    return [contours[x] for x in matched] if found else None


@timing
def find_face_corners(contours):
    points = np.concatenate(contours)[:, 0, :]
    # TODO: improve
    bottom_left = tuple(points[points[:, 0].argmin()])
    top_left = tuple(points[points[:, 1].argmin()])
    top_right = tuple(points[points[:, 0].argmax()])
    bottom_right = tuple(points[points[:, 1].argmax()])

    return [bottom_left, top_left, top_right, bottom_right]


@timing
def projective_transform(image, corners):
    width = 120
    height = 120

    points1 = np.float32(corners)
    points2 = np.float32([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]])

    M = cv2.getPerspectiveTransform(points1, points2)
    return cv2.warpPerspective(image, M, (width, height))


@timing
def get_colors(image):
    height, width, _ = image.shape
    stride = width / 3
    offset = (width / 3) / 2

    centers = [[offset, offset],             [offset + stride, offset],              [offset + stride * 2, offset],
               [offset, offset + stride],    [offset + stride, offset + stride],     [offset + stride * 2, offset + stride],
               [offset, offset + stride * 2], [offset + stride, offset + stride * 2], [offset + stride * 2, offset + stride * 2]]

    calibrated_colors = {
        (0, 0, 255):      [(314, 360), (0, 12)],  # red
        (0, 255, 0):      [[106, 181]],  # green
        (0, 165, 255):    [[13, 25]],    # orange
        (255, 0, 0):      [[208, 215]],  # blue
        (255, 255, 255):  [[216, 230]],  # white
        (0, 255, 255):    [[44, 66]]     # yellow
    }

    error_color = (255, 0, 255)
    colors = [error_color] * 9
    for i, c in enumerate(centers):
        c = tuple(map(int, c))
        roi = image[c[0]-4:c[0]+4, c[1]-4:c[1]+4]
        pixel = cv2.mean(roi)
        float_pixel = np.float32([[[x / 255.0 for x in pixel]]])
        hsv = cv2.cvtColor(float_pixel, cv2.COLOR_BGR2HSV)
        h = hsv[0, 0, 0]
        print('h:    ', h)
        detected = error_color
        for k, v in calibrated_colors.items():
            found = False
            for range in v:
                if range[0] <= h <= range[1]:
                    found = True
            if found:
                detected = k
                break
        colors[i] = detected

    return colors


def get_colors_image(colors):
    width, height, = (120, 120)
    stride = width // 3

    image = np.ones((width, height, 3), np.uint8) * 255

    x = 0
    y = 0
    for i, c in enumerate(colors):
        cv2.rectangle(image, (x, y), (x + stride, y + stride), c, -1)
        cv2.rectangle(image, (x, y), (x + stride, y + stride), (0, 0, 0), 2)
        y += stride
        if i == 2 or i == 5:
            y = 0
            x += stride
    return image


def slic(image):
    num_superpixels = 400
    num_levels = 4
    prior = 2
    num_histogram_bins = 5
    num_iterations = 4

    height, width, channels = image.shape
    seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels, num_superpixels, num_levels, prior, num_histogram_bins)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    seeds.iterate(hsv_image, num_iterations)

    color_image = np.zeros((height, width, 3), np.uint8)
    color_image[:] = (0, 0, 255)

    labels = seeds.getLabels()
    num_label_bits = 2
    labels &= (1 << num_label_bits) - 1
    labels *= 1 << (16 - num_label_bits)

    mask = seeds.getLabelContourMask(False)

    mask_inv = cv2.bitwise_not(mask)
    result_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_image, color_image, mask=mask)
    result = cv2.add(result_bg, result_fg)

    return result


def main():
    frame = cv2.imread(input_file)
    # frame = cv2.flip(frame, 0)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = resize(frame)
    show_image('Frame', frame)

    original_frame = frame

    blurred_frame = blur(frame)
    show_image('Blurred', blurred_frame)

    edges_frame = edges(blurred_frame)
    show_image('Edges', edges_frame)

    # threshold = threshold(frame)
    # show_image('Threshold', threshold)

    dilated_frame = dilate(edges_frame)
    show_image('Dilated', dilated_frame)

    contours = get_contours(dilated_frame)
    contours_frame = frame.copy()
    cv2.drawContours(contours_frame, contours, -1, (0, 255, 0), 3)
    show_image('Contours', contours_frame)

    contours = approximate_contours(contours)
    contours_frame = frame.copy()
    cv2.drawContours(contours_frame, contours, -1, (0, 255, 0), 3)
    show_image('Approximated contours', contours_frame)

    contours = filter_contours_by_area(contours)
    contours_frame = frame.copy()
    cv2.drawContours(contours_frame, contours, -1, (0, 255, 0), 3)
    show_image('Filtered by area contours', contours_frame)

    contours = filter_non_convex_contours(contours)
    contours_frame = frame.copy()
    cv2.drawContours(contours_frame, contours, -1, (0, 255, 0), 3)
    show_image('Convex contours', contours_frame)

    contours = find_face_pieces(contours)
    if not contours:
        return

    contours_frame = frame.copy()
    cv2.drawContours(contours_frame, contours, -1, (0, 255, 0), 3)
    show_image('Face pieces', contours_frame)

    face_contours = contours

    corners = find_face_corners(contours)
    corners_frame = frame.copy()
    for c in corners:
        cv2.circle(corners_frame, c, 5, (0, 255, 0), 3)
    show_image('Face corners', corners_frame)

    face = projective_transform(blurred_frame, corners)
    show_image('Face', face)

    # slic_face = slic(face)
    # show_image('SLIC', slic_face)

    colors = get_colors(face)
    colors_image = get_colors_image(colors)
    show_image('Colors', colors_image)


if __name__ == "__main__":
    input_file = 'tests/cluttered.jpg'
    main()
    print('total: %0.3f ms' % timing.total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
