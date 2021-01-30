import numpy as np
import cv2
from sklearn import preprocessing

def get_fix_colors(name_size):
    colors = [[abs(((i + 1) % 13) * 75 - 255) % 255,
               abs(((i + 1) % 13) * 75 - 0) % 255,
               abs(((i + 1) % 13) * 75 - 255) % 255] for i in range(name_size)]
    # print(colors)
    return colors


def draw_grid(img, line_color=(255, 0, 0), thickness=1, type_=cv2.LINE_8, pxstep=200):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep


def vid_heatmap_normalize(det_nparray, min_lim=None, max_lim=None):
    """
    by theory the max value will be 1 detection per second * total (video) duration
    :param det_nparray:
    :param vid_dur:
    :param min_lim:
    :param max_lim:
    :return:
    """
    if min_lim is None:
        min_lim = 0 # default min is zero
    if max_lim is None:
        max_lim = det_nparray.max() # default max is the max value in array

    # 'relu' the values in array
    det_nparray[det_nparray < min_lim] = min_lim
    det_nparray[det_nparray > max_lim] = max_lim

    # normalize using the max and min
    det_nparray = (det_nparray - min_lim) / (max_lim - min_lim) * 255
    # det_nparray = (det_nparray - det_nparray.min()) / (det_nparray.max() - det_nparray.min()) * 255
    # det_nparray = preprocessing.MinMaxScaler().fit_transform(det_nparray) * 255

    # convert to uint8 for image viewing
    det_nparray = det_nparray.astype(np.uint8)

    return det_nparray

#
# accumulated_exposures = np.zeros((10, 10), dtype=np.float)
# cv2.fillConvexPoly(accumulated_exposures, np.array([[1,1],[1,3],[3,3],[3,1]]), (10.22))
# accumulated_exposures = vid_heatmap_normalize(accumulated_exposures, 11, 1, 9)
# print(accumulated_exposures)