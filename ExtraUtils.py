import cv2

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