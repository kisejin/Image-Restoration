 import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import logging


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def read_img(path, enable=None):
    img = plt.imread(path)
    return (img, rgb2gray(img))[enable=='gray']

def show_img(img: np.array, ax, option = None):
    ax.imshow(img, cmap=option)
    ax.axis('off')

def matrix_degraded(r,m,l: int, option = 'v'):
    nrow = r
    ncol = m
    if option in ('v', 'h'):
      if option == 'h':
            n = m + l - 1
            nrow, ncol = m, n
      else:
            n = r + l - 1
            ncol = n
      A = np.zeros((nrow,ncol), dtype = np.float16)

      for i in range(nrow):
          A[i,i:i+l] = 1/l
      return A

 def restore_img(img,l=30, option = 'v'):
    A = matrix_degraded(img.shape[0], img.shape[1],
                l=30, option = option)
    U, D, VT = svd(A)
    if A.shape[0] > A.shape[1]:
      D_1 = np.eye(A.shape[0], A.shape[1])@np.diag(1/D)
    else:
      D_1 = np.diag(1/D)@np.eye(A.shape[0], A.shape[1])
    A_1 = VT.T@D_1.T@U.T

    return img@A_1.T if option == 'h' else A_1@img

def restore_rgb(img: np.array, l =30, option = 'v'):
    if option in ('v', 'h'):
        r = restore_img(img[:,:,0],l=l, option = option)
        r -= np.mean(r)

        g = restore_img(img[:,:,1],l=l, option = option)
        g -= np.mean(g)

        b = restore_img(img[:,:,2],l=l, option = option)
        b -= np.mean(b)
        return np.stack((r,g,b),axis=2)

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input file path')

    return parser.parse_args()



if __name__ == '__main__':
    args = init_args()
    path = args.input
    fig, ax = plt.subplots(1,2)
    # Doc anh tu file va lay mau trang den
    img_ver_gray = read_img(path + '/parrot_vertical.jpg','gray')
    img_hor_gray = read_img(paht + '/car_horizontal.jpg','gray')

    # Phuc hoi anh theo phuong ngang va dung
    img_hor_restore = restore_img(img_hor,l=30, option = 'h')
    img_ver_restore = restore_img(img_ver,l=30, option = 'v')

    # Doc anh tu file lay mau RGB
    img_ver_rgb = read_img(path + '/parrot_vertical.jpg')
    img_hor_rgb = read_img(path + '/car_horizontal.jpg')

    # Phuc hoi anh mau theo phuong ngang va dung
    img_ver_rgb_res = restore_rgb(img_ver_rgb, l =30, option = 'v')
    img_hor_rgb_res = restore_rgb(img_hor_rgb, l =30, option = 'h')

    img_arr = np.array([
        img_ver_gray, img_hor_gray,
        img_ver_restore ,img_hor_restore,
        img_ver_rgb, img_hor_rgb,
        img_ver_rgb_res, img_hor_rgb_res
    ])

    fig, ax = plt.subplots(4,2, figsize = (10,10))
    ax = ax.ravel()
    for i in range(0,8,2):
        option = None
        if i < 4:
          option = 'gray'
        if i+1 < 8:
          show_img(img_arr[i],ax[i], option)
          ax[i].set_title('Hinh phuong thang dung')
          show_img(img_arr[i+1],ax[i+1],option)
          ax[i+1].set_title('Hinh phuong ngang')

