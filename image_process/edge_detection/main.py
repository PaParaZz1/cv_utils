import cv2
import numpy as np
import edge as E
import noise as N
import motion_blur as M
import sharpen as S


def img_process_func(img):
    img_func_list = [lambda x:x, N.noise_gauss, N.noise_gauss_possion, N.noise_salt_pepper, M.motion_blur_func_linear, M.motion_blur_func_nonlinear, M.gauss_blur]
    img_list = [F(img) for F in img_func_list]

    origin_func_list = [lambda x:x]
    edge_func_list = [E.gradient_sobel, E.gradient_laplace, E.gradient_canny]
    sharpen_func_list = [S.sharpen_naive, S.unsharpened_mask]
    process_func_list = origin_func_list + edge_func_list + sharpen_func_list

    def single_img_func(img):
        processed_list = [F(img) for F in process_func_list]
        return processed_list

    handle = map(single_img_func, img_list)
    result = list(handle)
    return result


def save_img_list(img_list):
    imgs = []
    for item in img_list:
        img_processed = []
        for t in item:
            img_processed.append(t)
        img_processed = np.concatenate(img_processed, axis=1)
        imgs.append(img_processed)
    imgs = np.concatenate(imgs, axis=0)
    print(imgs.shape)
    return imgs


def main():
    img_path_list = ['test.jpg', 'flower.jpg', 'scene.jpg', 'texture.jpg']
    img_list = [cv2.imread(x) for x in img_path_list]
    handle = map(img_process_func, img_list)
    result = list(handle)
    handle = map(save_img_list, result)
    result = list(handle)

    for i in range(len(result)):
        path = img_path_list[i].split('.')[0]+'_process.jpg'
        cv2.imwrite(path, result[i])


if __name__ == "__main__":
    main()
