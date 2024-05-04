import os, numpy as np, sys
import PIL
import cv2

maskPath_list = ['Fold1_label', 'Fold2_label', 'Fold3_label', 'Fold4_label', 'Fold5_label']
picturePath_list = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']


for i, path in enumerate(maskPath_list):
    save_mask_path = os.path.join('preprocess', path)
    save_picture_path = os.path.join('preprocess', picturePath_list[i])
    os.makedirs(save_mask_path)
    os.makedirs(save_picture_path)
    mask_list = os.listdir(path)
    line_start = False
    upper = 0
    downer = 0
    for mask in mask_list:
        img = cv2.imread(os.path.join(path, mask), cv2.IMREAD_GRAYSCALE)
        y, x = np.where(img != 0)
        min_pos = (x.min(), y.min())
        max_pos = (x.max(), y.max())
        x_len = x.max() - x.min()
        y_len = y.max() - y.min()
        middle_pos = (x.min() + (x_len)//2, y.min() + (y_len)//2)
        cap_start_pos = img[y.min() : y.min() + y_len, middle_pos[0]-(y_len//2) : middle_pos[0] + (y_len//2)]
        cv2.imwrite(os.path.join(save_mask_path, mask), cap_start_pos)
        mask = mask.replace('png', 'jpg')
        mask = mask.replace('mask', 'img')
        img2 = cv2.imread(os.path.join(picturePath_list[i], mask), cv2.IMREAD_GRAYSCALE)
        cap_start_pos = img2[y.min() : y.min() + y_len, middle_pos[0]-(y_len//2) : middle_pos[0] + (y_len//2)]
        cv2.imwrite(os.path.join(save_picture_path, mask), cap_start_pos)
        #print(np.where(img != 0))


#for line in img:
#    print(line)

