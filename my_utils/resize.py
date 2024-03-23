import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt

list_img = [os.path.join('test_images/128_inpaint',img) for img in os.listdir('test_images/128_inpaint') if (img.startswith("image"))]
list_mask = [os.path.join('test_images/128_inpaint',img) for img in os.listdir('test_images/128_inpaint') if (img.startswith("mask"))]
list_img.sort()
list_mask.sort()
for idx, (img_path, mask_path) in enumerate(zip(list_img, list_mask)):
    
    imgname = os.path.splitext(os.path.basename(img_path))[0]
    maskname = os.path.splitext(os.path.basename(mask_path))[0]
    print(imgname, maskname)
    # read image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
    img = cv2.resize(img, (16,16))
    newmask = cv2.resize(mask, (16,16),interpolation=cv2.INTER_NEAREST)
    #mask = newmask.reshape((16,16,1))
    newmask = np.where(newmask > 0.5, 255, 0).astype(np.uint8)
    mask = np.expand_dims(newmask, axis=-1)
    new_img = img * (mask/ 255.)
    plt.imshow(newmask)
    plt.show()
    new_img = new_img*255
    cv2.imwrite(f'test_images/16_inpaint/image0{idx}.png',new_img)
    cv2.imwrite(f'test_images/16_inpaint/mask0{idx}.png',newmask)