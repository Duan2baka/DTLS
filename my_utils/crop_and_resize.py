import cv2

image = cv2.imread('000010.png')

x, y, width, height = 0, 40, image.shape[1], image.shape[0]

cropped_img = image[20:height-20, x:x+width]

# cv2.imshow('crop', cropped_img)

# cv2.waitKey(0)

# cv2.destroyAllWindows()
resize = cv2.resize(cropped_img, (128,128))

cv2.imwrite('image12.png', resize)