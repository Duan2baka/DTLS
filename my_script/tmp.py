from PIL import Image
import sys
sys.path.append(".")
sys.path.append("..")

image = Image.open("./test_images/16_128_lr_image/blur_1.png")

width, height = image.width, image.height

print("width ", width, "height ", height)