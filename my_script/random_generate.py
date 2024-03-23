import os
import random
from PIL import Image

save_dir = "./random_images"
os.makedirs(save_dir, exist_ok=True)

for i in range(100):
    image = Image.new("RGBA", (16, 16))
    pixels = []
    for _ in range(16 * 16):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        a = 255
        pixels.append((r, g, b, a))

    image.putdata(pixels)

    image = image.resize((128, 128), Image.BOX)

    save_path = os.path.join(save_dir, f"image_{i}.png")
    image.save(save_path)

    print(f"saved at {save_path}")

print("random generation complete")