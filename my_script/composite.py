from PIL import Image
import os

folder_path = 'generation_results'
output_size = (512, 512) 

image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')])[:16]

output_image = Image.new('RGB', output_size)

for i, image_file in enumerate(image_files):
    image_path = os.path.join(folder_path, image_file)
    image = Image.open(image_path)

    image = image.resize((output_size[0] // 4, output_size[1] // 4))

    x = (i % 4) * (output_size[0] // 4)
    y = (i // 4) * (output_size[1] // 4)

    output_image.paste(image, (x, y))

output_image.save('tmp/output_image.jpg')