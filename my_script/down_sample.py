import os
import argparse
import torch
import tqdm
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image

def interpolate_images(input_folder, output_folder, size, mode, output_folder_128):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_128, exist_ok=True)

    for filename in tqdm.tqdm(os.listdir(input_folder)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)

            img_tensor = to_tensor(img)  # Convert PIL image to PyTorch tensor

            img_1 = F.interpolate(img_tensor.unsqueeze(0), size=size, mode=mode)
            img_2 = img_1
            img_1 = img_1.squeeze(0)

            img_interpolated = to_pil_image(img_1)  # Convert the tensor back to PIL image

            save_path = os.path.join(output_folder, filename)
            img_interpolated.save(save_path)
            #print(f"Interpolated and saved: {filename}")

            img_1 = F.interpolate(img_2, size=128, mode=mode)
            img_1 = img_1.squeeze(0)
            img_interpolated = to_pil_image(img_1)  # Convert the tensor back to PIL image

            save_path = os.path.join(output_folder_128, filename)
            img_interpolated.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolate PNG images.")
    parser.add_argument("input_folder", help="Input folder containing PNG images.")
    parser.add_argument("output_folder", help="Output folder to save the interpolated images.")
    parser.add_argument("output_folder_128", help="Output folder to save the interpolated images.")
    parser.add_argument("--size", type=int, default=16, help="Size for interpolation.")
    parser.add_argument("--mode", default="bilinear", help="Interpolation mode.")

    args = parser.parse_args()

    interpolate_images(args.input_folder, args.output_folder, args.size, args.mode, args.output_folder_128)