"""Custom modules and functions."""
from main import test_image_creater
from helper_programs import single_image_upscale

from torchvision.utils import save_image
from pathlib import Path
import torch
import os

def main():
    img = test_image_creater.TestImage(Path(r"./Test_Bilder/tip_upscaled.jpeg"))
    params = Path(r"./saved_models/PBCxDIV2K_randomcrops_ep2_batchs8_dim256x256/generator.pth")
    upscaled = single_image_upscale.image_upscale(img, params, 256, return_grid=False)

    save_image(upscaled, "./out/upscaled_pipette.jpeg", normalize=False)


if __name__ == "__main__":
    main()
