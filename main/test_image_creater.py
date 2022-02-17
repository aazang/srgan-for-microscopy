import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np

# Fine tuned parameters for pretrained models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


# Creating a TestImage Class
class TestImage():
    def __init__(self, input_path):
        self.input_path = input_path
        self.name = input_path.name
        self.original_width, self.original_height = Image.open(input_path).size

    def get_input_path(self):
        return self.input_path

    def get_pixels(self):
        opened_image = Image.open(self.input_path)
        pixels = list(opened_image.getdata())
        width, height = opened_image.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        return pixels

    def show_image(self):
        opened_image = Image.open(self.input_path).show()
        return opened_image

    def hr_transform(self, hr_desired):
        opened_image = Image.open(self.input_path)
        self.hr_transform = transforms.Compose([
                                            transforms.Resize((hr_desired, hr_desired), Image.BICUBIC),
                                            transforms.ToTensor(),
                                            # transforms.ToPILImage(),
                                            transforms.Normalize(mean, std),
                                        ])
        return self.hr_transform(opened_image)

    def lr_transform(self, hr_desired):
        opened_image = Image.open(self.input_path)
        self.lr_transform = transforms.Compose([
                                            transforms.Resize((hr_desired // 4, hr_desired // 4), Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std),
                                        ])
        return self.lr_transform(opened_image)
