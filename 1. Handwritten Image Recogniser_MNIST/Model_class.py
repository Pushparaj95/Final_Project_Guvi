import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
import torchvision.transforms as transforms

def preprocess_canvas_image(combined_image):
    inverted_image = ImageOps.invert(combined_image)

    # Trim excess background
    bbox = inverted_image.getbbox()
    cropped_image = inverted_image.crop(bbox)

    # Make the image square by adding padding
    max_dim = max(cropped_image.size)
    delta_w = max_dim - cropped_image.size[0]
    delta_h = max_dim - cropped_image.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    square_image = ImageOps.expand(cropped_image, padding, fill=0)

    # Resize to 28x28 as expected by the model
    resized_image = square_image.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(resized_image).unsqueeze(0)  # Add batch dimension

    return image_tensor

class ConvNet(nn.Module):
    def __init__(self, dropout=0.0):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout = nn.Dropout(dropout)      
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)         
        x = F.relu(self.fc1(x))
        x = self.dropout(x)                
        x = F.relu(self.fc2(x))               
        x = self.fc3(x)                      
        return x