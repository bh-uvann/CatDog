import gradio as gr
import cv2 as cv
import torch,torch.nn as nn


class ImageClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 56
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 28
            
            nn.AdaptiveAvgPool2d((7, 7)) 
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) # No Sigmoid here if using BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = ImageClassification() 

# 2. Load the weights
model.load_state_dict(torch.load('cat_dog_model.pth'))

model.eval()


def predict_image(img):
    # Preprocess the input image (same as your training)
    img = cv.resize(img, (224, 224))
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
        prediction = model(img).item()
    
    # Return as a dictionary for a nice label UI
    return {"Dog": prediction, "Cat": 1 - prediction}

interface = gr.Interface(
    fn=predict_image, 
    inputs=gr.Image(), 
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs Dog Classifier"
)
interface.launch(share=True) # share=True gives you a public URL for 72 hours!