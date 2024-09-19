import torch.nn as nn
from timm import create_model

def build_model(num_classes=196):
    model = create_model('efficientnet_b1', pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model
