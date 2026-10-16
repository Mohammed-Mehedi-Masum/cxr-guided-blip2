import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def build_densenet_classifier(device="cuda"):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.LayerNorm(512),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )
    for p in model.features.parameters(): p.requires_grad = False
    for p in model.features.denseblock3.parameters(): p.requires_grad = True
    for p in model.features.transition3.parameters(): p.requires_grad = True
    for p in model.features.denseblock4.parameters(): p.requires_grad = True
    for p in model.features.norm5.parameters(): p.requires_grad = True
    return model.to(device)

def run_mc_dropout(model, image_tensor, num_passes=30, device="cuda"):
    model.train()
    probs = []
    image_tensor = image_tensor.to(device)
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(image_tensor)
            probs.append(F.softmax(logits, dim=1)[:, 1].item())
    mean_p = float(torch.tensor(probs).mean())
    std_p = float(torch.tensor(probs).std(unbiased=True))
    pred_label = 1 if mean_p >= 0.5 else 0
    return mean_p, pred_label, std_p

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.features.denseblock4
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        self.target_layer.register_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, target_class].backward()
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = F.relu(torch.mean(self.activations, dim=1).squeeze())
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap.detach().cpu().numpy()

def load_blip2_lora(model_id="Salesforce/blip2-opt-2.7b", r=16, alpha=32):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, load_in_4bit=True, device_map="auto")
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, bias="none", target_modules=["q_proj", "v_proj", "k_proj", "out_proj"])
    return processor, get_peft_model(model, cfg)
