import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

num_classes = 43
model_path = "Faster_RCNN_Model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
            "Limită de viteză 20 km/h", "Limită de viteză 30 km/h", "Limită de viteză 50 km/h",
            "Limită de viteză 60 km/h", "Limită de viteză 70 km/h", "Limită de viteză 80 km/h",
            "Sfârșitul limitei de viteză de 80 km/h", "Limită de viteză 100 km/h", "Limită de viteză 120 km/h",
            "Depășirea interzisă", "Depășirea interzisă pentru camioane", "Prioritate la următoarea intersecție",
            "Drum cu prioritate", "Cedează trecerea", "STOP", "Acces interzis vehiculelor",
            "Acces interzis vehiculelor peste 3.5 tone", "Intrare interzisă", "Atenție generală",
            "Curbă periculoasă la stânga", "Curbă periculoasă la dreapta", "Dublă curbă periculoasă",
            "Drum denivelat", "Drum alunecos", "Drumul se îngustează pe dreapta", "Lucrări pe carosabil",
            "Semnale rutiere", "Traversare pietoni", "Atenție copii", "Traversare bicicliști",
            "Atenție, gheață/zăpadă pe carosabil", "Trecere de animale sălbatice",
            "Sfârșitul tuturor limitărilor de viteză și depășire", "Obligatoriu la dreapta înainte",
            "Obligatoriu la stânga înainte", "Doar înainte", "Mergi înainte sau la dreapta",
            "Mergi înainte sau la stânga", "Menține banda dreaptă", "Menține banda stângă",
            "Sens giratoriu obligatoriu", "Sfârșitul drumului cu prioritate", "Mergi înainte"
]

def load_model(model_path, num_classes):
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(model_path, num_classes)

image_path = "1.png"
image = Image.open(image_path).convert("RGB")

transform = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    predictions = model(img_tensor)

boxes = predictions[0]["boxes"].cpu().numpy()
labels = predictions[0]["labels"].cpu().numpy()
scores = predictions[0]["scores"].cpu().numpy()

for i in range(len(boxes)):
    if scores[i] > 0.9:
        x_min, y_min, x_max, y_max = boxes[i]
        print(f"Bounding Box {i+1}: [({x_min:.1f}, {y_min:.1f}), ({x_max:.1f}, {y_max:.1f})], "f"Class: {class_names[labels[i]]}, Confidence: {scores[i]:.2%}")

fig, ax = plt.subplots(1, figsize=(8, 6))
ax.imshow(image)

for i in range(len(boxes)):
    if scores[i] > 0.9:
        x_min, y_min, x_max, y_max = boxes[i]
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x_min, y_min, f"{class_names[labels[i]]} ({scores[i]:.2%})", color="white",bbox=dict(facecolor="red", alpha=0.5))

plt.show()
