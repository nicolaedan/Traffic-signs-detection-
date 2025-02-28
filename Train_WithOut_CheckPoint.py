import torch
import torchvision
import os
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.ops as ops
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class GTSRB_Dataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["Path"]
        img = Image.open(img_path).convert('RGB')

        x_min, y_min, x_max, y_max = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
        boxes = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        labels = torch.tensor([row['ClassId']], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        return F.to_tensor(img), target

    def __len__(self):
        return len(self.df)

train_csv = "Train.csv"
test_csv = "Test.csv"

train_dataset = GTSRB_Dataset(train_csv)
test_dataset = GTSRB_Dataset(test_csv)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(43).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def accuracy(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    total_detected = 0
    correct_detections = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            predictions = model(images)

            for pred, target in zip(predictions, targets):
                pred_boxes = pred["boxes"]
                pred_labels = pred["labels"]
                pred_scores = pred["scores"]

                true_boxes = target["boxes"]
                true_labels = target["labels"]

                total_detected += len(pred_boxes)

                if len(true_boxes) == 0 or len(pred_boxes) == 0:
                    continue

                ious = ops.box_iou(pred_boxes, true_boxes)
                max_iou, matched_gt = ious.max(dim=1)

                matched_gts = set()
                for i, iou in enumerate(max_iou):
                    if iou > iou_threshold and matched_gt[i].item() not in matched_gts:
                        correct_detections += 1
                        matched_gts.add(matched_gt[i].item())

    precision = correct_detections / total_detected if total_detected > 0 else 0

    return precision

for epoch in range(0, 50):
    model.train()
    total_loss = 0

    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    precision = accuracy(model, test_loader, device)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    print(f"Precision: {precision:.2%}")

    model_save_path = f"fasterrcnn_gtsrb_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), model_save_path)
