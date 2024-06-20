import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.functional as F
from transformers import BeitImageProcessor, BeitForImageClassification, get_linear_schedule_with_warmup

def my_collate_fn(batch):
    batch = [x for x in batch if x is not None] 
    if not batch:
        return torch.Tensor(), torch.Tensor()
    return torch.utils.data.dataloader.default_collate(batch)

class WikiArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, class_txt, processor=None):
        self.img_labels = pd.read_csv(csv_file, sep=',', header=None, names=['path', 'classIndex'], encoding='utf-8')
        self.img_dir = img_dir
        self.processor = processor
        with open(class_txt, 'r') as f:
            self.class_index = f.read().splitlines()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            img_name = self.img_labels.iloc[idx, 0]
            img_path = os.path.join(self.img_dir, img_name)
            image = Image.open(img_path).convert('RGB')
            label = int(self.img_labels.iloc[idx, 1])
            if self.processor:
                inputs = self.processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].squeeze()
            return pixel_values, label
        except FileNotFoundError:
            return None

feature_extractor = BeitImageProcessor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

train_data = WikiArtDataset(csv_file='C:/Users/iop09/Desktop/wikiart_csv/style_train.csv',
                            img_dir='C:/Users/iop09/Desktop/archive/',
                            class_txt='C:/Users/iop09/Desktop/wikiart_csv/style_class.txt',
                            processor=feature_extractor)

val_data = WikiArtDataset(csv_file='C:/Users/iop09/Desktop/wikiart_csv/style_val.csv',
                          img_dir='C:/Users/iop09/Desktop/archive/',
                          class_txt='C:/Users/iop09/Desktop/wikiart_csv/style_class.txt',
                          processor=feature_extractor)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=my_collate_fn)
val_loader = DataLoader(val_data, batch_size=64, shuffle=True, collate_fn=my_collate_fn)

'''
class OtherModel(nn.Module):
    def __init__(self, num_classes=27):
        super(OtherModel, self).__init__()
        # Load a pretrained ResNet50 model
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(self.model.children())[:-1])

        in_features = self.model.fc.in_features  # Assuming this is the number of input features for the conv layer

        # Adjusted additional convolutional layers with an increased depth
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_features, in_features // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features // 2, in_features // 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features // 4),
            nn.ReLU(inplace=True)
        )

        # Adjusted classifier with an additional fully connected layer and dropout for regularization
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features // 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Apply adaptive pooling to reduce to a fixed size output
        x = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        x = self.conv_layers(x)
        x = self.classifier(x)

        return x
'''
class OtherModel(nn.Module):
    def __init__(self, num_labels=27):
        super(OtherModel, self).__init__()
        self.num_labels = num_labels
        self.beit = BeitForImageClassification.from_pretrained(
            'microsoft/beit-base-patch16-224-pt22k-ft22k', 
            num_labels=self.num_labels, 
            ignore_mismatched_sizes=True
        )

    def forward(self, pixel_values):
        outputs = self.beit(pixel_values=pixel_values)
        return outputs.logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OtherModel().to(device)
num_epochs = 2
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps = 0, num_training_steps = total_steps)

cost_function = nn.CrossEntropyLoss()

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')


for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    model.train()
    running_loss = 0.0
    running_corrects = 0
    processed_samples = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = cost_function(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_samples += inputs.size(0)

        if processed_samples >= 1000:
            average_loss = running_loss / processed_samples
            accuracy = running_corrects.double() / processed_samples
            print(f'Processed 1000 samples - Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')
            running_loss = 0.0
            running_corrects = 0
            processed_samples = 0

    epoch_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = cost_function(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_accuracy = val_corrects.double() / len(val_loader.dataset)
    print(f'Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    scheduler.step(val_loss)
    if val_loss < best_loss:
        print('Validation loss decreased, saving model...')
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, 'best_model.pth')

print('Finished Training')
