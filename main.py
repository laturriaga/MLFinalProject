import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from train_model import train_model
from test_model import test_model

if __name__ == '__main__':
    #--- Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    num_epochs = 25
    learning_rate = 1e-4
    batch_size = 16
    #---

    #--- Train and test paths
    data_dir = "./data"  # Assuming "data" folder is in the same directory as the notebook
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    #---

    #--- Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }
    #---

    #--- Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    #---

    #--- Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #---

    #--- Printing class names
    # class_names = train_dataset.classes
    # print("Classes:", class_names)
    #---

    #--- Instantiating model, loss function, and optimizer
    # model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #---

    #--- Freeze only the first few layers (efficientnet_b0 has a "features" module containing multiple blocks)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    for name, param in model.named_parameters():
        # print(name)
        if "features.0" in name or "features.1" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    #---

    #--- Replace dense layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    #---

    model, losses, train_accuracies, test_accuracies = train_model(model, loss_func, optimizer, train_loader, test_loader, num_epochs, device)

    torch.save(model.state_dict(), 'trained_models/efficientnet-b1-2.pth')

    epochs = range(0, 5 * len(train_accuracies), 5)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')

    plt.title('Train vs Test Accuracy')
    plt.xlabel('Epochs (every 5 epochs)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(losses)
    plt.show()