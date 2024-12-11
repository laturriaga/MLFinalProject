from tqdm import tqdm
from test_model import test_model


def train_model(model, loss_func, optimizer, train_loader, test_loader, num_epochs, device):
    model = model.to(device)
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch_idx in tqdm(range(num_epochs)):
        model.train()
        batch_losses = []
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        if epoch_idx % 1 == 0:
            train_accuracy = test_model(model, train_loader, device, using_test=False)
            test_accuracy = test_model(model, test_loader, device, using_test=True)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
        epoch_loss = sum(batch_losses) / len(batch_losses)
        print(f'Epoch {epoch_idx + 1} Loss: {epoch_loss:.4f}')
        losses.append(epoch_loss)
    return model, losses, train_accuracies, test_accuracies
