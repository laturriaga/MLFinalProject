import torch

def test_model(model, dataloader, device, using_test=True):
    model = model.to(device)
    model.eval()
    successes = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        pred_classes = torch.argmax(outputs, dim=1)
        batch_success = torch.where(pred_classes == targets, 1.0, 0.0)
        successes.append(batch_success)
    accuracy = torch.cat(successes, dim=0).mean()
    if using_test:
        print(f'Testing accuracy: {100*accuracy:.4f}%')
    else:
        print(f'Training accuracy: {100*accuracy:.4f}%')
    return accuracy.item()