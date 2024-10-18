import wandb
from torcheval.metrics import R2Score, Mean
import torch

def train(model, trainloader, optimizer, criterion_reg, epoch, device):
    TrainR2 = R2Score(device=device)
    TrainRegLoss = Mean(device=device)

    # Training block
    model.train()
    for batch, (subj_id, data, mni_coords, targets, binTargets) in enumerate(trainloader):
        # Move tensors to configured device
        data = data.to(device)
        targets_regression = targets.to(device)
        subj_id = subj_id.long().to(device)
        mni_coords = mni_coords.to(device)

        # Keep only 1.5 sec after the stimulus color-change
        data = data[:, :, :, 400:1000]

        # Forward Pass
        optimizer.zero_grad()
        reg = model(data, subj_id, mni_coords)

        # Calculate loss for the classification and regression task
        loss = criterion_reg(reg, targets_regression)
        # Backprop
        loss.backward()
        optimizer.step()

        TrainRegLoss.update(loss.detach(), weight=data.size(0))
        TrainR2.update(reg.squeeze(1), targets_regression.squeeze(1))

    res = {'lr': optimizer.param_groups[0]["lr"],
           'TrainLossReg': TrainRegLoss.compute(),
           'TrainR2': TrainR2.compute()}

    wandb.log(res, step=epoch)

    return model, res['TrainLossReg'], res['TrainR2']


def evaluate(model, loader, criterion_reg, epoch, device):
    # Turn on evaluation mode which disables dropout.

    ValidR2 = R2Score(device=device)
    ValidRegLoss = Mean(device=device)

    model.eval()
    with torch.no_grad():
        for batch, (subj_id, data, mni_coords, targets, binTargets) in enumerate(loader):
            # Move tensors to configured device
            data = data.to(device)
            targets_regression = targets.to(device)
            subj_id = subj_id.long().to(device)
            mni_coords = mni_coords.to(device)

            # Forward pass through model
            data = data[:, :, :, 400:1000]

            # Tokenize:
            reg = model(data, subj_id, mni_coords)

            # Calculate loss for the classification and regression task
            loss = criterion_reg(reg, targets_regression)

            # Record the loss across the validation set
            ValidRegLoss.update(loss.detach(), weight=data.size(0))
            ValidR2.update(reg.squeeze(1), targets_regression.squeeze(1))

    res = {'ValidLossReg': ValidRegLoss.compute(), 'ValidR2': ValidR2.compute()}

    wandb.log(res, step=epoch)

    return res['ValidLossReg'], res['ValidR2']

def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            # print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()