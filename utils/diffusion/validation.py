from tqdm import tqdm


def validate(ema_model, valloader, writer, step, device):
    
    loader = tqdm(valloader, desc='Validation loop')
    loss = 0.0

    for mel, _ in loader:
        mel = mel.unsqueeze(1).to(device)
        loss += ema_model(mel)

    loss = loss / len(valloader.dataset)

    writer.log_validation(loss, step)