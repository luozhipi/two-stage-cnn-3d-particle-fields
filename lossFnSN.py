import torch

def ridgeTV(height, width, para1, para2):
    def loss(y_true, y_pred):
        a = torch.square(y_pred[:, :int(height - 1), :int(width - 1), :] - y_pred[:, 1:, :int(width - 1), :])
        b = torch.square(y_pred[:, :int(height - 1), :int(width - 1), :] - y_pred[:, :int(height - 1), 1:, :])
        total_variation = torch.sum(torch.pow(a + b, para1))  # good para1: 1
        total_variation = para2 * total_variation  # good para2: 0.0001
        l2_loss = (1 - para2) * torch.sum(torch.square(y_true - y_pred))
        return (l2_loss + total_variation) / height / width
    return loss
