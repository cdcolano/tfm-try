import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from data.dataset_e2e import get_dataloaders

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)  # (B, C, H, W) -> (HW, B, C)
        x = self.transformer_encoder(x)
        x = x.permute(1, 2, 0).view(x.size(1), x.size(2), int(x.size(0)**0.5), int(x.size(0)**0.5))  # (HW, B, C) -> (B, C, H, W)
        return self.linear(x)

class AttentionResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(AttentionResNet, self).__init__()
        self.resnet = resnet18(pretrained=pretrained)
        self.self_attention = SelfAttentionBlock(d_model=512, nhead=8, num_layers=1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.self_attention(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser(description='Train Attention ResNet on custom dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--n_pnts', type=int, default=32, help='Number of points for the dataset')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--save_path', type=str, default='./models', help='Path to save the models')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    return parser.parse_known_args()

def setup_distributed(args):
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()

def main():
    args, unknown = parse_args()

    if args.distributed:
        setup_distributed(args)

    # Get the dataloaders
    train_loader, test_loader = get_dataloaders(args, num_mesh_images=[5, 5])

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionResNet(num_classes=2, pretrained=True)
    model = model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training function
    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        with tqdm(train_loader, desc='Training', leave=False) as tbar:
            for batch in tbar:
                images = batch['imgs'].to(device)
                labels = batch['labels'].to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tbar.set_postfix(loss=running_loss / (tbar.n + 1))

        return running_loss / len(train_loader)

    # Validation function
    def validate(model, test_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        with tqdm(test_loader, desc='Validation', leave=False) as tbar:
            for batch in tbar:
                images = batch['imgs'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                tbar.set_postfix(loss=running_loss / (tbar.n + 1), accuracy=100 * correct / total)

        accuracy = 100 * correct / total
        return running_loss / len(test_loader), accuracy

    # Main training loop
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        train_loss = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = validate(model, test_loader, criterion, device)

        if not args.distributed or (args.distributed and args.local_rank == 0):
            print(f'Train Loss: {train_loss:.4f}, '
                  f'Test Loss: {test_loss:.4f}, '
                  f'Test Accuracy: {test_accuracy:.2f}%')

            # Save the model
            os.makedirs(args.save_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_path, f'attention_resnet_epoch{epoch+1}.pth'))

if __name__ == '__main__':
    main()
