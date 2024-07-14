import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from data.dataset_e2e import get_dataloaders  # Replace with the actual name of your script file
from transformers import ViTConfig, ViTForImageClassification

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ViT model on test dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--distributed', action='store_true', help='Use distributed evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on')
    parser.add_argument('--resume_ckpt', type=str, required=True, help='Path to the checkpoint to evaluate')
    return parser.parse_known_args()

def init_distributed():
    dist_url = "env://"  # default
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )
    dist.barrier()

def load_model(resume_ckpt, device):
    config = ViTConfig(image_size=256, num_labels=2, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = ViTForImageClassification(config)
    model = model.to(device)

    ckpt = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)

    # Remove "module." prefix from keys
    state_dict = ckpt["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    return model

def evaluate(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs = batch['imgs'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs).logits
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy()[:, 1])

    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_outputs)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    return accuracy, auc, f1, precision, recall

def main(args):
    if args.distributed:
        init_distributed()

    num_mesh_images = [5, 5]
    _, test_loader = get_dataloaders(args, num_mesh_images = [-1,num_mesh_images])

    model = load_model(args.resume_ckpt, args.device)

    if args.distributed:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    accuracy, auc, f1, precision, recall = evaluate(model, test_loader, args.device)

    if not args.distributed or (args.distributed and int(os.environ['RANK']) == 0):
        print(f'Accuracy: {accuracy:.4f}')
        print(f'AUC: {auc:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate ViT model on test dataset')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--distributed', action='store_true', help='Use distributed evaluation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the evaluation on')
    parser.add_argument('--resume_ckpt', type=str, required=True, help='Path to the checkpoint to evaluate')
    parser.add_argument('--n_pnts', type=int, default=32, help='Number of points for the dataset')  # Added this line
    args, unknown = parser.parse_known_args()

    main(args)
