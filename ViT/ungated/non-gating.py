import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from datasets import load_dataset
from einops import rearrange
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

# ==========================================
# 1. DATASET DEFINITION
# ==========================================
@torch.no_grad()
def visualize_routing_decisions(model, dataset, device, epoch, image_index=0, save_dir='routing_visuals', use_gating=True):
    if not use_gating:
        print("    [INFO] Gating is disabled. Skipping routing visualization.")
        return

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    image_tensor, label_idx = dataset[image_index]
    image_batch = image_tensor.unsqueeze(0).to(device)

    outputs = model(image_batch, deterministic=True, use_gating=True)
    w_t = outputs.get('w_t')
    logits = outputs['logits']

    pred_idx = logits.argmax(dim=-1).item()

    try:
        hf_features = dataset.hf_split.features['label']
        true_name = hf_features.int2str(label_idx)
        pred_name = hf_features.int2str(pred_idx)
    except:
        true_name = f"Class {label_idx}"
        pred_name = f"Class {pred_idx}"

    routing_weights = w_t[0].mean(dim=0).cpu().numpy()
    grid_size = int(math.sqrt(len(routing_weights)))
    heatmap = routing_weights.reshape(grid_size, grid_size)

    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    heatmap_resized = cv2.resize(heatmap, (144, 144), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    title_color = "green" if label_idx == pred_idx else "red"
    
    axes[0].imshow(img_np)
    axes[0].set_title(f"True: {true_name}\nPred: {pred_name}", color=title_color, fontweight='bold')
    axes[0].axis('off')

    im = axes[1].imshow(heatmap_resized, cmap='jet', alpha=0.8, vmin=0, vmax=1)
    axes[1].set_title(f"MARL Agent Routing Map (Epoch {epoch})")
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_np)
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.4, vmin=0, vmax=1)
    axes[2].set_title("Overlay (High value = Retained)")
    axes[2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"routing_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close(fig) 
    
    print(f"    [SAVED PLOT] {save_path}")
    model.train() 

class HFImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = item['image']
        label = item['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class SimpleRouter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        half = max(d_model // 2, 16)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, half),
            nn.ELU(),
            nn.Linear(half, 1)
        )

    def forward(self, local_features: torch.Tensor):
        logits = self.mlp(local_features).squeeze(-1) # (B, N)
        w = torch.sigmoid(logits)
        return w

class ChunkwiseRALAAttention(nn.Module):
    def __init__(self, d_model: int, head: int = 8, chunk_size: int = 16, gamma: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        self.chunk_size = chunk_size
        self.gamma = gamma
        self.d_k = d_model // head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o_gate = nn.Linear(d_model, d_model)
        self.w_o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, w_gating=None, use_dilution=False, use_gating=True):
        b, n, d = x.shape
        T = n // self.chunk_size
        C = self.chunk_size

        q = rearrange(self.w_q(x), 'b (T C) (h dk) -> b T h C dk', T=T, C=C, h=self.head) * (self.d_k ** -0.25)
        k = rearrange(self.w_k(x), 'b (T C) (h dk) -> b T h C dk', T=T, C=C, h=self.head) * (self.d_k ** -0.25)
        v = rearrange(self.w_v(x), 'b (T C) (h dk) -> b T h C dk', T=T, C=C, h=self.head)

        phi_q = F.elu(q) + 1.0 
        phi_k = F.elu(k) + 1.0 

        if use_gating and w_gating is not None:
            w_chunks = rearrange(w_gating, 'b (T C) -> b T C', T=T, C=C)
            w_expanded = w_chunks.unsqueeze(2).unsqueeze(-1)  
            k_gated = w_expanded * phi_k                      
            k_gated_f32 = k_gated.to(torch.float32)
            v_f32 = v.to(torch.float32)
            KV_chunks = torch.matmul(k_gated_f32.transpose(-2, -1), v_f32)  
            Z_chunks = k_gated_f32.sum(dim=-2)                              
            w_bar = w_chunks.mean(dim=-1)  
        else:
            k_f32 = phi_k.to(torch.float32)
            v_f32 = v.to(torch.float32)
            KV_chunks = torch.matmul(k_f32.transpose(-2, -1), v_f32)  
            Z_chunks = k_f32.sum(dim=-2)  
            w_bar = torch.ones(b, T, device=x.device) 

        outputs = []
        S = torch.zeros(b, self.head, self.d_k, self.d_k, device=x.device, dtype=torch.float32)
        Z = torch.zeros(b, self.head, self.d_k, device=x.device, dtype=torch.float32)

        for t in range(T):
            decay_factor = 1.0 - (self.gamma * (1.0 - w_bar[:, t]))
            decay_S = decay_factor.view(b, 1, 1, 1)
            decay_Z = decay_factor.view(b, 1, 1)

            if use_dilution and t > 0:
                dilution_scale = self.gamma * (1.0 - w_bar[:, t])  
                gamma_tau = dilution_scale.view(b, 1, 1, 1) * S / max(t, 1)
                S = (S * decay_S) + KV_chunks[:, t] + gamma_tau
            else:
                S = (S * decay_S) + KV_chunks[:, t]

            Z = (Z * decay_Z) + Z_chunks[:, t]

            phi_q_t = phi_q[:, t].to(torch.float32)          
            nom = torch.matmul(phi_q_t, S)                   
            denom = (phi_q_t * Z.unsqueeze(-2)).sum(dim=-1, keepdim=True) + 1e-5

            out_t = nom / denom
            if self.training: out_t = self.dropout(out_t)
            out_t = torch.clamp(out_t, min=-65000.0, max=65000.0)
            outputs.append(out_t.to(q.dtype))

        out = torch.stack(outputs, dim=1)                    
        out = rearrange(out, 'b T h C dk -> b (T C) (h dk)')

        if use_gating:
            gate = torch.sigmoid(self.w_o_gate(x))                
            out = out * gate
            
        out = self.w_o_proj(out)
        return out, phi_k

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=144, patch_size=12, in_chans=3, embed_dim=256):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class EncoderBlock(nn.Module):
    def __init__(self, d_model, head, chunk_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ChunkwiseRALAAttention(d_model, head=head, chunk_size=chunk_size)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_model * 4)

    def forward(self, x, w_gating=None, use_dilution=False, use_gating=True):
        res = x
        x_normed = self.norm1(x)
        out, phi_k = self.attn(x_normed, w_gating, use_dilution, use_gating)
        x = res + out
        x = x + self.mlp(self.norm2(x))
        return x, phi_k

class ViT(nn.Module):
    def __init__(self, image_size=144, patch_size=12, num_classes=100,
                 d_model=256, depth=6, head=8, chunk_size=16):
        super().__init__()
        self.depth = depth
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, d_model))
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, head, chunk_size) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.router = SimpleRouter(d_model)

    def forward(self, x, use_dilution=False, deterministic=False, use_gating=True):
        x = self.patch_embed(x) + self.pos_embed  

        w_list = []
        for block in self.blocks:
            w = self.router(x) if use_gating else None
            x, phi_k = block(x, w, use_dilution, use_gating)
            if use_gating:
                w_list.append(w)

        x = self.norm(x)
        logits = self.head(x.mean(dim=1))  

        return {
            'logits': logits,
            'w_t': torch.stack(w_list, dim=1) if use_gating else None, 
        }

# ==========================================
# 3. HELPERS AND IMAGENET MAIN LOOP
# ==========================================
def compute_bimodal_sparsity(w_t):
    return (w_t * (1.0 - w_t)).mean()

@torch.no_grad()
def evaluate(model, val_loader, accelerator, epoch, use_gating=True):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in val_loader:
        outputs = model(images, deterministic=True, use_gating=use_gating)
        loss = criterion(outputs['logits'], labels)

        logits_gathered = accelerator.gather_for_metrics(outputs['logits'])
        labels_gathered = accelerator.gather_for_metrics(labels)
        
        preds = logits_gathered.argmax(dim=-1)
        correct += (preds == labels_gathered).sum().item()
        total += labels_gathered.size(0)
        total_loss += loss.item() * labels_gathered.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total * 100.0
    if accelerator.is_main_process:
        print(f"\n--- Validation [Epoch {epoch+1}] | Loss={avg_loss:.4f} | Accuracy={accuracy:.2f}% ---\n")
    model.train()

def main():
    # --- DDP FIX ADDED HERE ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # --- HYPERPARAMETERS ---
    USE_GATING = False           # Toggle Baseline (False) vs MARL (True)
    epochs = 40
    batch_size = 64
    lr = 1e-4
    max_lambda_budget = 0.05    
    warmup_epochs = 10          
    target_density = 0.5        
    lambda_sparse = 1.0         

    if accelerator.is_main_process:
        dir_name = f"gating_True_budget_{max_lambda_budget}" if USE_GATING else "baseline_ungated"
        ckpt_dir = f"checkpoints_imagenet/{dir_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n--- Starting ImageNet-100 {'Gated' if USE_GATING else 'Baseline'} Training ---")

    hf_train = load_dataset("clane9/imagenet-100", split="train")
    hf_val = load_dataset("clane9/imagenet-100", split="validation")

    transform = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HFImageNetDataset(hf_train, transform=transform)
    val_dataset = HFImageNetDataset(hf_val, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ViT(image_size=144, patch_size=12, num_classes=100, d_model=256, depth=16, head=8, chunk_size=16)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_ce_none = nn.CrossEntropyLoss(reduction='none')

    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    model.train()

    for epoch in range(epochs):
        current_lambda_budget = max_lambda_budget * (epoch / warmup_epochs) if epoch < warmup_epochs else max_lambda_budget
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]") if accelerator.is_main_process else train_loader

        for images, labels in loop:
            optimizer.zero_grad()
            outputs = model(images, use_dilution=True, use_gating=USE_GATING)
            
            loss_ce = criterion_ce_none(outputs['logits'], labels).mean()
            
            # Safe calculation: only apply budget/sparse loss if gating is ON
            w_t = outputs.get('w_t')
            if USE_GATING and w_t is not None:
                budget_loss = current_lambda_budget * ((w_t.mean() - target_density) ** 2)
                sparse_loss = lambda_sparse * compute_bimodal_sparsity(w_t)
            else:
                budget_loss = torch.tensor(0.0, device=accelerator.device)
                sparse_loss = torch.tensor(0.0, device=accelerator.device)
            
            total_loss = loss_ce + budget_loss + sparse_loss
            accelerator.backward(total_loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if accelerator.is_main_process:
                preds = outputs['logits'].argmax(dim=-1)
                acc = (preds == labels).float().mean().item() * 100
                if USE_GATING:
                    loop.set_postfix(Loss=f"{loss_ce.item():.3f}", Bdgt=f"{budget_loss.item():.3f}", R=f"{w_t.mean().item()*100:.1f}%", Acc=f"{acc:.1f}%")
                else:
                    loop.set_postfix(Loss=f"{loss_ce.item():.3f}", Acc=f"{acc:.1f}%")

        accelerator.wait_for_everyone()
        evaluate(model, val_loader, accelerator, epoch, use_gating=USE_GATING)
        
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            ckpt_path = f"{ckpt_dir}/vit_epoch{epoch+1}.pth"
            torch.save(unwrapped_model.state_dict(), ckpt_path)
            print(f'    [SAVED] {ckpt_path}')


# ==========================================
# 4. CIFAR-10 TRANSFER LEARNING
# ==========================================
class HFCifarDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        image = item['img']  # CIFAR uses 'img'
        label = item['label']
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def finetune_cifar10():
    # DDP Penalty removed for speed (since frozen layers don't need gradient syncing)
    accelerator = Accelerator()
    
    # ==========================================
    # --- HYPERPARAMETERS & CHECKPOINT CONFIG ---
    # ==========================================
    USE_GATING = False           # Toggle Baseline (False) vs MARL/Sigmoid (True)
    epochs = 40
    batch_size = 128
    lr = 1e-3
    
    # Transfer Learning Settings
    LOAD_CHECKPOINT = True
    CHECKPOINT_PATH = 'checkpoints_imagenet/baseline_ungated/vit_epoch15.pth' 
    LINEAR_PROBING = True        # <--- SET TO TRUE: Only trains the final layer!

    if accelerator.is_main_process:
        dir_name = f"gating_{USE_GATING}_transfer" if LOAD_CHECKPOINT else f"gating_{USE_GATING}_scratch"
        ckpt_dir = f"checkpoints_cifar10/{dir_name}"
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"\n--- Starting CIFAR-10 {'Gated' if USE_GATING else 'Baseline'} Training ---")
        if LOAD_CHECKPOINT:
            print(f"    Loading weights from: {CHECKPOINT_PATH}")
            print(f"    Linear Probing mode: {'ON (Backbone Frozen)' if LINEAR_PROBING else 'OFF (Full Finetuning)'}")

    # 1. Load Hugging Face CIFAR-10
    hf_cifar_train = load_dataset("cifar10", split="train")
    hf_cifar_val = load_dataset("cifar10", split="test")

    transform_train = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((144, 144)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = HFCifarDataset(hf_cifar_train, transform=transform_train)
    val_dataset = HFCifarDataset(hf_cifar_val, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ==========================================
    # 2. MODEL INSTANTIATION & WEIGHT LOADING
    # ==========================================
    if LOAD_CHECKPOINT:
        # Step A: Instantiate with 100 classes to match the saved ImageNet weights
        model = ViT(image_size=144, patch_size=12, num_classes=100, d_model=256, depth=16, head=8, chunk_size=16)
        
        # Step B: Load the checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False))
        else:
            if accelerator.is_main_process:
                print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}. Aborting.")
            return

        # Step C: Swap the classification head for CIFAR-10 (10 classes)
        model.head = nn.Linear(model.head.in_features, 10)
        
        # Step D: Apply Linear Probing (Freeze everything except the head)
        if LINEAR_PROBING:
            for name, param in model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
    else:
        # Training from scratch directly on 10 classes
        model = ViT(image_size=144, patch_size=12, num_classes=10, d_model=256, depth=16, head=8, chunk_size=16)

    # 3. Setup Optimizer
    # ONLY pass the head parameters to the optimizer if Linear Probing is ON
    params_to_optimize = model.head.parameters() if (LOAD_CHECKPOINT and LINEAR_PROBING) else model.parameters()
    optimizer = optim.AdamW(params_to_optimize, lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 4. Training Loop
    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"CIFAR Epoch [{epoch+1}/{epochs}]") if accelerator.is_main_process else train_loader

        for images, labels in loop:
            optimizer.zero_grad()
            
            outputs = model(images, deterministic=True, use_dilution=False, use_gating=USE_GATING)
            loss = criterion(outputs['logits'], labels)
            
            accelerator.backward(loss)
            optimizer.step()

            if accelerator.is_main_process:
                preds = outputs['logits'].argmax(dim=-1)
                acc = (preds == labels).float().mean().item() * 100
                loop.set_postfix(Loss=f"{loss.item():.4f}", Acc=f"{acc:.1f}%")

        accelerator.wait_for_everyone()
        evaluate(model, val_loader, accelerator, epoch, use_gating=USE_GATING)

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = f"{ckpt_dir}/vit_epoch{epoch+1}.pth"
            torch.save(unwrapped_model.state_dict(), save_path)
            print(f"    [SAVED CKPT] {save_path}")

if __name__ == "__main__":
    # Choose which pipeline to run:
    
    # main()  # ImageNet Training
    finetune_cifar10() # CIFAR-10 Training
