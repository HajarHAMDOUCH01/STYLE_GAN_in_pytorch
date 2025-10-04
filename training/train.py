import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os
import gc
from pathlib import Path
from torch.amp import GradScaler, autocast
import torch.nn.functional as F 

import sys  
sys.path.append("/content/Face-Generator-StyleGAN-PyTorch")

from model.style_gan import StyleGAN, Discriminator
from training_config import training_config


torch.backends.cudnn.benchmark = True
scaler = GradScaler("cuda")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CelebADataset(Dataset):
    """CelebA dataset loader"""
    def __init__(self, root, transform=None, limit=None):
        self.root = Path(root)
        self.transform = transform
        self.image_files = sorted(list(self.root.glob("*.jpg")))[:limit]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_transforms(img_size):
    """Data preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])


def compute_gradient_penalty(discriminator, real_images, fake_images):
    """R1 regularization (paper section C, equation from Mescheder et al.)"""
    real_images.requires_grad_(True)
    real_scores = discriminator(real_images)
    
    gradients = torch.autograd.grad(
        outputs=real_scores.sum(),
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # R1 penalty: ||∇D(x)||²
    r1_penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
    return r1_penalty


def discriminator_loss(real_scores, fake_scores):
    # For real images: maximize log(D(x))
    real_loss = F.softplus(-real_scores).mean()
    # For fake images: maximize log(1 - D(G(z)))
    fake_loss = F.softplus(fake_scores).mean()
    return real_loss + fake_loss


def generator_loss(fake_scores):
    # Maximize log(D(G(z)))
    return F.softplus(-fake_scores).mean()


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def generate_samples(generator, device, epoch, save_dir, num_samples=16):
    """Generate and save sample images"""
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, generator.z_dim, device=device)
        samples = generator.generate(z, truncation_psi=0.7)
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = (samples + 1) / 2
        samples = torch.clamp(samples, 0, 1)
        
        import torchvision.utils as vutils
        grid = vutils.make_grid(samples, nrow=4, padding=2, normalize=False)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title(f'Generated Samples - Epoch {epoch}')
        save_path = os.path.join(save_dir, f'samples_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    generator.train()


def load_from_checkpoint(generator, discriminator, g_optimizer, d_optimizer, checkpoint_path=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    def strip_prefix(state_dict, prefix="_orig_mod."):
        """Remove torch.compile() wrapper prefix from state dict keys"""
        return {
            k.replace(prefix, ""): v 
            for k, v in state_dict.items()
        }
    
    gen_state = checkpoint["generator_state_dict"]
    if any(k.startswith("synthesis._orig_mod") for k in gen_state.keys()):
        # Checkpoint has _orig_mod prefix in synthesis module
        gen_state_fixed = {}
        for k, v in gen_state.items():
            if k.startswith("synthesis._orig_mod."):
                # Remove _orig_mod from synthesis keys
                new_key = k.replace("synthesis._orig_mod.", "synthesis.")
                gen_state_fixed[new_key] = v
            else:
                gen_state_fixed[k] = v
        gen_state = gen_state_fixed
    
    generator.load_state_dict(gen_state)
    
    # Load discriminator state dict with prefix handling  
    disc_state = checkpoint["discriminator_state_dict"]
    if any("_orig_mod" in k for k in disc_state.keys()):
        disc_state = strip_prefix(disc_state)
    
    discriminator.load_state_dict(disc_state)
    
    # Optimizers should load fine
    g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
    d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
    
    start_epoch = checkpoint["epoch"] + 1
    
    print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return generator, discriminator, g_optimizer, d_optimizer, start_epoch

def train_stylegan(config, checkpoint_path=None):
    
    img_size = config["image_size"]
    z_dim = config["z_dim"]
    w_dim = config["w_dim"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    generator = StyleGAN(
        z_dim=z_dim,
        w_dim=w_dim,
        img_size=img_size,
        img_channels=3,
        mapping_layers=config["mapping_layers"],
        style_mixing_prob=config["style_mixing_prob"]
    ).to(device)
    generator = generator.to(memory_format=torch.channels_last)

    discriminator = Discriminator(
        img_size=img_size,
        img_channels=3
    ).to(device)
    discriminator = discriminator.to(memory_format=torch.channels_last)
    
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config["g_lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_eps"]
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config["d_lr"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        eps=config["adam_eps"]
    )
    
    # Dataset
    transform = get_transforms(img_size)
    dataset = CelebADataset(
        root=config["dataset_path"],
        transform=transform,
        limit=10000  # the images in this repo is of a 10000 subset 
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config["num_workers"] > 0 else False
    )
    
    print(f"\nTraining Configuration:")
    print(f"- Image size: {img_size}x{img_size}")
    print(f"- Z dimension: {z_dim}")
    print(f"- W dimension: {w_dim}")
    print(f"- Batch size: {batch_size}")
    print(f"- Generator LR: {config['g_lr']}")
    print(f"- Discriminator LR: {config['d_lr']}")
    print(f"- Style mixing prob: {config['style_mixing_prob']}")
    print(f"- R1 gamma: {config['r1_gamma']}")
    print(f"- Dataset size: {len(dataset)}")
    print(f"- Total epochs: {num_epochs}\n")

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        generator, discriminator, g_optimizer, d_optimizer, start_epoch = load_from_checkpoint(generator, discriminator, g_optimizer, d_optimizer, checkpoint_path)
        print(f"resuming from {checkpoint_path} at epoch {start_epoch}")
    else:
        print(f"starting training from epoch 0")
    
    generator.synthesis = torch.compile(generator.synthesis)
    discriminator = torch.compile(discriminator)



    g_losses = []
    d_losses = []
    r1_penalties = []

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        
        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_r1 = 0
        num_batches = 0
        g_updates = 0  # Track generator updates separately
        
        loop = tqdm(dataloader, leave=True, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, real_images in enumerate(loop):
            real_images = real_images.to(device, non_blocking=True)
            batch_size_actual = real_images.size(0)
            
            # ==================== Train Discriminator ====================
            d_optimizer.zero_grad()
            
            # Generate fake images
            z1 = torch.randn(batch_size_actual, z_dim, device=device)
            z2 = torch.randn(batch_size_actual, z_dim, device=device)
            
            with torch.no_grad():
                fake_images, _ = generator(z1, z2)
            
            # Discriminator scores
            real_scores = discriminator(real_images)
            fake_scores = discriminator(fake_images.detach())
            
            # Discriminator loss
            d_loss = discriminator_loss(real_scores, fake_scores)
            
            # R1 regularization
            r1_interval = config.get("r1_interval", 16)
            r1_penalty = torch.tensor(0.0, device=device)
            
            if batch_idx % r1_interval == 0:
                r1_penalty = compute_gradient_penalty(discriminator, real_images, fake_images)
                d_loss_total = d_loss + config["r1_gamma"] * r1_penalty * r1_interval
                epoch_r1 += r1_penalty.item()
            else:
                d_loss_total = d_loss
            
            scaler.scale(d_loss_total).backward()
            
            if config["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config["grad_clip"])
            
            scaler.step(d_optimizer)
            scaler.update()

            # ==================== Train Generator (Multiple Times) ====================
            n_critic = config.get("n_critic", 2)  # Train G this many times per D update
            
            for _ in range(n_critic):
                g_optimizer.zero_grad()
                
                # Generate new fake images
                z1 = torch.randn(batch_size_actual, z_dim, device=device)
                z2 = torch.randn(batch_size_actual, z_dim, device=device)
                
                fake_images, _ = generator(z1, z2)
                
                # Generator loss
                fake_scores = discriminator(fake_images)
                g_loss = generator_loss(fake_scores)
                
                scaler.scale(g_loss).backward()
                
                if config["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), config["grad_clip"])
                
                scaler.step(g_optimizer)
                scaler.update()
                
                epoch_g_loss += g_loss.item()
                g_updates += 1
            
            epoch_d_loss += d_loss.item()
            num_batches += 1
            
            loop.set_postfix({
                "G_loss": f"{g_loss.item():.4f}",
                "D_loss": f"{d_loss.item():.4f}",
                "R1": f"{r1_penalty.item():.4f}" if r1_penalty.item() > 0 else "0.0000",
            })
            
            if batch_idx % 100 == 0:
                clear_memory()
        
        avg_g_loss = epoch_g_loss / g_updates  # Divide by actual number of G updates
        avg_d_loss = epoch_d_loss / num_batches
        avg_r1 = epoch_r1 / max(1, num_batches // r1_interval)
    
        
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        r1_penalties.append(avg_r1)
        
        print(f"\nEpoch {epoch+1} Complete:")
        print(f"  Generator Loss: {avg_g_loss:.4f}")
        print(f"  Discriminator Loss: {avg_d_loss:.4f}")
        print(f"  R1 Penalty: {avg_r1:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config["save_every"] == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
                'r1_penalties': r1_penalties,
                'config': config  
            }
            checkpoint_path = os.path.join(save_dir, f'stylegan_checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Generate samples
        if (epoch + 1) % config["sample_every"] == 0:
            generate_samples(generator, device, epoch + 1, save_dir)
        
        clear_memory()
    
    # Save final model
    print("\nTraining complete! Saving final model...")
    final_path = os.path.join(save_dir, 'stylegan_final.pth')
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'config': config
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    # Plot training curves
    plot_training_curves(g_losses, d_losses, r1_penalties, save_dir)
    
    return generator, discriminator, g_losses, d_losses


def plot_training_curves(g_losses, d_losses, r1_penalties, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(g_losses, label='Generator Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(d_losses, label='Discriminator Loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Discriminator Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(r1_penalties, label='R1 Penalty', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Penalty')
    axes[2].set_title('R1 Regularization')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved: {save_path}")


if __name__ == "__main__":
    
    generator, discriminator, g_losses, d_losses = train_stylegan(training_config, checkpoint_path="/content/drive/MyDrive/stylegan_checkpoints/stylegan_checkpoint_epoch_22.pth")
    print("\n✓ Training completed successfully!")