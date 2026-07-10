import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data_loader import get_data
from models import Generator, Discriminator

def train_gan():
    # Hyperparameters
    ticker = "TSLA"  # High volatility ticker as requested
    seq_len = 24
    batch_size = 64
    z_dim = 10
    hidden_dim = 64
    num_epochs = 100
    lr = 0.0002
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data Ingestion
    dataloader, _, _ = get_data(ticker=ticker, seq_len=seq_len, batch_size=batch_size)
    
    # 2. GAN Architecture
    G = Generator(z_dim=z_dim, hidden_dim=hidden_dim, out_dim=1).to(device)
    D = Discriminator(input_dim=1, hidden_dim=hidden_dim).to(device)
    
    # Binary Cross Entropy Loss for the Minimax Game
    criterion = nn.BCELoss()
    
    # Adam optimizers for both Generator and Discriminator
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # History for plotting
    G_losses = []
    D_losses = []
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # 3. Training Loop (Minimax Game)
        for batch_idx, real_data in enumerate(dataloader):
            real_data = real_data.to(device)
            current_batch_size = real_data.size(0)
            
            # Ground truth labels for Real (1s) and Fake (0s)
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)
            
            # ============================================
            # Train Discriminator: Maximize log(D(x)) + log(1 - D(G(z)))
            # ============================================
            opt_D.zero_grad()
            
            # Pass real data through D
            out_real = D(real_data)
            loss_D_real = criterion(out_real, real_labels)
            
            # Generate fake data from noise
            z = torch.randn(current_batch_size, seq_len, z_dim).to(device)
            fake_data = G(z)
            
            # Pass fake data through D (detach to avoid backprop through G)
            out_fake = D(fake_data.detach())
            loss_D_fake = criterion(out_fake, fake_labels)
            
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            
            # Gradient Clipping for Discriminator
            # This mitigates the exploding gradient problem common in RNNs/LSTMs
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            opt_D.step()
            
            # ============================================
            # Train Generator: Maximize log(D(G(z)))
            # ============================================
            opt_G.zero_grad()
            
            # We want D to classify our fake data as Real (1)
            out_fake_for_G = D(fake_data)
            loss_G = criterion(out_fake_for_G, real_labels)
            loss_G.backward()
            
            # Gradient Clipping for Generator
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            opt_G.step()
            
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        
        G_losses.append(avg_g_loss)
        D_losses.append(avg_d_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Loss D: {avg_d_loss:.4f} | Loss G: {avg_g_loss:.4f}")
            
    # Save the trained generator weights
    torch.save(G.state_dict(), "generator_model.pth")
    print("Training Complete. Model saved to 'generator_model.pth'")
    
    # Save loss graph
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_graph.png")
    plt.close()
    print("Saved training loss graph to 'loss_graph.png'")

if __name__ == "__main__":
    train_gan()
