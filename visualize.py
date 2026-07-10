import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import get_data
from models import Generator

def visualize():
    ticker = "TSLA"
    seq_len = 24
    z_dim = 10
    hidden_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Get Real Data
    print("Fetching real historical data...")
    _, _, real_scaled_returns = get_data(ticker=ticker, seq_len=seq_len, batch_size=1)
    real_scaled_returns = real_scaled_returns.flatten()
    
    # 2. Load Trained Generator
    print("Loading Trained Generator...")
    G = Generator(z_dim=z_dim, hidden_dim=hidden_dim, out_dim=1).to(device)
    try:
        G.load_state_dict(torch.load("generator_model.pth", map_location=device))
        G.eval()
    except FileNotFoundError:
        print("Model file 'generator_model.pth' not found. Please run train.py first.")
        return
        
    # 3. Generate Synthetic Data
    print("Generating synthetic data...")
    num_samples = 1000
    with torch.no_grad():
        z = torch.randn(num_samples, seq_len, z_dim).to(device)
        fake_data = G(z).cpu().numpy().squeeze()
    
    fake_scaled_returns = fake_data.flatten()
    
    # 4. Validation Plot
    plt.figure(figsize=(16, 6))
    
    # Plot 1: Line Graph comparison for a few sequences (Temporal Dynamics)
    plt.subplot(1, 2, 1)
    # Plot 3 real sequences
    for i in range(3):
        plt.plot(real_scaled_returns[i*seq_len:(i+1)*seq_len], color='blue', alpha=0.6, 
                 label='Real' if i==0 else "")
    # Plot 3 fake sequences
    for i in range(3):
        plt.plot(fake_data[i], color='red', alpha=0.6, linestyle='dashed',
                 label='Synthetic' if i==0 else "")
        
    plt.title(f"Sample Sequences: Real vs Synthetic ({ticker})")
    plt.xlabel("Time Step (Days)")
    plt.ylabel("Scaled Returns")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Probability Density Function (PDF) overlap (Statistical Distribution)
    plt.subplot(1, 2, 2)
    # Using matplotlib's histogram density to approximate PDF
    plt.hist(real_scaled_returns, bins=60, density=True, alpha=0.5, color='blue', label='Real Data')
    plt.hist(fake_scaled_returns, bins=60, density=True, alpha=0.5, color='red', label='Synthetic Data')
    
    plt.title("Probability Density Function (PDF) Overlap")
    plt.xlabel("Scaled Returns")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("validation_plot.png")
    print("\nSUCCESS! Saved validation plot to 'validation_plot.png'")
    print("Check out the PDF overlap. If it's a close match, the Generator has successfully captured the market dynamics!")
    plt.show()

if __name__ == "__main__":
    visualize()
