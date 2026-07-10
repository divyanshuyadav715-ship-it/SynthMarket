import numpy as np
from scipy.stats import entropy
import torch
from models import Generator

class ModelDriftMonitor:
    def __init__(self):
        """
        Calculates the baseline probability distribution of the synthetic market data
        used during training. This acts as our "Ground Truth" for drift detection.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(z_dim=10, hidden_dim=64, out_dim=1).to(self.device)
        try:
            self.generator.load_state_dict(torch.load("generator_model.pth", map_location=self.device))
            self.generator.eval()
            self._generate_baseline()
        except Exception as e:
            print("Warning: Generator not found. Drift Monitor will use uniform baseline.")
            self.baseline_dist = np.ones(50) / 50.0
            self.bins = np.linspace(-1, 1, 51)
            
    def _generate_baseline(self):
        with torch.no_grad():
            # Generate a large sample of the training distribution
            z = torch.randn(500, 24, 10).to(self.device)
            baseline_data = self.generator(z).cpu().numpy().flatten()
            
        # Create a PDF (Probability Density Function) histogram
        counts, self.bins = np.histogram(baseline_data, bins=50, density=True)
        self.baseline_dist = counts + 1e-8  # Add epsilon to prevent log(0) in KL Divergence
        self.baseline_dist /= np.sum(self.baseline_dist)
        
    def check_drift(self, live_market_data, threshold=0.1):
        """
        Monitors Model Drift using KL Divergence.
        Calculates the statistical distance between the Training Distribution and the Live Market Distribution.
        """
        if len(live_market_data) < 2:
            return False, 0.0
            
        # Calculate distribution of the incoming live data
        counts, _ = np.histogram(live_market_data, bins=self.bins, density=True)
        live_dist = counts + 1e-8
        live_dist /= np.sum(live_dist)
        
        # KL Divergence: How much does the live data diverge from what the agent learned?
        kl_divergence = entropy(live_dist, self.baseline_dist)
        
        # If divergence is too high, the market regime has changed -> Trigger Drift Alert
        drift_alert = kl_divergence > threshold
        return drift_alert, kl_divergence
