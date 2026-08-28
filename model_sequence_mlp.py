import torch
import torch.nn as nn

class SequenceMLPClassifier(nn.Module):
    """
    Čistě sekvenční baseline model.
    Klasifikuje protein přímo z jeho globálního ESM-2 sequence embeddingu (1280-dim),
    bez použití 3D kapes a bez chemických ligandových queries.
    """
    def __init__(self, in_features=1280, hidden_dim=256, num_classes=5, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        """
        x: [B, 1280] globální ESM-2 embedding celé sekvence proteinu.
        """
        return self.net(x)
