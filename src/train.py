import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

from src.model import SignalClassifier

# Set random seeds for reproducibility across PyTorch, NumPy, and Python
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

FEATURE_COLS = ["Return", "SMA_10", "SMA_50", "Volatility", "RSI", "MACD", "MACD_Signal", "BB_upper", "BB_lower", "Momentum_5", "Momentum_10", "Volume_Change"]

def train_model(feat_df,
                epochs: int = 50,
                batch_size: int = 32,
                lr: float = 1e-3,
                test_ratio: float = 0.2,
                patience: int = 10):
    """
    Chronological train/test split (Section V-A) then train the classifier.
    Returns trained model, scaler, and the test split dataframe.
    """
    # --- Chronological split (no shuffling) ---
    split = int(len(feat_df) * (1 - test_ratio))
    train_df = feat_df.iloc[:split]
    test_df  = feat_df.iloc[split:]

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df["Label"].values.astype(np.int64)
    X_test  = test_df[FEATURE_COLS].values.astype(np.float32)

    # Fit scaler only on train to avoid look-ahead bias
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # --- PyTorch datasets ---
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    loader   = DataLoader(train_ds, batch_size=batch_size, shuffle=False)

    # --- Model, loss, optimiser (Section IV-B/C) ---
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=y_train)
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model     = SignalClassifier(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

   best_loss = float('inf')
   patience_counter = 0

    model.train()
    #Training Loop
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimiser.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}")

        # Early stopping — stop if loss hasn't improved in 'patience' epochs
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    return model, scaler, test_df, X_test
