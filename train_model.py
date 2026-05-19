# ============================================================
# FINAL TRANSFORMER MODEL (DEPLOYMENT READY)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from hydroeval import evaluator, kge

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ============================================================
# 1. LOAD DATA
# ============================================================

# IMPORTANT:
# Put TRANSFORMERDATA.xlsx in SAME folder as this file

df = pd.read_excel("TRANSFORMERDATA.xlsx")

# ============================================================
# 2. DATE PROCESSING
# ============================================================

df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# ============================================================
# 3. FEATURE ENGINEERING
# ============================================================

# Create lag features
for i in range(1, 6):
    df[f'wl_lag{i}'] = df['wl'].shift(i)

# Rolling average
df['wl_roll3'] = df['wl'].rolling(3).mean()

# Future targets
leads = [1, 3, 5, 7, 10]

for lead in leads:
    df[f'wl_t+{lead}'] = df['wl'].shift(-lead)

# Remove NaN rows
df.dropna(inplace=True)

# ============================================================
# 4. FEATURES & TARGETS
# ============================================================

features = (
    ['imerg'] +
    [f'wl_lag{i}' for i in range(1, 6)] +
    ['wl_roll3']
)

targets = [f'wl_t+{lead}' for lead in leads]

X = df[features].values
y = df[targets].values

# ============================================================
# 5. SCALING
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler for deployment
joblib.dump(scaler, "scaler.pkl")

print("Scaler saved successfully!")

# ============================================================
# 6. CREATE SEQUENCES
# ============================================================

sequence_length = 10

X_seq = []
y_seq = []

for i in range(sequence_length, len(X_scaled)):
    X_seq.append(X_scaled[i-sequence_length:i])
    y_seq.append(y[i])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

split = int(0.8 * len(X_seq))

X_train = X_seq[:split]
X_test = X_seq[split:]

y_train = y_seq[:split]
y_test = y_seq[split:]

# ============================================================
# 8. TRANSFORMER BLOCK
# ============================================================

def transformer_block(
    inputs,
    num_heads=4,
    key_dim=16,
    ff_dim=64,
    rate=0.2
):

    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(inputs, inputs)

    out1 = LayerNormalization()(inputs + attention_output)

    ffn_output = Dense(ff_dim, activation='relu')(out1)
    ffn_output = Dropout(rate)(ffn_output)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)

    return LayerNormalization()(out1 + ffn_output)

# ============================================================
# 9. BUILD MODEL
# ============================================================

inp = Input(
    shape=(X_train.shape[1], X_train.shape[2])
)

x = transformer_block(inp)

x = GlobalAveragePooling1D()(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.3)(x)

out = Dense(len(leads))(x)

model = Model(inputs=inp, outputs=out)

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# ============================================================
# 10. TRAINING
# ============================================================

callbacks = [
    EarlyStopping(
        patience=10,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        patience=5,
        factor=0.5
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 11. SAVE MODEL
# ============================================================

model.save("kosi_transformer_model.keras")

print("Model saved successfully!")

# ============================================================
# 12. PREDICTIONS
# ============================================================

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# ============================================================
# 13. EVALUATION
# ============================================================

def evaluate_all(y_true, y_pred, name):

    print(f"\n{name} PERFORMANCE")

    for i, lead in enumerate(leads):

        r = np.corrcoef(
            y_true[:, i],
            y_pred[:, i]
        )[0, 1]

        rmse = np.sqrt(
            mean_squared_error(
                y_true[:, i],
                y_pred[:, i]
            )
        )

        nse = 1 - np.sum(
            (y_true[:, i] - y_pred[:, i])**2
        ) / np.sum(
            (y_true[:, i] - np.mean(y_true[:, i]))**2
        )

        mae = mean_absolute_error(
            y_true[:, i],
            y_pred[:, i]
        )

        kge_val = evaluator(
            kge,
            y_pred[:, i],
            y_true[:, i]
        )[0].item()

        print(
            f"Lead {lead} -> "
            f"R:{r:.3f}, "
            f"RMSE:{rmse:.3f}, "
            f"NSE:{nse:.3f}, "
            f"MAE:{mae:.3f}, "
            f"KGE:{kge_val:.3f}"
        )

# Run evaluation
evaluate_all(y_train, pred_train, "TRAIN")
evaluate_all(y_test, pred_test, "TEST")

# ============================================================
# 14. SAVE CSV RESULTS
# ============================================================

train_output = pd.DataFrame()

for i, lead in enumerate(leads):
    train_output[f'Actual_t+{lead}'] = y_train[:, i]
    train_output[f'Predicted_t+{lead}'] = pred_train[:, i]

train_output.to_csv(
    "Train_Predictions.csv",
    index=False
)

test_output = pd.DataFrame()

for i, lead in enumerate(leads):
    test_output[f'Actual_t+{lead}'] = y_test[:, i]
    test_output[f'Predicted_t+{lead}'] = pred_test[:, i]

test_output.to_csv(
    "Test_Predictions.csv",
    index=False
)

print("\nTrain and Test CSV files saved!")

# ============================================================
# 15. PLOTS
# ============================================================

for i, lead in enumerate(leads):

    plt.figure(figsize=(10, 4))

    plt.plot(
        y_test[:100, i],
        label='Actual'
    )

    plt.plot(
        pred_test[:100, i],
        label='Predicted'
    )

    plt.title(f"Lead {lead} Days")

    plt.legend()

    plt.grid()

    plt.show()

print("\nALL TASKS COMPLETED SUCCESSFULLY!")