import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# ====================== CONFIG ======================
SEQ_LEN = 30
NUM_FEATURES = 63
NUM_CLASSES = 10

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = "best_cnn_gru_model.h5"
# ====================================================


def load_sequences(csv_path):
    df = pd.read_csv(csv_path)
    X, y = [], []
    for clip_id, clip_df in df.groupby("clip_id"):
        clip_df = clip_df.sort_values("frame_number")
        if len(clip_df) != SEQ_LEN:
            continue
        features = clip_df[[f"f{i}" for i in range(NUM_FEATURES)]].values
        label = clip_df["label"].iloc[0]
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


def create_model():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(SEQ_LEN, NUM_FEATURES)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(128, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        GRU(128, return_sequences=True),
        Dropout(0.4),
        GRU(64),

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def main():
    print("Loading data...")
    X_train, y_train = load_sequences("train.csv")
    X_val, y_val = load_sequences("val.csv")
    X_test, y_test = load_sequences("test.csv")

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape  : {X_val.shape}")
    print(f"Test shape : {X_test.shape}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    model = create_model()
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                         save_best_only=True, verbose=1)
    ]

    print("\nTraining started...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Loss    : {test_loss:.4f}")

    # ---- Per-class evaluation: this is what actually tells you if the
    # ---- model has collapsed onto one or two classes, which a single
    # ---- overall accuracy number can hide.
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\nClassification report (per class):")
    print(classification_report(y_test, y_pred, digits=3))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved as: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()