import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import os
from datetime import datetime

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Load data
def load_data(data_path='preprocessed_fer2013.npz'):
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: File '{data_path}' not found!")
        exit()
    
    data = np.load(data_path, allow_pickle=True)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    emotion_names = data['emotion_names']
    
    # Reshape for CNN
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_val = X_val.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"\nDataset shapes:")
    print(f"  Training:   {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test:       {X_test.shape}")
    print(f"\nClasses: {len(emotion_names)}")
    print(f"Emotions: {list(emotion_names)}")
    
    # Check class distribution
    print(f"\nClass distribution:")
    for i, emotion in enumerate(emotion_names):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"  {emotion}: Train={train_count}, Test={test_count}")
    
    print("="*60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, emotion_names

# Build IMPROVED model
def build_improved_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Improved CNN with better architecture for emotion recognition
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Compute class weights
def get_class_weights(y_train):
    """
    Compute class weights to handle imbalanced dataset
    """
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("\n📊 Class weights (to handle imbalance):")
    for i, weight in class_weight_dict.items():
        print(f"  Class {i}: {weight:.2f}")
    
    return class_weight_dict

# Create FIXED data augmentation
def create_data_generator():
    """
    FIXED: Simpler data augmentation
    """
    datagen = ImageDataGenerator(
        rotation_range=10,           # Reduced from 20
        width_shift_range=0.1,       # Reduced from 0.15
        height_shift_range=0.1,      # Reduced from 0.15
        horizontal_flip=True,
        zoom_range=0.1,              # Reduced from 0.15
        fill_mode='nearest'
        # Removed brightness_range - causing issues
    )
    return datagen

# Setup callbacks
def setup_callbacks():
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    
    callbacks = [
        ModelCheckpoint(
            'fer2013_model_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(log_dir=log_dir)
    ]
    
    return callbacks

# Train model
def train_model(model, X_train, y_train, X_val, y_val, class_weights,
                epochs=50, batch_size=64, use_augmentation=True):
    """
    Train with class weights and proper augmentation
    """
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = setup_callbacks()
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Data augmentation: {use_augmentation}")
    print(f"Class weights: Yes (to handle imbalance)")
    print("="*60)
    
    if use_augmentation:
        print("\n🔄 Training with data augmentation...")
        datagen = create_data_generator()
        
        # FIX: Proper way to use fit_generator
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,  # IMPORTANT: Fixed steps
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,  # Handle imbalance
            verbose=1
        )
    else:
        print("\n📊 Training without augmentation...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
    
    return history

# Plot history
def plot_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    ax1.plot(epochs_range, history.history['accuracy'], 'b-o', label='Training', linewidth=2)
    ax1.plot(epochs_range, history.history['val_accuracy'], 'r-s', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, history.history['loss'], 'b-o', label='Training', linewidth=2)
    ax2.plot(epochs_range, history.history['val_loss'], 'r-s', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Training history saved as '{save_path}'")

# Evaluate
def evaluate_model(model, X_test, y_test):
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'='*60}")
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    return test_acc, test_loss, y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, emotion_names, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, 
                yticklabels=emotion_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('True', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved as '{save_path}'")

# Classification report
def print_classification_report(y_true, y_pred, emotion_names):
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=emotion_names, digits=4, zero_division=0)
    print(report)

# Visualize predictions
def visualize_predictions(model, X_test, y_test, emotion_names, n_samples=16):
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    predictions = model.predict(X_test[indices], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(48, 48)
        true_label = emotion_names[y_test[idx]]
        pred_label = emotion_names[pred_labels[i]]
        confidence = predictions[i][pred_labels[i]] * 100
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                         fontsize=10, fontweight='bold', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Prediction samples saved")

# MAIN
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FER2013 CNN - IMPROVED VERSION")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, emotion_names = load_data()
    num_classes = len(emotion_names)
    
    # Compute class weights
    class_weights = get_class_weights(y_train)
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    model = build_improved_cnn(input_shape=(48, 48, 1), num_classes=num_classes)
    
    print("\nModel Summary:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        class_weights=class_weights,
        epochs=50,
        batch_size=64,
        use_augmentation=True
    )
    
    # Plot
    plot_history(history)
    
    # Evaluate
    test_acc, test_loss, y_pred = evaluate_model(model, X_test, y_test)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, emotion_names)
    
    # Report
    print_classification_report(y_test, y_pred, emotion_names)
    
    # Samples
    print("\nGenerating prediction samples...")
    visualize_predictions(model, X_test, y_test, emotion_names)
    
    # Save
    model.save('fer2013_final_model.keras')
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    print("\nGenerated files:")
    print("  1. fer2013_model_best.keras")
    print("  2. fer2013_final_model.keras")
    print("  3. training_history.png")
    print("  4. confusion_matrix.png")
    print("  5. prediction_samples.png")
    print("="*60)
def load_data(data_path='preprocessed_fer2013.npz'):
    """
    Load preprocessed FER2013 data (train/test split)
    """
    print("="*60)
    print("LOADING PREPROCESSED DATA")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: File '{data_path}' not found!")
        print("Please run the preprocessing script first: python preprocess_fer2013.py")
        exit()
    
    data = np.load(data_path, allow_pickle=True)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    emotion_names = data['emotion_names']
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, 48, 48, 1)
    X_val = X_val.reshape(-1, 48, 48, 1)
    X_test = X_test.reshape(-1, 48, 48, 1)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"\nDataset shapes:")
    print(f"  Training:   {X_train.shape} - Labels: {y_train.shape}")
    print(f"  Validation: {X_val.shape} - Labels: {y_val.shape}")
    print(f"  Test:       {X_test.shape} - Labels: {y_test.shape}")
    print(f"\nNumber of classes: {len(emotion_names)}")
    print(f"Emotions: {list(emotion_names)}")
    print(f"Pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print("="*60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, emotion_names

# Build Deep CNN Model (Recommended)
def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Deep CNN architecture for emotion recognition
    - 4 Convolutional blocks with BatchNorm and Dropout
    - 2 Fully connected layers
    - Best for accuracy
    """
    model = models.Sequential([
        # Block 1: 64 filters
        layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2: 128 filters
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3: 256 filters
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4: 512 filters
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(512, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected layers
        layers.Flatten(),
        
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Build Simple CNN Model (Faster training)
def build_simple_cnn(input_shape=(48, 48, 1), num_classes=7):
    """
    Lighter CNN for faster training and testing
    - 4 Convolutional blocks
    - 2 Fully connected layers
    - Good for quick experiments
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fully connected
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Data augmentation generator
def create_data_generator():
    """
    Create data augmentation for training
    Improves generalization by creating variations
    """
    datagen = ImageDataGenerator(
        rotation_range=20,              # Rotate up to 20 degrees
        width_shift_range=0.15,         # Shift horizontally
        height_shift_range=0.15,        # Shift vertically
        horizontal_flip=True,           # Mirror image
        zoom_range=0.15,                # Zoom in/out
        fill_mode='nearest',            # Fill empty pixels
        brightness_range=[0.8, 1.2]     # Adjust brightness
    )
    return datagen

# Setup callbacks
def setup_callbacks(model_name='fer2013_model'):
    """
    Setup training callbacks for better training
    """
    # Create logs directory
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    
    callbacks = [
        # Save best model
        ModelCheckpoint(
            f'{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard for visualization
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]
    
    return callbacks

# Train model
def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=50, batch_size=64, use_augmentation=True):
    """
    Train the CNN model
    """
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Setup callbacks
    callbacks = setup_callbacks()
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Data augmentation: {use_augmentation}")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Loss: Sparse Categorical Crossentropy")
    print("="*60)
    
    # Train
    if use_augmentation:
        print("\n🔄 Training with data augmentation...")
        datagen = create_data_generator()
        datagen.fit(X_train)
        
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("\n📊 Training without data augmentation...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    
    return history

# Plot training history
def plot_history(history, save_path='training_history.png'):
    """
    Visualize training progress
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy plot
    ax1.plot(epochs_range, history.history['accuracy'], 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
    ax1.plot(epochs_range, history.history['val_accuracy'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Loss plot
    ax2.plot(epochs_range, history.history['loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
    ax2.plot(epochs_range, history.history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Training history saved as '{save_path}'")

# Evaluate model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate on test set
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL ON TEST SET")
    print("="*60)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"{'='*60}")
    
    # Predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    return test_acc, test_loss, y_pred

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, emotion_names, save_path='confusion_matrix.png'):
    """
    Create confusion matrix visualization
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, 
                yticklabels=emotion_names,
                cbar_kws={'label': 'Count'},
                linewidths=1, linecolor='gray')
    
    plt.xlabel('Predicted Emotion', fontsize=13, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix - FER2013 Emotion Recognition', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✓ Confusion matrix saved as '{save_path}'")

# Classification report
def print_classification_report(y_true, y_pred, emotion_names):
    """
    Detailed per-class metrics
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred, target_names=emotion_names, digits=4)
    print(report)

# Visualize predictions
def visualize_predictions(model, X_test, y_test, emotion_names, n_samples=16):
    """
    Show predictions vs ground truth
    """
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    predictions = model.predict(X_test[indices], verbose=0)
    pred_labels = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(48, 48)
        true_label = emotion_names[y_test[idx]]
        pred_label = emotion_names[pred_labels[i]]
        confidence = predictions[i][pred_labels[i]] * 100
        
        axes[i].imshow(img, cmap='gray')
        
        # Green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%', 
                         fontsize=10, fontweight='bold', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Prediction samples saved as 'prediction_samples.png'")

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FER2013 CNN EMOTION RECOGNITION MODEL")
    print("="*60)
    
    # Configuration
    EPOCHS = 50
    BATCH_SIZE = 64
    USE_AUGMENTATION = True
    MODEL_TYPE = 'deep'  # 'deep' or 'simple'
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, emotion_names = load_data()
    num_classes = len(emotion_names)
    
    # Build model
    print("\n" + "="*60)
    print("BUILDING MODEL")
    print("="*60)
    
    if MODEL_TYPE == 'deep':
        print("Using: Deep CNN (Best accuracy)")
        model = build_cnn_model(input_shape=(48, 48, 1), num_classes=num_classes)
    else:
        print("Using: Simple CNN (Faster training)")
        model = build_simple_cnn(input_shape=(48, 48, 1), num_classes=num_classes)
    
    print("\nModel Architecture:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
    
    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        use_augmentation=USE_AUGMENTATION
    )
    
    # Plot history
    plot_history(history)
    
    # Evaluate
    test_acc, test_loss, y_pred = evaluate_model(model, X_test, y_test)
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, emotion_names)
    
    # Classification report
    print_classification_report(y_test, y_pred, emotion_names)
    
    # Visualize predictions
    print("\nGenerating prediction samples...")
    visualize_predictions(model, X_test, y_test, emotion_names)
    
    # Save final model
    model.save('fer2013_final_model.keras')
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"\n📈 Final Test Accuracy: {test_acc*100:.2f}%")
    print(f"\n📁 Generated files:")
    print("  1. fer2013_model_best.keras    - Best model during training")
    print("  2. fer2013_final_model.keras   - Final trained model")
    print("  3. training_history.png        - Accuracy/Loss curves")
    print("  4. confusion_matrix.png        - Performance matrix")
    print("  5. prediction_samples.png      - Sample predictions")
    print("  6. logs/                       - TensorBoard logs")
    print(f"\n💡 View TensorBoard: tensorboard --logdir=logs")
    print("="*60)