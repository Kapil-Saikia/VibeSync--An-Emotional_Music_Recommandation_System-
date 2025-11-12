import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# Load images from train and test folders separately
def load_fer2013_dataset(base_path):
    """
    Load FER2013 dataset with train/test folder structure:
    base_path/
        train/
            angry/
                img1.jpg
                img2.jpg
            happy/
            ...
        test/
            angry/
            happy/
            ...
    """
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    
    # Check if paths exist
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train folder not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test folder not found at: {test_path}")
    
    print("="*60)
    print("LOADING FER2013 DATASET")
    print("="*60)
    
    # Load training data
    print("\n📁 Loading TRAINING data...")
    X_train, y_train, emotion_names = load_images_from_folder(train_path)
    
    # Load test data (use same emotion_names order)
    print("\n📁 Loading TEST data...")
    X_test, y_test, _ = load_images_from_folder(test_path, emotion_names)
    
    # Create validation split from training data (10%)
    print("\n📊 Creating validation split from training data...")
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Total samples: {len(X_train) + len(X_val) + len(X_test)}")
    print(f"Emotions: {list(emotion_names)}")
    print(f"Number of classes: {len(emotion_names)}")
    print("="*60)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, emotion_names

def load_images_from_folder(folder_path, emotion_names=None):
    """
    Load all images from emotion subfolders
    """
    images = []
    labels = []
    
    # Get emotion folders
    if emotion_names is None:
        emotion_folders = sorted([f for f in os.listdir(folder_path) 
                                if os.path.isdir(os.path.join(folder_path, f))])
        emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotion_folders)}
        emotion_names = emotion_folders
    else:
        emotion_to_label = {emotion: idx for idx, emotion in enumerate(emotion_names)}
        emotion_folders = emotion_names
    
    print(f"\nEmotion folders: {emotion_folders}")
    
    total_images = 0
    
    # Load images from each emotion folder
    for emotion in emotion_folders:
        emotion_path = os.path.join(folder_path, emotion)
        
        if not os.path.exists(emotion_path):
            print(f"⚠️  Warning: Folder '{emotion}' not found, skipping...")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(emotion_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\n  {emotion}: {len(image_files)} images")
        
        label = emotion_to_label[emotion]
        loaded = 0
        failed = 0
        
        # Try to use tqdm for progress bar
        try:
            from tqdm import tqdm
            iterator = tqdm(image_files, desc=f"    Loading", ncols=80, leave=False)
        except ImportError:
            iterator = image_files
        
        for img_file in iterator:
            img_path = os.path.join(emotion_path, img_file)
            
            # Read image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize to 48x48 if needed
                if img.shape != (48, 48):
                    img = cv2.resize(img, (48, 48))
                
                images.append(img)
                labels.append(label)
                loaded += 1
            else:
                failed += 1
        
        print(f"    ✓ Loaded: {loaded} | ✗ Failed: {failed}")
        total_images += loaded
    
    # Convert to numpy arrays
    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')
    
    # Normalize to [0, 1]
    images = images / 255.0
    
    print(f"\n  Total loaded: {total_images} images")
    print(f"  Shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    return images, labels, emotion_names

# Plot class distribution
def plot_class_distribution(y_train, y_val, y_test, emotion_names):
    """
    Plot distribution across train/val/test sets
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = [
        (y_train, 'Training Set', axes[0]),
        (y_val, 'Validation Set', axes[1]),
        (y_test, 'Test Set', axes[2])
    ]
    
    for labels, title, ax in datasets:
        unique, counts = np.unique(labels, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_names)))
        
        bars = ax.bar([emotion_names[i] for i in unique], counts, 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Emotion', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} ({len(labels)} samples)', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n✓ Distribution plot saved as 'dataset_distribution.png'")

# Visualize sample images
def visualize_samples(images, labels, emotion_names, n_samples=15):
    """
    Show random sample images with labels
    """
    indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Denormalize for display
        img_display = (images[idx] * 255).astype(np.uint8)
        
        axes[i].imshow(img_display, cmap='gray')
        axes[i].set_title(f"{emotion_names[labels[idx]]}", 
                         fontsize=11, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(indices), 15):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Sample images saved as 'sample_images.png'")

# Save preprocessed data
def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                          emotion_names, output_path='preprocessed_fer2013.npz'):
    """
    Save all preprocessed data
    """
    np.savez_compressed(
        output_path,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        emotion_names=emotion_names
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024**2)
    print(f"\n✓ Preprocessed data saved to '{output_path}'")
    print(f"  File size: {file_size_mb:.2f} MB")

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FER2013 PREPROCESSING - TRAIN/TEST FOLDER STRUCTURE")
    print("="*60)
    
    # IMPORTANT: Set your base path here
    # Option 1: If you have FER2013 folder containing train and test
    base_path = 'FER2013'
    
    # Option 2: If train and test are in current directory
    # base_path = '.'
    
    # Option 3: Full path
    # base_path = '/path/to/your/FER2013'
    
    print(f"\nBase path: {base_path}")
    
    # Check if path exists
    if not os.path.exists(base_path):
        print(f"\n❌ ERROR: Path '{base_path}' does not exist!")
        print("\nPlease update 'base_path' variable with the correct path.")
        print("\nExamples:")
        print("  base_path = 'FER2013'  # if FER2013 folder is in current directory")
        print("  base_path = '.'  # if train and test are in current directory")
        print("  base_path = 'C:/Users/YourName/Desktop/FER2013'  # full path")
        exit()
    
    try:
        # Step 1: Load dataset
        print("\n[1/5] Loading dataset from train and test folders...")
        X_train, X_val, X_test, y_train, y_val, y_test, emotion_names = load_fer2013_dataset(base_path)
        
        # Step 2: Plot distributions
        print("\n[2/5] Plotting class distributions...")
        plot_class_distribution(y_train, y_val, y_test, emotion_names)
        
        # Step 3: Visualize samples
        print("\n[3/5] Visualizing sample images...")
        visualize_samples(X_train, y_train, emotion_names)
        
        # Step 4: Print statistics
        print("\n[4/5] Dataset statistics:")
        print("="*60)
        for i, emotion in enumerate(emotion_names):
            train_count = np.sum(y_train == i)
            val_count = np.sum(y_val == i)
            test_count = np.sum(y_test == i)
            total = train_count + val_count + test_count
            print(f"{emotion:12s}: Train={train_count:5d} | Val={val_count:4d} | Test={test_count:5d} | Total={total:5d}")
        print("="*60)
        
        # Step 5: Save preprocessed data
        print("\n[5/5] Saving preprocessed data...")
        save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, emotion_names)
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE!")
        print("="*60)
        print("\n📦 Generated files:")
        print("  1. preprocessed_fer2013.npz  - Preprocessed dataset")
        print("  2. dataset_distribution.png   - Class distribution plots")
        print("  3. sample_images.png          - Sample visualizations")
        
        print("\n🚀 Next step:")
        print("  Run the CNN training script to train your model!")
        
        print("\n💡 To load data later:")
        print("  data = np.load('preprocessed_fer2013.npz', allow_pickle=True)")
        print("  X_train = data['X_train']")
        print("  y_train = data['y_train']")
        print("  emotion_names = data['emotion_names']")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease check:")
        print("  1. The base_path is correct")
        print("  2. 'train' and 'test' folders exist")
        print("  3. Emotion folders exist inside train and test")
        print("  4. Image files (.jpg) exist in emotion folders")