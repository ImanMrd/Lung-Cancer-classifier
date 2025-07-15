import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class LungCancerClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = 3
        self.class_names = ['normal', 'benign', 'malignant']
        self.model = None
        self.history = None
        
    def load_and_preprocess_data(self, data_dir, split):
        """
        Load and preprocess images from directory structure:
        data_dir/
        ├── train/
        │   ├── normal/
        │   ├── benign/
        │   └── malignant/
        ├── validation/
        │   ├── normal/
        │   ├── benign/
        │   └── malignant/
        └── test/
            ├── normal/
            ├── benign/
            └── malignant/
        """
        images = []
        labels = []
        
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Directory {split_dir} does not exist!")
            return np.array([]), np.array([])
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist!")
                continue
                
            file_count = 0
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(class_idx)
                        file_count += 1
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
            
            print(f"Loaded {file_count} images from {class_name} class in {split} set")
        
        return np.array(images), np.array(labels)
    
    def create_data_generators_from_directory(self, data_dir):
        """Create data generators directly from directory structure"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation and test
        val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(data_dir, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=True
        )
        
        val_generator = val_test_datagen.flow_from_directory(
            os.path.join(data_dir, 'validation'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        test_generator = val_test_datagen.flow_from_directory(
            os.path.join(data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            classes=self.class_names,
            shuffle=False
        )
        
        return train_generator, val_generator, test_generator
    
    def build_model(self, use_pretrained=True):
        """Build transfer learning model using ResNet50"""
        try:
            if use_pretrained:
                print("Attempting to load pre-trained ResNet50 weights...")
                # Load pre-trained ResNet50 without top layers
                base_model = ResNet50(
                    weights='imagenet',
                    include_top=False,
                    input_shape=(*self.img_size, 3)
                )
                print("✓ Pre-trained weights loaded successfully!")
            else:
                print("Loading ResNet50 without pre-trained weights...")
                base_model = ResNet50(
                    weights=None,
                    include_top=False,
                    input_shape=(*self.img_size, 3)
                )
                print("✓ ResNet50 architecture loaded (no pre-trained weights)")
        except Exception as e:
            print(f"✗ Failed to load pre-trained weights: {e}")
            print("Falling back to ResNet50 without pre-trained weights...")
            base_model = ResNet50(
                weights=None,
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            print("✓ ResNet50 architecture loaded (no pre-trained weights)")
        
        # Freeze base model layers initially (only if pre-trained)
        if use_pretrained:
            base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, data_dir, epochs=50):
        """Train the model using directory structure"""
        # Create data generators
        train_gen, val_gen, test_gen = self.create_data_generators_from_directory(data_dir)
        
        # Store test generator for later evaluation
        self.test_generator = test_gen
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
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
            ModelCheckpoint(
                'best_lung_cancer_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def fine_tune_model(self, data_dir, epochs=20):
        """Fine-tune the model by unfreezing some layers"""
        # Unfreeze the last few layers of the base model
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create data generators
        train_gen, val_gen, test_gen = self.create_data_generators_from_directory(data_dir)
        
        # Define callbacks for fine-tuning
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return fine_tune_history
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot training & validation loss
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, test_generator=None):
        """Evaluate the model using test generator"""
        if test_generator is None:
            test_generator = self.test_generator
        
        # Reset generator
        test_generator.reset()
        
        # Make predictions
        predictions = self.model.predict(test_generator, verbose=1)
        y_pred_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        y_true = test_generator.classes
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_classes == y_true)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        
        return predictions, y_pred_classes
    
    def predict_single_image(self, image_path):
        """Predict a single image"""
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        prediction = self.model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return self.class_names[predicted_class], confidence
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def build_alternative_model(self):
        """Build a custom CNN model as an alternative to ResNet50"""
        print("Building custom CNN model...")
        
        model = tf.keras.Sequential([
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fourth convolutional block
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            # Fifth convolutional block
            tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.5),
            
            # Dense layers
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("✓ Custom CNN model built successfully!")
        return model

# Main execution
def main():
    # Initialize classifier
    classifier = LungCancerClassifier(img_size=(224, 224), batch_size=32)
    
    # Set your data directory path
    data_dir = "data"  # Update this path to your data directory
    
    print("Data directory structure:")
    print(f"{data_dir}/")
    print("├── train/")
    print("│   ├── normal/")
    print("│   ├── benign/")
    print("│   └── malignant/")
    print("├── validation/")
    print("│   ├── normal/")
    print("│   ├── benign/")
    print("│   └── malignant/")
    print("└── test/")
    print("    ├── normal/")
    print("    ├── benign/")
    print("    └── malignant/")
    print()
    
    # Load and display dataset information
    print("Loading dataset information...")
    for split in ['train', 'validation', 'test']:
        X, y = classifier.load_and_preprocess_data(data_dir, split)
        print(f"{split.capitalize()} set: {len(X)} total images")
        if len(y) > 0:
            for i, class_name in enumerate(classifier.class_names):
                count = np.sum(y == i)
                print(f"  {class_name}: {count} images")
        print()
    
    # Build model with fallback options
    print("Building model...")
    try:
        model = classifier.build_model(use_pretrained=True)
    except Exception as e:
        print(f"Failed to build model with pre-trained weights: {e}")
        print("Trying alternative approaches...")
        
        # Option 1: Try manual download
        if classifier.download_weights_manually():
            try:
                model = classifier.build_model(use_pretrained=True)
            except:
                print("Manual download didn't work, using custom model...")
                model = classifier.build_alternative_model()
        else:
            # Option 2: Use custom CNN model
            print("Using custom CNN model instead of ResNet50...")
            model = classifier.build_alternative_model()
    
    print(f"Model built with {model.count_params()} parameters")
    print()
    
    # Train model
    print("Training model...")
    history = classifier.train_model(data_dir, epochs=50)
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Fine-tune model
    print("Fine-tuning model...")
    fine_tune_history = classifier.fine_tune_model(data_dir, epochs=20)
    
    # Evaluate model
    print("Evaluating model...")
    predictions, y_pred_classes = classifier.evaluate_model()
    
    # Save model
    classifier.save_model("lung_cancer_classifier_final.h5")
    
    # Example prediction for a single image
    # predicted_class, confidence = classifier.predict_single_image("path/to/test/image.jpg")
    # print(f"Predicted: {predicted_class} (Confidence: {confidence:.4f})")

if __name__ == "__main__":
    main()