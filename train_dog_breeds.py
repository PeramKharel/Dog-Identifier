import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import datetime

class DogBreedTrainer:
    def __init__(self, data_dir='data', img_size=224, batch_size=32):
        """
        Initialize the trainer
        Args:
            data_dir: path to folder with train/validation/test subfolders
            img_size: input image size (224 is standard for MobileNetV2)
            batch_size: number of images per batch
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = None
        self.class_names = None
        self.model = None
        self.history = None
        
        # Create directories for saving models and plots
        os.makedirs('saved_models', exist_ok=True)
        os.makedirs('training_plots', exist_ok=True)
        
    def create_data_generators(self):
        """
        Create data generators with augmentation for training
        """
        print("\n" + "="*50)
        print("STEP 1: CREATING DATA GENERATORS")
        print("="*50)
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Normalize pixel values to [0,1]
            rotation_range=30,            # Random rotation up to 30 degrees
            width_shift_range=0.2,        # Random horizontal shift
            height_shift_range=0.2,       # Random vertical shift
            shear_range=0.2,              # Shear transformations
            zoom_range=0.2,                # Random zoom
            horizontal_flip=True,          # Random horizontal flip
            brightness_range=[0.8, 1.2],   # Random brightness adjustment
            fill_mode='nearest'             # Fill missing pixels
        )
        
        # Validation and test generators (only rescaling, no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'train'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42  # For reproducibility
        )
        
        self.validation_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'validation'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        # Get class information
        self.num_classes = len(self.train_generator.class_indices)
        self.class_names = list(self.train_generator.class_indices.keys())
        
        print(f"\n📊 Dataset Summary:")
        print(f"  • Number of classes: {self.num_classes}")
        print(f"  • Training samples: {self.train_generator.samples}")
        print(f"  • Validation samples: {self.validation_generator.samples}")
        print(f"  • Test samples: {self.test_generator.samples}")
        print(f"\n  • Classes: {self.class_names[:5]}...")  # Show first 5
        
        # Calculate steps per epoch
        self.steps_per_epoch = self.train_generator.samples // self.batch_size
        self.validation_steps = self.validation_generator.samples // self.batch_size
        
        return True
    
    def build_model(self):
        """
        Build model using transfer learning with MobileNetV2
        This is industry standard for image classification with limited data
        """
        print("\n" + "="*50)
        print("STEP 2: BUILDING THE MODEL")
        print("="*50)
        
        # Load pre-trained MobileNetV2 (trained on ImageNet)
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,           # Don't include the classification layer
            weights='imagenet'            # Use pre-trained weights
        )
        
        # Freeze the base model layers (they won't be trained initially)
        base_model.trainable = False
        
        # Build the complete model
        self.model = keras.Sequential([
            # Pre-trained base
            base_model,
            
            # Global average pooling (reduces overfitting)
            layers.GlobalAveragePooling2D(),
            
            # Dropout for regularization
            layers.Dropout(0.3),
            
            # Dense layer for learning breed-specific features
            layers.Dense(512, activation='relu'),
            
            # Another dropout
            layers.Dropout(0.3),
            
            # Output layer with softmax for multi-class classification
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
        
        return self.model
    
    def train_model(self, epochs=50, fine_tune_epochs=30):
        """
        Train the model in two phases:
        1. Train only the new top layers
        2. Fine-tune some of the base model layers
        """
        print("\n" + "="*50)
        print("STEP 3: TRAINING THE MODEL")
        print("="*50)
        
        # Callbacks for better training
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                'saved_models/best_model_phase1.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # TensorBoard for visualization
            keras.callbacks.TensorBoard(
                log_dir=f'logs/fit/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1
            )
        ]
        
        # PHASE 1: Train only the new layers
        print("\n🔰 PHASE 1: Training top layers...")
        history_phase1 = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # PHASE 2: Fine-tuning
        print("\n🔧 PHASE 2: Fine-tuning the model...")
        
        # Unfreeze the base model
        self.model.layers[0].trainable = True
        
        # Freeze early layers (keep them as feature extractors)
        # Typically freeze first 100 layers of MobileNetV2
        for layer in self.model.layers[0].layers[:100]:
            layer.trainable = False
            
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 10x lower
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Update callbacks for phase 2
        callbacks_phase2 = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.000001),
            ModelCheckpoint('saved_models/best_model_phase2.h5', monitor='val_accuracy', 
                           save_best_only=True)
        ]
        
        # Continue training
        history_phase2 = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.steps_per_epoch,
            epochs=fine_tune_epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_steps,
            callbacks=callbacks_phase2,
            verbose=1
        )
        
        # Combine histories
        self.history = {}
        for key in history_phase1.history.keys():
            self.history[key] = history_phase1.history[key] + history_phase2.history[key]
        
        return self.history
    
    def evaluate_model(self):
        """
        Evaluate the model on test data
        """
        print("\n" + "="*50)
        print("STEP 4: EVALUATING THE MODEL")
        print("="*50)
        
        # Load the best model from phase 2
        self.model = keras.models.load_model('saved_models/best_model_phase2.h5')
        
        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(
            self.test_generator,
            steps=self.test_generator.samples // self.batch_size,
            verbose=1
        )
        
        print(f"\n📊 Test Results:")
        print(f"  • Test Loss: {test_loss:.4f}")
        print(f"  • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        return test_loss, test_accuracy
    
    def plot_training_history(self):
        """
        Plot training history
        """
        print("\n" + "="*50)
        print("STEP 5: PLOTTING TRAINING HISTORY")
        print("="*50)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.axvline(x=len(self.history['accuracy'])-30, color='red', linestyle='--', 
                   label='Fine-tuning starts', alpha=0.7)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.axvline(x=len(self.history['loss'])-30, color='red', linestyle='--', 
                   label='Fine-tuning starts', alpha=0.7)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_plots/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✅ Training history plot saved to 'training_plots/training_history.png'")
    
    def save_model_final(self):
        """
        Save the final model and class names
        """
        # Save model
        model_path = 'saved_models/final_dog_breed_model.h5'
        self.model.save(model_path)
        print(f"\n💾 Model saved to: {model_path}")
        
        # Save class names
        classes_path = 'saved_models/class_names.txt'
        with open(classes_path, 'w') as f:
            for class_name in self.class_names:
                f.write(class_name + '\n')
        print(f"💾 Class names saved to: {classes_path}")
        
        # Also save in Keras format for newer versions
        self.model.save('saved_models/final_dog_breed_model.keras')
        print(f"💾 Model also saved in Keras v3 format")
    
    def predict_single_image(self, image_path):
        """
        Predict breed for a single image
        """
        from PIL import Image
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions)[-3:][::-1]
        
        results = []
        for idx in top_3_idx:
            breed = self.class_names[idx]
            confidence = predictions[idx]
            results.append((breed, confidence))
        
        return results, img

def main():
    """
    Main training function
    """
    print("\n" + "="*60)
    print("🐕 DOG BREED CLASSIFIER - TRAINING SCRIPT")
    print("="*60)
    
    # Initialize trainer
    trainer = DogBreedTrainer(
        data_dir='data',
        img_size=224,
        batch_size=32  # Adjust based on your GPU memory
    )
    
    # Step 1: Create data generators
    trainer.create_data_generators()
    
    # Step 2: Build model
    trainer.build_model()
    
    # Step 3: Train model
    print("\n🎯 Ready to train!")
    epochs = int(input("Enter number of epochs for phase 1 (recommended: 30-50): ") or "40")
    fine_tune_epochs = int(input("Enter number of epochs for fine-tuning (recommended: 20-30): ") or "25")
    
    trainer.train_model(epochs=epochs, fine_tune_epochs=fine_tune_epochs)
    
    # Step 4: Evaluate
    trainer.evaluate_model()
    
    # Step 5: Plot results
    trainer.plot_training_history()
    
    # Step 6: Save final model
    trainer.save_model_final()
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE! Your dog breed classifier is ready!")
    print("="*60)
    
    # Test with a new image
    while True:
        print("\n" + "-"*40)
        print("TEST THE MODEL")
        print("-"*40)
        print("1. Test with an image")
        print("2. Exit")
        
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == '1':
            image_path = input("Enter path to test image: ").strip()
            if os.path.exists(image_path):
                results, img = trainer.predict_single_image(image_path)
                
                # Display results
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Show image
                ax1.imshow(img)
                ax1.set_title("Input Image", fontsize=14, fontweight='bold')
                ax1.axis('off')
                
                # Show predictions bar chart
                breeds = [r[0].replace('_', ' ').title() for r in results]
                confidences = [r[1] for r in results]
                y_pos = np.arange(len(breeds))
                
                bars = ax2.barh(y_pos, confidences, color=['#2ecc71', '#f39c12', '#e74c3c'])
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(breeds)
                ax2.set_xlabel('Confidence', fontsize=12)
                ax2.set_title('Top 3 Predictions', fontsize=14, fontweight='bold')
                
                # Add confidence percentages on bars
                for i, (bar, conf) in enumerate(zip(bars, confidences)):
                    ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{conf*100:.1f}%', va='center')
                
                plt.tight_layout()
                plt.show()
                
                print(f"\n📊 Top Prediction: {breeds[0]} ({confidences[0]*100:.2f}%)")
            else:
                print("❌ File not found!")
        
        elif choice == '2':
            print("\n👋 Thank you for using Dog Breed Classifier!")
            break

if __name__ == "__main__":
    main()