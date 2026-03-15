import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class DogBreedPredictor:
    def __init__(self, model_path='saved_models/final_dog_breed_model.keras', 
                 classes_path='saved_models/class_names.txt'):
        """
        Load the saved model and class names
        """
        print("🐕 Loading dog breed classifier...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            # Try .h5 format if .keras doesn't exist
            model_path = model_path.replace('.keras', '.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
        # Load class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(self.class_names)} breed classes")
        
        self.img_size = 224  # Must match training size
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        """
        try:
            # Load and convert to RGB
            img = Image.open(image_path).convert('RGB')
            
            # Save original for display
            original = img.copy()
            
            # Resize for model
            img = img.resize((self.img_size, self.img_size))
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array, original
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None
    
    def predict(self, image_path, top_k=5):
        """
        Predict breed for an image
        """
        # Preprocess image
        img_array, original = self.preprocess_image(image_path)
        
        if img_array is None:
            return None, None
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)[0]
        
        # Get top k predictions
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            breed = self.class_names[idx].replace('_', ' ').title()
            confidence = predictions[idx] * 100
            results.append((breed, confidence))
        
        return results, original
    
    def display_prediction(self, image_path, results, original):
        """
        Display the image with predictions
        """
        if results is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        ax1.imshow(original)
        ax1.set_title("Input Image", fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Show predictions as bar chart
        breeds = [r[0] for r in results]
        confidences = [r[1] for r in results]
        y_pos = np.arange(len(breeds))
        
        # Color bars based on confidence
        colors = ['#27ae60' if c > 70 else '#f39c12' if c > 40 else '#e74c3c' 
                 for c in confidences]
        
        bars = ax2.barh(y_pos, confidences, color=colors)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(breeds)
        ax2.set_xlabel('Confidence (%)', fontsize=12)
        ax2.set_title('Top Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add confidence values on bars
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{conf:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print("\n📊 PREDICTION RESULTS:")
        print("-" * 40)
        for i, (breed, conf) in enumerate(results, 1):
            emoji = "🎯" if i == 1 else "   "
            print(f"{emoji} {i}. {breed}: {conf:.2f}%")
    
    def batch_predict(self, folder_path, top_k=3):
        """
        Predict all images in a folder
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                images.append(os.path.join(folder_path, file))
        
        if not images:
            print("No images found in folder!")
            return
        
        print(f"\n📁 Found {len(images)} images. Processing...")
        
        results = {}
        for img_path in images:
            preds, _ = self.predict(img_path, top_k=1)  # Just get top prediction
            if preds:
                results[os.path.basename(img_path)] = preds[0]
                print(f"  • {os.path.basename(img_path)}: {preds[0][0]} ({preds[0][1]:.1f}%)")
        
        return results

def main():
    """
    Main function to use the model
    """
    # Initialize predictor
    try:
        predictor = DogBreedPredictor()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nPlease make sure you have trained and saved a model first.")
        print("Run 'train_dog_breeds.py' to train a model.")
        return
    
    while True:
        print("\n" + "="*50)
        print("🐕 DOG BREED IDENTIFIER")
        print("="*50)
        print("1. Predict single image")
        print("2. Batch predict folder")
        print("3. Show model info")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Single image prediction
            image_path = input("Enter path to image: ").strip()
            
            if not os.path.exists(image_path):
                print("❌ File not found!")
                continue
            
            results, original = predictor.predict(image_path, top_k=5)
            predictor.display_prediction(image_path, results, original)
        
        elif choice == '2':
            # Batch prediction
            folder_path = input("Enter folder path: ").strip()
            
            if not os.path.exists(folder_path):
                print("❌ Folder not found!")
                continue
            
            predictor.batch_predict(folder_path)
        
        elif choice == '3':
            # Show model info
            print("\n📋 MODEL INFORMATION:")
            print("-" * 30)
            print(f"Number of breeds: {len(predictor.class_names)}")
            print(f"Input size: {predictor.img_size}x{predictor.img_size}")
            print(f"Model type: {type(predictor.model).__name__}")
            
            # Show sample breeds
            print("\nSample breeds (first 10):")
            for i, breed in enumerate(predictor.class_names[:10], 1):
                print(f"  {i}. {breed.replace('_', ' ').title()}")
        
        elif choice == '4':
            print("\n👋 Thank you for using Dog Breed Identifier!")
            break
        
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()