import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from thundersvm import SVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Train SVM with labeled images.')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output model. (required)')
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                        help='Path to the input directory containing labeled subdirectories. (required)')
    parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.9,
                        help='How many samples to choose for training. (default: 0.9)')
    parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                        help='Ratio with which to split training and validation sets. (default: 0.7)')

    return parser.parse_args()

def load_images_from_folder(folder):
    """Loads images from a folder and returns the image data and labels."""
    images = []
    labels = []
    print(os.listdir(folder))
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in tqdm(os.listdir(label_folder), desc=f'Loading {label_folder}', unit='image'):
                if filename.endswith(".jpg"):
                    img = Image.open(os.path.join(label_folder, filename))
                    if img.width != 16 or img.height != 16:
                        print("size error")
                        exit(0)
                    #img = img.resize((16, 16))  # Resize the image to 16x16
                    img_data = np.array(img)
                    images.append(img_data.flatten())  # Flatten the image into a 1D vector
                    labels.append(int(label))
    return np.array(images), np.array(labels)

def evaluate_model(model, X, y):
    """Evaluates the SVM model on the validation set and returns precision, recall, and F1 score."""
    y_pred = model.predict(X)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return precision, recall, f1

def main():
    """Main function."""
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise ValueError(f'Input directory `{args.input_dir}` does not exist!')
    print("loading the images")
    # Load images and labels from the input directory
    images, labels = load_images_from_folder(args.input_dir)

    if len(images) > 50000:
        indices = np.random.choice(len(images), size=5000, replace=False)
        images = images[indices]
        labels = labels[indices]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=1 - args.chosen_num_or_ratio, random_state=42)

    # Create an SVM classifier
    svm_classifier = SVC(kernel='linear', verbose=True)

    # Train the classifier on the training set
    svm_classifier.fit(X_train, y_train)

    # Evaluate the model on the validation set
    precision, recall, f1 = evaluate_model(svm_classifier, X_val, y_val)
    accuracy = svm_classifier.score(X_val, y_val)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)

    # Save the trained model
    weights = svm_classifier.coef_
    print(f"weight: {weights.shape}")
    weights = weights / np.linalg.norm(weights)
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'boundary.npy'), weights)

if __name__ == '__main__':
    main()