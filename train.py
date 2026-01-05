import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pca import compute_pca
from ann import train_ann

# --------------------
# Dataset Path
# --------------------
DATASET = "dataset/faces"
IMG_SIZE = (100, 100)
K = 20

X = []
y = []
label = 0

# --------------------
# Load Images
# --------------------
for person in os.listdir(DATASET):
    person_path = os.path.join(DATASET, person)
    if not os.path.isdir(person_path):
        continue

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, 0)

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        X.append(img.flatten())
        y.append(label)

    label += 1

X = np.array(X).T
y = np.array(y)

print("Total images:", X.shape[1])
print("Total persons:", label)

# --------------------
# PCA
# --------------------
mean_face, eigenfaces = compute_pca(X, K)
features = np.dot(eigenfaces.T, X - mean_face).T

# --------------------
# Train-Test Split
# --------------------
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.4, random_state=42
)

# --------------------
# ANN Training
# --------------------
model = train_ann(X_train, y_train)

# --------------------
# Testing
# --------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy * 100, "%")

#Mean Face Image
import matplotlib.pyplot as plt

mean_img = mean_face.reshape(100, 100)

plt.imshow(mean_img, cmap='gray')
plt.title("Mean Face")
plt.axis('off')
plt.savefig("output_mean_face.png")
plt.show()

#Eigenfaces
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    eigen_img = eigenfaces[:, i].reshape(100, 100)
    ax.imshow(eigen_img, cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')

plt.tight_layout()
plt.savefig("output_eigenfaces.png")
plt.show()
