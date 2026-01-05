import matplotlib.pyplot as plt

# K values (number of eigenfaces)
k_values = [5, 10, 15, 20, 25, 30]

# Corresponding accuracy values (example from experiments)
accuracy = [65, 72, 78, 85, 83, 80]

# Plot graph
plt.figure()
plt.plot(k_values, accuracy, marker='o')
plt.xlabel("Number of Eigenfaces (K)")
plt.ylabel("Recognition Accuracy (%)")
plt.title("Accuracy vs K Value")
plt.grid(True)

# Save BEFORE show
plt.savefig("output_accuracy_vs_k.png")
plt.show()
