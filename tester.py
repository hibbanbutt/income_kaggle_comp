from scipy.sparse import hstack
import numpy as np

# Create two simple matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5], [6]])

# Horizontally stack them
combined_matrix = hstack((matrix1, matrix2))

print(combined_matrix.toarray())
