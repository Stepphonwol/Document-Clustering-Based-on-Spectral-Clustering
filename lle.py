from sklearn.neighbors import BallTree
import numpy as np

class LLE:
    def __init__(self, X):
        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]

    def find_invariance(self):
        self.K = int(input("Number of nearest neighbors: "))
        # sparse matrix for embedding
        self.W = np.zeros((self.N, self.N))
        # for the ball tree, [N, D]
        tree = BallTree(self.X)
        for i in range(self.N):
            # gram matrix
            bold_x = np.zeros((self.D, self.K))
            neighbors = np.zeros((self.D, self.K))
            unit = np.ones((self.K, 1))
            dist, ind = tree.query([self.X[i]], k=self.K + 1)
            print(i)
            for j in range(self.K):
                bold_x[:,j] = self.X[i]
                neighbors[:,j] = self.X[ind[0][j+1]]
            diff = bold_x - neighbors
            gram_matrix = np.dot(np.transpose(diff), diff)
            pinv_g = np.linalg.pinv(gram_matrix)
            weights = np.dot(pinv_g, unit) / np.dot(np.dot(unit.T, pinv_g), unit)
            print(weights.shape)
            print(weights)
            # one to one
            for j in range(self.K):
                self.W[i][ind[0][j+1]] = weights[j]
        print(self.W[:,1])

    def embedding(self):
        I = np.eye(self.N, dtype=int)
        target = np.dot(I - self.W, np.transpose(I - self.W))
        eigenvalues, eigenvectors = np.linalg.eig(target)
        eig_seq = np.argsort(eigenvalues)
        #discard the first eigenvalue
        eig_seq_indice_2d = eig_seq[1:3]
        eig_seq_indice_3d = eig_seq[1:4]
        new_eig_vec_2d = eigenvectors[:,eig_seq_indice_2d]
        new_eig_vec_3d = eigenvectors[:,eig_seq_indice_3d]
        return new_eig_vec_2d, new_eig_vec_3d

    def analyze(self):
        self.find_invariance()
        return self.embedding()