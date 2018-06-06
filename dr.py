import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DR:
    def __init__(self, X, K):
        self.X = np.array(X)
        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        self.K = K

    def construct_similarity_graph(self):
        # cosine similarity
        self.W = cosine_similarity(self.X)
        print(self.W.shape)
        for i in range(self.N):
            for j in range(self.N):
                if self.W[i][j] < 0:
                    self.W[i][j] = 0
        print(self.W)

    def construct_degree_matrix(self):
        self.D = np.zeros((self.N, self.N))
        for i in range(self.N):
            self.D[i][i] = np.sum(self.W[i])
        print(self.D)

    def construct_laplacian_matrix(self):
        self.L = self.D - self.W

    def eigenmap(self):
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.pinv(self.D), self.L))
        print(eigenvectors.shape)
        print(eigenvectors)
        eig_seq = np.argsort(eigenvalues)
        print(eigenvalues[eig_seq])
        eig_seq_indice = eig_seq[0:self.K]
        new_eig_vec = eigenvectors[:,eig_seq_indice]
        print(new_eig_vec.shape)
        return new_eig_vec

    def analyze(self):
        self.construct_similarity_graph()
        self.construct_degree_matrix()
        self.construct_laplacian_matrix()
        return self.eigenmap()