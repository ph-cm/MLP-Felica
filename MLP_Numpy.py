import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        # Inicialização dos pesos e biases
        for i in range(len(self.layer_sizes)-1):
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
            self.weights.append(np.random.uniform(-limit, limit, 
                                (self.layer_sizes[i], self.layer_sizes[i+1])))
            self.biases.append(np.zeros((1, self.layer_sizes[i+1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        # Camadas ocultas
        for i in range(len(self.weights)-1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            self.activations.append(self.relu(z))
        
        # Camada de saída (softmax)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        self.activations.append(self.softmax(z))
        
        return self.activations[-1]
    
    def compute_loss(self, y_true, y_pred):
        # Cross-entropy loss
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X, y_true, learning_rate):
        m = X.shape[0]
        gradients = []
        
        # Gradiente da camada de saída
        dZ = self.activations[-1] - y_true
        dW = np.dot(self.activations[-2].T, dZ) / m
        db = np.sum(dZ, axis=0, keepdims=True) / m
        gradients.insert(0, (dW, db))
        
        # Backpropagation pelas camadas ocultas
        for i in range(len(self.weights)-2, -1, -1):
            dA = np.dot(dZ, self.weights[i+1].T)
            dZ = dA * self.relu_derivative(self.z_values[i])
            dW = np.dot(self.activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            gradients.insert(0, (dW, db))
        
        # Atualização dos pesos e biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32, verbose=True):
        for epoch in range(epochs):
            # Mini-batch training
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward e backward pass
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            # Cálculo da loss e acurácia
            if verbose and epoch % 100 == 0:
                y_pred = self.forward(X)
                loss = self.compute_loss(y, y_pred)
                accuracy = np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                
                

# Gerar dados de exemplo
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_informative=10, random_state=42)
y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os dados
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Criar e treinar a MLP
mlp = MLP(input_size=20, hidden_sizes=[64, 32], output_size=3)
mlp.train(X_train, y_train, epochs=1000, learning_rate=0.01, batch_size=32)

# Avaliar no conjunto de teste
y_pred = mlp.forward(X_test)
test_accuracy = np.mean(np.argmax(y_test, axis=1) == np.argmax(y_pred, axis=1))
print(f"\nTest Accuracy: {test_accuracy:.4f}")
