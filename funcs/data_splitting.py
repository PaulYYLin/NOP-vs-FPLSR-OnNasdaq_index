import numpy as np


class TimeSeriesDataSplitter:
    def __init__(self, config=None):
        self.config = {
            'seq_length': 90,
            'train_size': 0.8,
            'val_size': 0.1
        }
        if config:
            self.config.update(config)
        
    def create_sequences(self, data_x, data_y, seq_length=None):
        """Create sequences from time series data"""
        if seq_length is None:
            seq_length = self.config['seq_length']
            
        if len(data_x) != len(data_y):
            raise ValueError("Input and target data must have the same length")
        if seq_length <= 0:
            raise ValueError("Sequence length must be positive")
        if len(data_x) <= seq_length:
            raise ValueError("Data length must be greater than sequence length")
        
        # Use numpy operations for better efficiency
        total_sequences = len(data_x) - seq_length
        X = np.zeros((total_sequences, seq_length, data_x.shape[1]))
        y = np.zeros((total_sequences, data_y.shape[1]))
        
        for i in range(total_sequences):
            X[i] = data_x[i:i+seq_length]
            y[i] = data_y[i+seq_length]
        
        return X, y
    
    def split_data(self, X, y):
        """Split data into train, validation, and test sets"""
        total_samples = len(X)
        train_size = int(total_samples * self.config['train_size'])
        val_size = int(train_size * self.config['val_size'])
        
        # Training set
        X_train = X[val_size:train_size]
        y_train = y[val_size:train_size]
        
        # Validation set
        X_val = X[:val_size]
        y_val = y[:val_size]
        
        # Test set
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    def prepare_data(self, data_x, data_y):
        """Main method to prepare and split time series data"""
        # Create sequences
        X, y = self.create_sequences(data_x, data_y)
        
        # Split data
        return self.split_data(X, y)


