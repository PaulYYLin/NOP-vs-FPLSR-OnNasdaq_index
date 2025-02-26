import pandas as pd


# Dictionary mapping feature types to their scaling methods

class FeaturesScaler:
    def __init__(self, SCALING_METHODS):
        self.SCALING_METHODS = SCALING_METHODS

    def scale_target(self, df=None, inv_data = None, inverse=False):
        target_config = self.SCALING_METHODS['target']
        if inverse and inv_data is not None:
            return target_config['scaler'].inverse_transform(inv_data)
        else:
            df = df.copy()
            target_col = target_config['cols'][0]  # Get the target column name from config
            df[target_col] = target_config['scaler'].fit_transform(df[target_col].values.reshape(-1, 1))
            return df[target_col]

    def scale_features(self, df):
        """
        Scale all features according to their defined scaling methods
        """
        df = df.copy()  # Prevent modifications to original dataframe
        
        # Scale price features
        price_config = self.SCALING_METHODS['price']
        price_cols = [col for col in df.columns 
                    if any(p in col for p in price_config['pattern'])
                    and col not in price_config['exclude']]
        if price_cols:
            df[price_cols] = price_config['scaler'].fit_transform(df[price_cols])
            
            # Scale volume features
            volume_config = self.SCALING_METHODS['volume']
            volume_cols = [col for col in df.columns if any(p in col for p in volume_config['pattern'])]
            if volume_cols:
                df[volume_cols] = volume_config['scaler'].fit_transform(df[volume_cols])
            
            # Scale technical indicators
            for indicator, config in self.SCALING_METHODS['indicators'].items():
                cols = [col for col in df.columns if any(p in col for p in config['pattern'])]
                if cols:
                    df[cols] = config['scaler'].fit_transform(df[cols])
        
        return df
