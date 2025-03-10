# NOP-vs-FPLSR-OnNasdaq_index
This analysis compares the Nonlinear Prediction of Functional Time-Series Regression (NOP) Model and the Functional Partial Least Squares Regression (FPLSR) Model in predicting the Nasdaq Index closing prices over time.




### **PART 1: Data Collection & Scaling**

Stock list:
        tickers: ["META", "AAPL", "AMZN", "NFLX", "GOOGL", "^IXIC"] \
        start_date: "2015-01-01" \
        end_date: "2024-12-31"
    
We select stock price, trading volume, RSI, MACD, ATR, SMA, EMA as the indicators to predict the Nasdaq Index closing prices.


| Indicator Type            | Standardization Method |
|---------------------------|--------------------------|
| **Stock Price** (Open, High, Low, Close, Adj Close)  | MinMaxScaler|
| **Trading Volume (Volume)** | log1p (log transformation) |
| **Relative Strength Index (RSI, 0~100)**  | MinMaxScaler (scaled to 0~1) |
| **Moving Average Convergence Divergence (MACD)**  | StandardScaler (mean=0, std=1) |
| **Average True Range (ATR, Volatility)**  | MinMaxScaler |
| **Simple & Exponential Moving Averages (SMA, EMA)**  | MinMaxScaler (scaled to 0~1) |
| **`Nasdaq Index Closing Price`** | RobustScaler (scale to median and IQR) |

The Nasdq index is increasing over time, so we use RobustScaler to scale the target value to avoid the value which is never reached. If we use MinMaxScaler, the value will be too small and the model will not be able to predict the value. If we use log1p, the value is too smooth and the model will not be able to predict the value.


### **PART2: Model Training**

By referring to the paper 
*Wang, H., & Cao, J. (2023). Nonlinear prediction of functional time series. Environmetrics, 34(5), e2792. https://doi.org/10.1002/env.2792*, we apply LSTM to be our encoder to decrease the dimension of the data into latent space (5~20). But we only predict the Nasdaq Index closing prices, so we remove the decoder part. We also directly use the LSTM model to compare the performance of these three models.

In deep learning models structure, we apply huber loss function instead of mean squared error loss function to reduce the influence of outliers and also avoid the model over trained to become average as the result is too smooth.

NOP Model Structure:
<img width="669" alt="image" src="https://github.com/user-attachments/assets/be5ec90f-17f4-4a81-a887-04eec2cd8cb1" />


### **PART3: Model Evaluation**

The evaluation metrics are MAE, MSE, RMSE, R-squared, and MAPE.
We found that Directly using LSTM model to predict the Nasdaq Index closing prices is better than using NOP and FPLSR model.
However NOP model is the worst model among the three models. The reason is that maybe the hidden layer is too strong, so the model is overfitting the non-linear relationship between the features and the target from training data. After doing Dropout and Regularization, the result still remain. Or the LSTM encoder is not able to capture the non-linear relationship between the features and the target.
![image](https://github.com/user-attachments/assets/9562e898-7f64-4e11-b8a2-8478e6feecea)

![image](https://github.com/user-attachments/assets/cd7721f3-454f-4dfc-bc24-078cc5026f97)



### **Setup**

* Step1: Clone the repository:
```bash
git clone https://github.com/paullin0928/NOP-vs-FPLSR-OnNasdaq_index.git
```
* Step2: Create a virtual environment:
```bash
python3.11 -m venv venv
```
* Step3: Activate the virtual environment:
```bash
source venv/bin/activate
```
* Step4: Install the dependencies:
```bash
pip install -r requirements.txt
```
