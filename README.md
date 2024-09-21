<h2> Introduction: </h2>
The aim of the project is to try various <b> deep learning models </b> that are published in <b> research papers. </b> <br/> <br/>

<b> Dataset selection </b> - To evaluate whether a model is able to capture all the nuances in the data, Walmart share prices seemed to be a good option because of the following reasons:
1. Walmart stock prices reflect seasonal retail cycles and macroeconomic factors, offering complex patterns ideal for deep learning time-series models.
2. The stock is influenced by external variables such as consumer trends, inflation, and retail sales, allowing for rich feature exploration in forecasting.
3. Walmart's large historical dataset provides ample training data for deep learning models, helping improve generalization and accuracy in predictions. <br/>

Hence, Walmart stock prices offer complex seasonal patterns, macroeconomic influences, and ample historical data for deep learning time-series forecasting models. This combination makes it an ideal candidate for exploring the effectiveness of various forecasting architectures and their ability to generalize across varying market conditions.

<b> Dataset Source & Feature Engineering: </b> The data was downloaded from Yahoo Finance and the Federal Reserve Bank. However, what separates deep learning models from other ML models or statistical methods is the reduced need for feature engineering. At the same time, DL models also require a lot of data. 
So, to provide more features for the model to identify patterns, several features were provided that offer information about macroeconomic trends such as the Consumer Price Index, unemployment rate, and GDP growth rate.

Also, exogenous variables like oil prices and gold prices were added. Furthermore, binary variables like a day before Black Friday, holidays like Christmas, etc., also have a significant impact on sales and hence share prices. So, they were also taken into consideration.

In the end, since it is stock data, the technical indicator <b> Relative Strength Index (RSI) </b> was also added. The Relative Strength Index (RSI) is a technical indicator used in financial markets to measure the speed and magnitude of price movements. It ranges from 0 to 100 and is used to evaluate overbought or oversold conditions in a stock. A common interpretation is that an RSI value above 70 suggests that a stock is overbought, while a value below 30 indicates it may be oversold. RSI can be useful in stock price prediction as it helps to identify potential reversal points or trends in market behavior. By integrating it into deep learning models, it can provide insights into momentum and market sentiment, which are crucial for time-series forecasting of stock prices.
It is calculated as:

$$
RSI = 100 - \left( \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}} \right)
$$

To find the best window size, the dataset was trained on 7 days, 20 days, 50 days, and 100 days. The aim was to capture the short-term as well as the long-term pattern.   

<h2> Starting Models: </h2> The next task was to try various models.  
<br/>
<b> Model-1: Long Short Term Memory (LSTM) </b> LSTMs are basic comparison points in any time-series forecasting. Generally, they perform best when the data is not very large, which is the case with our dataset. 
So, it was expected to perform well. I compared various window sizes as mentioned above; the best R<sup>2</sup> score was achieved with the 100-day window. Thus, the model is capturing the long-term trend.

<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (LSTM).png" alt="R2-Score LSTM Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (LSTM).png" alt="Test Loss LSTM Models" width="400" height="200">
</div>

<br/> <br/> 
<b> Model-2: </b> GRU (Gated Recurrent Unit) models are simplified versions of LSTMs, designed to capture long-term dependencies in sequential data with fewer parameters and a simpler architecture, making them faster to train. Unlike LSTMs, GRUs merge the forget and input gates into a single update gate and lack an output gate, which reduces computational complexity but might result in slightly less control over memory management compared to LSTMs.

The graph shows that the 50-day window size performs the best among all tested configurations, achieving the highest R-square value, indicating it captures the optimal amount of historical data for accurate predictions. Shorter windows (7 and 20 days) underperform, suggesting insufficient historical context, while the 100-day window, though capturing more data, does not improve further, possibly due to overfitting or increased noise.


<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (GRU).png" alt="R2-Score GRU Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (GRU).png" alt="Test Loss GRU Models" width="400" height="200">
</div>


<b> Model-3: Temporal Convolution Model (TCN) </b>: TCNs are a type of neural network architecture designed for sequence modeling, combining dilated causal convolutions and residual connections to capture long-range dependencies effectively. They allow for parallel processing of sequences, providing a flexible receptive field and efficiently learning temporal patterns without the need for recurrent structures.

Like the other models, I also tried four windows to run the data. Each TCN model is built with stacked temporal blocks that apply dilated convolutions to capture long-range dependencies in the data. The code sets specific hyperparameters. When comparing the performance of this model to RNN variants, the performance is not as good. Similar to the GRU, the 50-day window size works best here as well.

However, compared to RNN variants, I started with a lower learning rate. The learning rate for all window sizes was different, as was the number of epochs. A lower learning rate and fewer epochs help avoid NaN values during the later stages of training and prevent the learning process from stopping. For all cases the value of kerne_size was taken 3.  

<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (TCN).png" alt="R2-Score TCN Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (TCN).png" alt="Test Loss TCN Models" width="400" height="200">
</div>

<b> Model-4: N-Beats </b> The N-Beats model is a powerful deep learning architecture specifically designed for univariate and multivariate time series forecasting. It uses a stack of fully connected neural networks to directly forecast time series data, leveraging backward and forward residual blocks to capture complex patterns, trends, and seasonality without requiring domain-specific feature engineering. N-Beats is known for its flexibility, high performance, and ability to generalize well across diverse time series datasets.

So, to capture trend and seasonality in the model, two different classes were defined. Seasonality is captured using by generating Fourier series components—specifically, sine and cosine terms—for both historical (backcast) and future (forecast) time windows. By decomposing the time series into multiple harmonics, the function allows the model to represent and learn complex cyclical patterns that occur at different frequencies. This approach enables the N-Beats model to effectively identify and forecast seasonal trends, making it well-suited for time series data with regular, repeating patterns, such as those seen in sales, weather, or financial data.

Similarly polynomial trend was captured using generating polynomial bases for both historical (backcast) and future (forecast) periods. By creating polynomial terms of increasing degrees (up to a specified degree) from the time vector, trend component allowed model to learn and represent linear, quadratic, or higher-order trends within the data. These polynomial components enable the N-Beats model to effectively model non-linear trends, making it well-suited for forecasting scenarios where the data exhibits gradual shifts, growth, or decay over time, thus enhancing the model's overall predictive performance.

These are parameters that were used for this model:
<table> 
<tr> <td> Hidden Size </td> <td> 256 </td> </tr>
<tr> <td> Number of blocks (trend, seasonality) </td> <td> 4 </td> </tr>
<tr> <td> Number of Layers </td> <td> 3 </td> </tr>
<tr> <td> Forecast Length </td> <td> 1 </td> </tr>
<tr> <td> Harmonics </td> <td> 10 </td> </tr>
<tr> <td> Polynomial degree for trend block </td> <td> 2 </td> </tr>
</table>  




