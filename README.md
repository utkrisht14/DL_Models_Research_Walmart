<h2> Introduction: </h2>
The aim of the project is to try various <b> deep learning models </b> that are published in the <b> research paper. </b> <br/> <br/>

<b> Dataset selection </b>- To evaluate whether a model is able to capture all the nuances in the data, Walmart share prices seemed to be a good option because of thefollowing reasons:
1. Walmart stock prices reflect seasonal retail cycles and macroeconomic factors, offering complex patterns ideal for deep learning time-series models.
2. The stock is influenced by external variables such as consumer trends, inflation, and retail sales, allowing for rich feature exploration in forecasting.
3. Walmart's large historical dataset provides ample training data for deep learning models, helping improve generalization and accuracy in predictions. <br/>

Hence Walmart stock prices offer complex seasonal patterns, macroeconomic influences, and ample historical data for deep learning time-series forecasting models.This combination makes it an ideal candidate for exploring the effectiveness of various forecasting architectures and their ability to generalize across varying market conditions.

<b> Dataset Source & Feature Engineering: </b> The data was downloaded from the Yahoo finance and Federal Reserve Bank. Though what separates deep learning model fromm other models ML models or statistical method, is no need for feature engineering. But at the same time DL models also required a lot of data. 
So to provide more features for model to find out the pattern, several features were provided which would be helpful to provide the information about the macroeconomic trend like Consumer Price Index, unemployment rate, gdp growth rate    

Also exogenous variables like oil prices and gold prices were also added. Furthermore, binary variables like a day before Black Friday, Holdiays like Christmas etc. also make a huge impact on the sales and hence share prices. So, they were also taken into consideration.

In the end since it is a stock data, a technical indicator <b> Relative Strength Index (RSI) </b> was also added. The Relative Strength Index (RSI) is a technical indicator used in financial markets to measure the speed and magnitude of price movements. It ranges from 0 to 100 and is used to evaluate overbought or oversold conditions in a stock.
A common interpretation is that an RSI value above 70 suggests that a stock is overbought, while a value below 30 indicates it may be oversold.RSI can be useful in stock price prediction as it helps to identify potential reversal points or trends in market behavior. By integrating it into deep learning models, it can provide insights into momentum and market sentiment, which are crucial for time-series forecasting of stock prices.
It is calculated as:

$$
RSI = 100 - \left( \frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}} \right)
$$

To find the best window size, the dataset was trained on 7 days, 20 days, 50 days and 100 days. The aim was to capture the short-term as well as the long-term pattern.   

<h2> Starting Models: </h2> Next task was to try various models.  
<br/>
<b> Model-1: Long Short Term Memory (LSTM) </b> LSTMs are basic comparison points in any time-series forecasting. Gennerally they perform best when the data is not very large and so is the case with ourdataset. 
So, it was expected to perform good. I tried to compare against many window size as mentioned above, the best R<sup>2</sup>score was given by 100 days window. So, model is capturing long-tern trend. <br/>
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
