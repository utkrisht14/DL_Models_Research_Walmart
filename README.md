<h2> Introduction: </h2>
The aim of the project is to try various <b> deep learning models </b> that are published in <b> research papers and see which performs best on financial dataset. </b> <br/> <br/>

<b> Dataset selection </b> - To evaluate whether a model is able to capture all the nuances in the data, Walmart share prices seemed to be a good option because of the following reasons:
1. Walmart stock prices reflect seasonal retail cycles and macroeconomic factors, offering complex patterns ideal for deep learning time-series models.
2. The stock is influenced by external variables such as consumer trends, inflation, and retail sales, allowing for rich feature exploration in forecasting.
3. Walmart's large historical dataset provides ample training data for deep learning models, helping improve generalization and accuracy in predictions. <br/>

Hence, Walmart stock prices offer complex seasonal patterns, macroeconomic influences, and ample historical data for deep learning time-series forecasting models. This combination makes it an ideal candidate for exploring the effectiveness of various forecasting architectures and their ability to generalize across varying market conditions.

<b> Dataset Source & Feature Engineering: </b> The data was downloaded from Yahoo Finance and the Federal Reserve Bank. What separates deep learning models from other ML models or statistical methods is the reduced need for feature engineering. At the same time, DL models also require a lot of data. 
So, to provide more features for the model to identify patterns, several features were provided that offer information about macroeconomic trends such as the Consumer Price Index, unemployment rate, and GDP growth rate.

Also, exogenous variables like oil prices and gold prices were added. Furthermore, binary variables like a day before Black Friday, holidays like Christmas, etc., also have a significant impact on sales and hence share prices. So, they were also taken into consideration.

In the end, since it is stock data, the technical indicator <b> Relative Strength Index (RSI) </b> was also added. The Relative Strength Index (RSI) is a technical indicator used in financial markets to measure the speed and magnitude of price movements. It ranges from 0 to 100 and is used to evaluate overbought or oversold conditions in a stock. A common interpretation is that an RSI value above 70 suggests that a stock is overbought, while a value below 30 indicates it may be oversold. RSI can be useful in stock price prediction as it helps to identify potential reversal points or trends in market behavior. By integrating it into deep learning models, it can provide insights into momentum and market sentiment, which are crucial for time-series forecasting of stock prices.
It is calculated as:

$$
\text{RSI} = 100 - \left(\frac{100}{1 + \frac{\text{Average Gain}}{\text{Average Loss}}}\right)
$$



To find the best window size, the dataset was trained on 7 days, 20 days, 50 days, and 100 days. The aim was to capture the short-term as well as the long-term pattern.   

<h2> Starting Models: </h2> The next task was to try various models.  
<h3> Model-1: Long Short Term Memory (LSTM) </h3> <br/>
LSTMs are basic comparison points in any time-series forecasting. Generally, they perform best when the data is not very large, which is the case with our dataset. 
So, it was expected to perform well. I compared various window sizes as mentioned above; the best R<sup>2</sup> score was achieved with the 100-day window. Thus, the model is capturing the long-term trend.

<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (LSTM).png" alt="R2-Score LSTM Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (LSTM).png" alt="Test Loss LSTM Models" width="400" height="200">
</div>

<h3> Model-2:  GRU (Gated Recurrent Unit) </h3>: 
GRU models are simplified versions of LSTMs, designed to capture long-term dependencies in sequential data with fewer parameters and a simpler architecture, making them faster to train. Unlike LSTMs, GRUs merge the forget and input gates into a single update gate and lack an output gate, which reduces computational complexity but might result in slightly less control over memory management compared to LSTMs.

The graph shows that the 50-day window size performs the best among all tested configurations, achieving the highest R-square value, indicating it captures the optimal amount of historical data for accurate predictions. Shorter windows (7 and 20 days) underperform, suggesting insufficient historical context, while the 100-day window, though capturing more data, does not improve further, possibly due to overfitting or increased noise.


<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (GRU).png" alt="R2-Score GRU Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (GRU).png" alt="Test Loss GRU Models" width="400" height="200">
</div>



<h3> Model-3: Temporal Convolution Model (TCN) </h3>: 
TCNs are a type of neural network architecture designed for sequence modeling, combining dilated causal convolutions and residual connections to capture long-range dependencies effectively. They allow for parallel processing of sequences, providing a flexible receptive field and efficiently learning temporal patterns without the need for recurrent structures.

Like the other models, I also tried four windows to run the data. Each TCN model is built with stacked temporal blocks that apply dilated convolutions to capture long-range dependencies in the data. The code sets specific hyperparameters. When comparing the performance of this model to RNN variants, the performance is not as good. Similar to the GRU, the 50-day window size works best here as well.

However, compared to RNN variants, I started with a lower learning rate. The learning rate for all window sizes was different, as was the number of epochs. A lower learning rate and fewer epochs help avoid NaN values during the later stages of training and prevent the learning process from stopping. For all cases the value of kerne_size was taken 3.   

<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (TCN).png" alt="R2-Score TCN Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (TCN).png" alt="Test Loss TCN Models" width="400" height="200">
</div>
<br/>

Below is the given model architecture of the TCN as proposed in research paper: <br/>
 <img src="scores_log/tcn.png" alt="TCN Model Architecture" width="500" height="220" style="margin-right: 10px;">


<h3> Model-4: N-Beats </h3> 
The N-Beats model is a powerful deep learning architecture specifically designed for univariate and multivariate time series forecasting. It uses a stack of fully connected neural networks to directly forecast time series data, leveraging backward and forward residual blocks to capture complex patterns, trends, and seasonality without requiring domain-specific feature engineering. N-Beats is known for its flexibility, high performance, and ability to generalize well across diverse time series datasets.

So, to capture trend and seasonality in the model, two different classes were defined. Seasonality is captured using by generating Fourier series components—specifically, sine and cosine terms—for both historical (backcast) and future (forecast) time windows. By decomposing the time series into multiple harmonics, the function allows the model to represent and learn complex cyclical patterns that occur at different frequencies. This approach enables the N-Beats model to effectively identify and forecast seasonal trends, making it well-suited for time series data with regular, repeating patterns, such as those seen in sales, weather, or financial data.

Similarly polynomial trend was captured using generating polynomial bases for both historical (backcast) and future (forecast) periods. By creating polynomial terms of increasing degrees (up to a specified degree) from the time vector, trend component allowed model to learn and represent linear, quadratic, or higher-order trends within the data. These polynomial components enable the N-Beats model to effectively model non-linear trends, making it well-suited for forecasting scenarios where the data exhibits gradual shifts, growth, or decay over time, thus enhancing the model's overall predictive performance.

These are parameters that were used for this model:
<table> 
<tr> <td> <b> Factor </td> <td> <b> Size </b> </td> </tr>
<tr> <td> Hidden Size </td> <td> 256 </td> </tr>
<tr> <td> Number of blocks (trend, seasonality) </td> <td> 4 </td> </tr>
<tr> <td> Number of Layers </td> <td> 3 </td> </tr>
<tr> <td> Forecast Length </td> <td> 1 </td> </tr>
<tr> <td> Number of harmonics for Fourier Series </td> <td> 10 </td> </tr>
<tr> <td> Polynomial degree for trend block </td> <td> 2 </td> </tr>
</table>  


The graph below shows that model perform best till what all other models that I have tried. Also, it shows that 20 days window is best in capturing the pattern best than other windows, followed by 7-days window. With the 100-day window showing the lowest R-square, suggesting that shorter windows capture the essential patterns better in this scenario. Also as evident from the test loss, this model has showed a lot of fluctuation in the graph as compared to other methods, especially the 100 days window period.  
<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (N- Beats) .png" alt="R2-Score N-BEATS Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (N-Beats).png" alt="Test Loss N-BEATS Models" width="400" height="200">
</div>

<br/>

Below is the given model architecture of the N-Beats as proposed in research paper: <br/>
 <img src="scores_log/n_beats.png" alt="N-Beats Model Architecture" width="500" height="280" style="margin-right: 10px;">

<h3> Model-5: Transformer Model </h3> 
The Transformer model, originally designed for natural language processing tasks, has emerged as a powerful architecture for time series forecasting due to its ability to capture complex temporal dependencies and patterns. Unlike traditional recurrent neural networks (RNNs) that process data sequentially, the Transformer employs self-attention mechanisms, allowing it to directly focus on the most relevant parts of the input sequence, regardless of their position. This capability makes it particularly effective for time series data, where capturing both short-term fluctuations and long-term trends is crucial.

As mentioned transformer uses attention and self-attention mechanism, the Query, Key, and Value vectors are used in the self-attention mechanism to determine the relevance of each input token to others, allowing the model to focus on important parts of the sequence when making predictions. Queries are matched against Keys to generate attention scores, which are then applied to Values to produce the output, capturing the relationships between different time steps.  

Here I impelmented the model from two points of view. In one I keep the learning rate constant and in another learning rate was changing as it was mentioned in the paper.  

$$
lrate = d_{model}^{-0.5} \cdot \min\left(stepnum^{-0.5}, stepnum \cdot warmupsteps^{-1.5}\right)
$$

Also the positional encoding were added using the formula as mentioned in the paper. 

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

Sine positional encoding were applied at the even position and Cosine positional encoding were applied at odd position. Learning rate scheduler was also adjusted according to the paper. The only difference is that instead of by-default 4000 warm-up steps, it was reduced down to 500 steps. Warmup steps refer to the initial phase in the training process where the learning rate is gradually increased over a specified number of steps (in this case, 4000 steps) before transitioning to the main learning rate schedule. Setting warmup steps to 4000 means that during the first 4000 training steps, the learning rate incrementally rises from zero to its peak value, which helps in stabilizing the early stages of training and avoiding abrupt updates to the model’s weights. 

But since my dataset is relatively small; reaching 4000 warmup steps might span a significant portion of overall training, potentially leading to an overly conservative learning rate for much of the training period. In this scenario, adjusting the number of warmup steps to a smaller number (like 500 in this case) would allow the model to reach its effective learning rate quicker, making better use of the limited epochs available for more aggressive learning once the warmup phase is complete.

But model seems to perform better when learning rate was constant which is clearly evident from the graph. In fact, adding the `LearingRateScheduler` like paper, has made the program worse in this case. Below are the parameters that have been used in this case. They are reduced in value, to fit according to this dataset.

<table> 
<tr> <td> <b> Factor </td> <td> <b> Size </b> </td> </tr>
<tr> <td> Transformer embedding Size </td> <td> 128 </td> </tr>
<tr> <td> Number of attention heads (trend, seasonality) </td> <td> 4 </td> </tr>
<tr> <td> Number of transformer encoder layers </td> <td> 1 </td> </tr>
<tr> <td> Forecast Length </td> <td> 1 </td> </tr>
<tr> <td> Hidden layer in feed forward network </td> <td> 64 </td> </tr>
<tr> <td> Dropout Rate </td> <td> 0.1 </td> </tr>
</table>  


Also please note that <b> decoder </b> is not used in this case because the task involves one-step prediction, where the model directly forecasts the next value in the sequence without requiring the autoregressive decoding process typically used for generating sequential outputs in language modeling tasks.

Below graph shows the R-square with or without learning rate scheduler. 
<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (Transformer).png" alt="R2-Score Transformer Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (Transformers).png" alt="Test Loss Transformer Models" width="400" height="200">
</div>

As we can see that above in the image, R-square is negative with learning rate scheduler as mentioned in paper, the reason is beacsue it might not be suitable for the data like this. Also because of the scheduler test loss was very high in the beginning but later it strats to go down. So, overall this model because of lack of data has performed worse than RNN variants like GRU and LSTM even with stable learning rate. The following could be the reasons behind this:
1. Insufficient Data Size: Transformers are highly data-hungry and perform best with large datasets. With only 10,000 rows, the model may struggle to learn meaningful patterns and generalize well, leading to underperformance compared to LSTM and GRU, which are better suited for smaller datasets.
2. Lack of Temporal Inductive Bias: Unlike LSTM and GRU models, which are designed to capture sequential dependencies with their recurrent structures, Transformers rely purely on self-attention mechanisms and may not inherently capture temporal patterns as effectively in smaller datasets, leading to poorer performance.
3. Overfitting Due to Model Complexity: Transformers have a high number of parameters, making them prone to overfitting when trained on smaller datasets. This can lead to poor generalization, especially compared to LSTMs and GRUs, which are simpler and more robust in such settings.

Below is the given model architecture of the Transformer as proposed in research paper: <br/>
 <img src="scores_log/transformer.png" alt="Transformer Model Architecture" width="500" height="220" style="margin-right: 10px;">

<h3> Model-6: Temporal Fusion Transformer </h3> 
So, far we have seen that, we have either used RNN variants for prediction or in last model we have used attention mechanism architecture. But what if we can combine both of the architecture. Temporal Fusion Transformer (TFT) uses both RNN and attention mechanisms. Specifically, it employs recurrent neural networks (often GRU layers) to capture sequential dependencies and learn temporal patterns in the data. Simultaneously, it integrates attention mechanisms, including multi-head attention, to focus on the most relevant parts of the input sequence and dynamically weigh features. This combination allows TFT to effectively model both temporal dependencies and feature importance, enhancing its interpretability and performance in time series forecasting tasks.

In developing the Temporal Fusion Transformer (TFT), following approach was taken inspired from the research paper:
1. Gated Residual Network (GRN): To capture complex interactions between features, GRNs were utilized extensively within the architecture . By incorporating gating mechanisms, dropout, and layer normalization, GRNs enhance the model’s ability to learn sophisticated relationships among both static covariates and dynamic features, ensuring stable and efficient training.
2. Variable Selection Network: To handle the dynamic feature relevance <i> (as model requires separate arrays for static and dynamic type) </i>, next task was to design Variable Selection Network to identify and prioritize the most informative inputs at each time step. This module, through learned attention weights, dynamically selects key features, enhancing the model’s focus on critical variables and boosting interpretability.
3. LSTM Encoder-Decoder: To effectively capture temporal dependencies, next task was to incoprporate LSTM-based encoder-decoder structure in the model. This approach allows the model to encode historical sequences and decode future patterns, capturing both short-term fluctuations and longer-term trends that are crucial for accurate forecasting.
4. Temporal Self-Attention Layer: Now to address the need for capturing long-range dependencies, a temporal self-attention mechanism was introduced. This component enables the model to weigh different time steps adaptively, allowing it to highlight and focus on the most relevant patterns within the sequence, something traditional RNNs often struggle with.
5. Static Covariate Encoder: Now task was to implement Static Covariate Encoder as according to paper using GRNs to incorporate static features, such as categorical or demographic information, that influence the overall forecast. This enrichment helps the model understand the broader context beyond just time-varying inputs.
6. Single Output Forecasting Layer: The final dense layer produces single-step forecasts by synthesizing information from the variable selection network, self-attention outputs, and static covariate encodings. This design focuses on precise, interpretable predictions without requiring extensive quantile outputs. In paper they have used quantiles which are used to provide probabilistic forecasts, offering a range of potential outcomes at different confidence levels (e.g., 10%, 50%, 90% quantiles). This approach is crucial in multi-step or multi-horizon forecasting scenarios, as it allows the model to quantify uncertainty, giving a more comprehensive view of the prediction range rather than a single point estimate. By generating predictions across various quantiles, the TFT can capture the inherent variability and risk in the forecast, which is particularly valuable in applications of forecasting in finance domain, where understanding prediction uncertainty is critical.
So, for our case, I have decided not to include these quantiles for making the program not too complex.

But even with this complex mechanism, the model was not able to perform better than vanilla RNN variants, the many reaons are same as that was in case of Transformer model. In other words, if training does not converge properly or oscillates due to inadequate data, it can lead to worse performance compared to more stable RNN architectures. Below graps summarizes the behaviour:

<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (TFT).png" alt="R2-Score Transformer Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (TFT).png" alt="Test Loss Transformer Models" width="400" height="200">
</div>

Also this table summaraizes which are given time varying features and which are given as static feature in our model:
<div style="display: flex; justify-content: space-between;">

<table>
<tr>
<td>

**Static Columns** | **Type of Variable**
--- | ---
Consumer Price Index | Continuous
Consumer Confidence Indicator | Continuous
GDP Growth Rate | Continuous
Day Before Weekend | Binary
Day Before Holiday | Binary
Day After Holiday | Binary
Day After Weekend | Binary

</td>
<td>

**Dynamic Columns** | **Type of Variable**
--- | ---
Share opening price | Continuous
Share highest price | Continuous
Share lowest price | Continuous
Volume traded | Continuous
Relative Strength Index | Continuous
Volatility Index | Continuous
Gold Prices | Continuous
Oil Prices | Continuous

</td>
</tr> 
</table>

</div>

Below is the given model architecture of the Temporal Fusion Transformer (TFT) as proposed in research paper: <br/>
 <img src="scores_log/tft.png" alt="TFT Model Architecture" width="500" height="220" style="margin-right: 10px;">

<h3> Model-7: Informer Model </h3> 
The Informer model was developed to address the challenges of long-sequence time series forecasting, particularly the high computational cost and inefficiencies of standard Transformer models. Informer introduces a sparse self-attention mechanism that selectively attends to the most relevant parts of the sequence, significantly reducing memory and computation while maintaining high forecasting accuracy. By integrating an encoder-decoder architecture with enhanced positional encoding and attention distillation techniques, Informer effectively captures complex temporal dependencies, making it well-suited for large-scale, data-intensive forecasting tasks across various domains.

Here is the workflow of the informer model:
1. Positional Encoding: First task was to make the psoitional encoding. This is same as the transformer mode. This approach allows the Informer to maintain the order of the data, enhancing its ability to capture the sequential nature of time series.
2. Informer Enocder & Decoder Layer: Next task was to design the encoder layer. The encoder layer integrates multi-head self-attention and a feedforward network with ELU activation, designed to capture both global dependencies and local patterns in the input data. By including layer normalization and residual connections, it is ensured that the network maintains stability and effective learning across deep layers.
After encoder layer decoder layer was designed. The decoder consists of self-attention, cross-attention, and a feedforward network, enabling the model to attend to both current target sequences and encoder outputs. The use of cross-attention helps the model focus on relevant historical patterns that influence future predictions, effectively merging past and present information.
So to summarize this part: The encoded data passes through the Informer Encoder Layer and Multi-Head Attention mechanisms, capturing both local and global dependencies, followed by the Informer Decoder Layer that integrates past and present information.
3. Output Projection to Final Prediction: In the last task, the processed information is passed through an output projection layer, generating the final prediction, completing the forecasting process.

Here are the parameters that are used in the model:
<table> 
<tr> <td> <b> Factor </td> <td> <b> Size </b> </td> </tr>
<tr> <td> Dimension of the model's embedding space </td> <td> 128  </td> </tr>
<tr> <td> Number of attention heads  </td> <td> 8 </td> </tr>
<tr> <td> Number of encoder layers </td> <td> 1 </td> </tr>
<tr> <td> Number of decoder layers </td> <td> 1 </td> </tr>
<tr> <td> Dropout Rate </td> <td> 0.2 </td> </tr>
</table>  

From the below graph, the model has shown good performance when the given window size was the 20. But it also showed the good performance when the window size was 100, highest in our case. But according to the model's value proposition, model should show good performance on 100 days window size than 20 days window size. The resoans could be following:
1. Here not all parts of the long sequence may be relevant for the prediction task. A shorter window size (20 in this case) allows the model to focus on the most recent and relevant patterns without being overwhelmed by less relevant historical data, leading to better performance.
2. With a limited dataset, longer windows consume a larger portion of the available data, leading to sparse training samples and potentially inadequate learning of relevant patterns. This scarcity of data can hinder the model's ability to generalize well, making shorter windows like 20 more effective as they utilize the available data more efficiently without overfitting.

<div style="display: flex;">
  <img src="scores_log/R-Square/R-Square (Informer) .png" alt="R2-Score Informer Models" width="400" height="200" style="margin-right: 10px;">
  <img src="scores_log/Test Loss/Test Loss (Informer).png" alt="Test Loss Informer Models" width="400" height="200">
</div>

Below is the given model architecture of the Informer as proposed in research paper: <br/>
 <img src="scores_log/informer.png" alt="Informer Model Architecture" width="500" height="220" style="margin-right: 10px;">

<h2> Results: </h2>
Various model has been build to check the which model performs best on the data given. Since dataset was small, in most of the cases RNN variants outperfom the Transfomer models. Also window size of 20 has shown promising output as compared to other window size. Out of all the models, <b> N-Beats </b> model has shown best performance. Below table summarizes for which model, was the best window size and also mentione the running time. Please note that there is also `EarlyStopping` function is applied to it. <br/>

| Model Name                    | Best Window Size | Running time on GPU | R2-Score |
|-------------------------------|------------------|---------------------|----------|
| LSTM                          | 100              | 1m 11s              | 0.82166  |
| GRU                           | 50               | 1m 1s               | 0.885    |
| Temporal Convolution Network  | 50               | 1m 30s              | 0.82263  |
| N-Beats                       | 20               | 4m 46s              | 0.89899  |
| Transformer (No LR Scheduler) | 7                | 1m 51s              | 0.74     |
| Temporal Fusion Transformer   | 20               | 6m 23s              | 0.62399  |
| Informer                      | 20               | 6m 32s              | 0.76795  |

So, from above N-Beats was our best model, followed by GRU. Transformer model didn't perform that well because of less data. The following table tries to capture the reason and performance of the model:
| Model Name                    | Strengths                                                     | Weaknesses                                                  | Reason for Performance                              |
|-------------------------------|---------------------------------------------------------------|-------------------------------------------------------------|-----------------------------------------------------|
| **LSTM**                      | Good at capturing sequential patterns; stable on smaller data | Limited long-term memory and struggles with complex patterns | Performed well due to effective handling of sequential dependencies in moderate data sizes. |
| **GRU**                       | Simplified structure; faster and more efficient than LSTM     | Can still miss long-range dependencies                      | Performed best among recurrent models due to its efficient gate structure and fewer parameters. |
| **Temporal Convolution Network** | Captures temporal dependencies with parallel processing    | Limited by fixed receptive fields; less flexible for varying patterns | Performs well with shorter windows due to its effective convolutional blocks but lacks global pattern recognition. |
| **N-Beats**                   | Strong for trend and seasonality capture; interpretable       | High training time and data-hungry                          | Performed best overall due to its ability to directly model trends and seasonality with residual blocks. |
| **Transformer (No LR Scheduler)** | Excellent at capturing global dependencies; parallelizable | Struggles on smaller datasets; overfits easily              | Underperformed due to data inefficiency and lack of temporal inductive biases. |
| **Temporal Fusion Transformer**   | Combines RNN and attention; interpretable; handles multivariate inputs | High computational cost; prone to overfitting               | Struggled due to data complexity and model’s sensitivity to hyperparameter tuning. |
| **Informer**                  | Efficient attention mechanism; scales well with large data    | Still computationally intensive; requires tuning            | Balanced performance due to selective attention mechanism but lacked deeper recurrent context. |


### Model Running Times (on GPU)
The follwoing graphs shows all the time taken by all the models. Please note that training might stop early because of `EarlyStopping`. These graph could be used to compare accuracy vs training time.   
<table>
  <tr>
    <td>
      <img src="scores_log/Running Time/Running Time (LSTM).png" alt="Running Time LSTM" width="300">
      <p align="center"><b>LSTM</b></p>
    </td>
    <td>
      <img src="scores_log/Running Time/Running Time (GRU).png" alt="Running Time GRU" width="300">
      <p align="center"><b>GRU</b></p>
    </td>
    <td>
      <img src="scores_log/Running Time/Running Time (TCN).png" alt="Running Time TCN" width="300">
      <p align="center"><b>Temporal Convolution Network</b></p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="scores_log/Running Time/Running Time (N-Beats).png" alt="Running Time N-Beats" width="300">
      <p align="center"><b>N-Beats</b></p>
    </td>
    <td>
      <img src="scores_log/Running Time/Running Time (Transformers).png" alt="Running Time Transformer" width="300">
      <p align="center"><b>Transformer</b></p>
    </td>
    <td>
      <img src="scores_log/Running Time/Running Time (TFT).png" alt="Running Time TFT" width="300">
      <p align="center"><b>Temporal Fusion Transformer</b></p>
    </td>
  </tr>
  <tr>
    <td>
      <img src="scores_log/Running Time/Running Time (Informer).png" alt="Running Time Informer" width="300">
      <p align="center"><b>Informer</b></p>
    </td>
  </tr>
</table>

<h2> Future Prospects: </h2>
In this research setup, I tried some popular deep learning algorithms for time series forecasting. However, there are other algorithms that could be explored in the future, such as Amazon's DeepAR and Chronos. Similarly, zero-shot algorithms like TimeGPT could be used to assess performance when the model has not seen any specific data. If the performance is promising—especially since TimeGPT is also trained on financial data—it might be worth considering. Additionally, other new breakthroughs in AI, such as diffusion models (though primarily used for generating images), could be interesting to try.

Furthermore, these models could be tested on datasets like M3 and M4, which encompass various types of seasonality, including daily, hourly, and yearly patterns. 

<b> Note: </b> This research was conducted for experimental purposes only, not for any investment purposes.






 
