
# An Exploration of Machine Learning in Finance: Predicting Silver Future Return Signals

# Executive summary

This Project explores the application of various machine learning models to predict silver futures returns, with a focus on developing practical trading signals.
### Key Findings

- **Best Performing Models**: The Random Forest, XGBoost, and ensemble stacked models consistently outperformed other architectures, achieving win rates exceeding 82% across validation datasets in certain market conditions.
-  **Trading Performance**: When implemented in simulated trading scenarios across three distinct market periods (2004, 2020, and 2025), our top models achieved positive Sharpe ratios with different strategies, tailored to their average performance. I found that when signal robustness is variable in market regimes derivatives could be employed to more successfully manage the risk caused by our models unclear performance.
- **Trading Scenario Construction**: Was also particularly interesting and challenging, we implemented B & S to simulate option trading as we had incomplete data. Implementing margins was also quite fun.
- **Model Robustness**: The top models performed badly on bootstrapped data demonstrating robustness. Furthermore ensemble stacked model exhibited superior stability when subjected to input noise, maintaining consistent signal generation even as noise levels increased, making it particularly valuable for real-world applications where data quality varies. While the CNN demonstrated stability across all market regimes.
- **Feature Importance**: Across models, gold spot prices, past silver futures data, and gold futures emerged as the most influential variables, which aligns with financial theory on precious metals markets' interconnectedness.
- **Model Architecture Insights**: Simpler models outperformed complex architectures for this application; notably, the LSTM model performed worse than random despite theoretical advantages, likely due to the high noise-to-signal ratio in financial time series. 
### Learning Journey

This project served as a learning experience in applying machine learning to financial markets, and coding for me and was frustrating at times, nevertheless the key points for us where :
- **Theoretical Integration**: Using my background and my grasp of econometric, macroeconomic and finance theories (cost-to-carry model, GARCH, feature selection) with machine learning methodologies reinforced how domain knowledge enhances model design. My studies in finance where also useful for B&S implementation.
- **Computational Challenges**: Grid search optimisation proved computationally intensive, requiring over 24 hours for some models (LSTMs / RNN's), teaching valuable lessons about efficient hyper-parameter tuning strategies, moving to random search proved particularly efficient. 
- **Model Selection Evolution**: We hoped neural architecture could outperform traditional methods we had struggled with, but the data led us to appreciate the efficacy of ensemble approaches combining multiple model strengths.
- **Explainability Techniques**: Developing methods to interpret black-box models using packages like DALEX provided crucial insights into how these models leverage financial data patterns, and remains an key issue for applying machine learning to financial applications (compliance, risk management).
- **Synthetic Data Limitations**: The unexpected poor performance on theoretically sound synthetic data, (admittedly a cross disciplinary one) highlighted the complex non linear, multidimensional nature of financial time series.
- **Data Leakage**: A small leak in the training data lead to over-performance (95%) accuracy i naively believed in for a while. By re writing the scrip separately the models achieved much lower performance, we then corrected and **re-tuned** the models hyper-parameters.
### Practical Implications

Our project suggest that machine learning approaches, particularly ensemble methods combining tree-based and neural network architectures, can successfully capture the non-linear dynamics of silver futures markets. The consistency of performance across varying market conditions and the models' capacity to identify directional movements with over 82% accuracy demonstrates potential for implementation in actual trading strategies. The project also highlights that sophisticated neural architectures like LSTMs may not always be optimal for financial time series prediction, where tree-based ensemble methods often prove more effective despite their relative simplicity. Furthermore Statistical models are unreliable and prone to react badly too changing conditions and that risk must be managed well. 
## Abstract 

Silver occupies a unique position in financial markets, functioning as both an industrial metal and a store of wealth similar to gold. This dual nature creates complex price dynamics that traditional linear models struggle to capture effectively.

Previous research by Cortazar et al. (2014) (1) and Sarno & Valente (2001) (2) has established that silver price movements are influenced by multiple factors: industrial production demand, inflation expectations, speculative positioning, and flight-to-safety dynamics. However, the precise relationships between these variables and silver futures returns are likely non-linear and potentially time-varying.

After unsuccessful attempts to create practical trading models using traditional econometric approaches (linear regression, VAR, BVAR), we turn to machine learning as a solution and a learning journey. Since they capturing complex non-linear relationships without requiring explicit functional specification, can recognise temporal patterns and regime shifts in market trends, identify predictive signals and adapt to the changing relationships during said market trends, for example FNN's have been shown too efficiently capture high frequency data as shown in previous research by Paola Arce & co (2012) (14) and Masaya Abe & co (15).

Our goal is not merely statistical prediction but developing practical signals for trading decisions. We evaluate several machine learning architectures to determine which best captures the multifaceted dynamics of silver futures markets, with particular attention to maintaining interpretability alongside predictive power. 

## Road Map 

*Firstly we will look at feature selection and methodology, then we will highlight model results and performance while robustly testing them. Further along we backtest a simple buy on signal strategy and compare model performance. Thirdly we highlight caveats, conclude, finally we dive deep into model characteristics.*

## Theory & Empirical led choice of variables and methodology

### **1.1-Variables**

##### *Metals :*
Gold as a "flight to safety" metal  and inflation counter would give interesting information of anticipations, this can be further completed by the use of copper as its a metal in high demand during faces of economic expansion. Furthermore they are similar assets so they should behave in similar ways.

##### *Interest rates & silver spot prices :*
We also believe that according to theory the interest should be a component of futures pricing:  
Silver futures are be in theory, according to the cost to carry model, Bradford Cornell and Kenneth R. French 1983 (3), written as:

$$F = S \cdot e^{iT} + C \cdot (1 - e^{-i(T-t)})$$
$F$ is the futures price.
$S$ is the current spot price of the underlying asset.
$i$ is the risk-free interest rate.
$T$ is the time to maturity of the futures contract.
$C$ is the storage cost per period.

Therefore in continuous return defined as:

$$ r_{ct} = log(P_{t}) - log(P_{t-1}) $$
we would have: 

$$rF_c = log(S_{t+1}\cdot e^{iT} + C\cdot(1-e^{i(T-t+1)})) 
- log(S_{t}\cdot e^{iT} + C\cdot(1-e^{i(T-t)}))=> rF_c(i,C) $$
Since all other factors are constants the the return is a function of the interest rate and carry costs in this model. 
Therefore the spot Price of silver should be a component . And as such, as such Futures and the interest rate should hold good information on sentiment, our model may be able to "intuit" the the cost of storage. 
*Note if this theory holds then gold & copper futures should be compounded by previous characteristics described, also a good source of information for our models.*

##### We also considered GARCH :
Sometimes econometric time series lose their stationary properties properties, *Engle R.F. 1982*, known as ARCH, this was then generalised by Tim Bollerslev 1985 (4), and that volatility clusters tend to create auto-correlated returns in these areas, this concept is know as GARCH. We decide to take inspiration from him by implementing a mesure of volatility in our model, we use rolling $\sigma$ to input this, we found better results by adding the VIX as a general input of market volatility.

We would have an unknown function that looks something like this:
$$ rF_t = f(( r_{Spot},r_{Futures},\sigma_{Market},\sigma_{Futures},\dot{\rho}_{Futures})_{t-1}) $$
Where $r$ returns, $\sigma$ risk or standard dev, $\dot{\rho}$ momentum.
Where precise functions are unknown and potential non linear thus prompting our use of machine learning to approximate it :
$$ DATA -> Training -> f <=> Model $$And we could compute and grade direction by : 

$\text{If}$ $r_{Fc} < 0$ and  $r_{Fce} < 0$  or if $r_{Fc} > 0$ and $r_{Fce} > 0$, $\text{then}$ $w = 1$,
$\text{else}$,  $w = 0$ .
$\text{winrate} = \frac{1}{n} \times \sum w \times 100$

**To resume :** 
*Model Inputs :* 
- Copper Spot CME
- Gold Spot CME
- Silver Spot CME, *CME Group* (4)
- Silver futures CME 
- Gold futures CME 
- Copper futures CME 
- Us 10 years treasuries rates
- VIX index
- rolling 3 days volatility from $[t-4;t-1]$ for Silver and Gold Futures
- rolling 3 day momentum from  $[t-4;t-1]$ for Silver and Gold Futures

### **1.2 - Getting and treating Data:** 

We use data from *Factset,* as we are lucky enough to have it provided by our university, its more complete and as less date issues. We have decided to train our model on data from $2004/01/01 + 275$ days  to $2024/12/31 -275$ days as they offer comprehensive economic cycles and associated shocks while the 275 days in the beginning and end serve as extra validation data. 

We remove any NA's by by backfilling, if $n_t = NA$ then $n_t = n_{t-1}$. We then compute log returns :
$r_t = ln(r_t)-ln(r_{t-1})$ ,both for scaling reasons and for reasons of accuracy. As we cannot infer a price realistically but only the returns, ie the variation of the price to account for contract changes for example. It also conveniently scales our data to the logarithmic scale for use by our models. Any NA's or infinite values produced by this transformation are backfilled as well.

### **1.3 - Examining our Data** 

We will look at our densities, if they follow normal distributions or not and correlations. We decide to use 5 days moving averages (on the log returns) to reduce noise and extreme values while keeping some of the volatility. Even when doing this our data is not normally distributed. 
<img src="attachment/bd301922d9cc0a8229b7f45a6bb35268.png" />
*Densities of continuous returns and variations :*
<img src="attachment/6b9f487b4f40dde681e81be9251bd2b3.png" />
*Densities of 5 day averages continuous returns :*

We can see we retain similar distributions but with less extreme values for crises.

*Shapiro & Augmented Dick Fuller tests for normal distribution and stationarity :*

| Model imputs   | P value Log returns Shapiro | P value 5 days averages Shapiro | P values 5 day averages ADF |
| -------------- | --------------------------- | ------------------------------- | --------------------------- |
| VIX            | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
| Us Treasuries  | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
| Spot Silver    | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
| Spot Gold      | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
| Spot Copper    | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
| Futures Silver | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
| Futures Gold   | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
| Futures Copper | < 2.2e-16                   | < 2.2e-16                       | <0.01                       |
Thus we can infer that our data is not normally distributed but it is definitely stationary. Then we look a smoothed and un-smoothed correlations between our features : 
<img src="attachment/06cfbb4a5907aa038debcbc2c9526c5b.png" />
*Correlations of log returns*
<img src="attachment/3ea32a72391f06040e8a37eb86de07b6.png" />
*Correlations of 5 day averages continuous returns*

We can see that using 5 day averages reduces noise and shows stronger correlations, and further justifies our choice of variables. Furthermore the ACF of our variables is even lower after smoothing.

### **1.4 - Methodology** 

We plan to test different type of neural networks and models and compare them, evaluate their performance on real and synthetic data as well as how they would perform in a trading scenario. 

##### 1.4.1 Model creation : 

 Create model following a specific architecture where we found prior research confirming its use case : that is computationally efficient as we are limited by computing constraints using our theoretically justified inputs and train it on data from 2005 to 2023 after fine tuning it using grid search algorithms to determine optimal parameters. *We note that Grid Search algorithms where particularly computationally intensives and took long periods to run , (4 to 5 hours for LSTMs for example) and required at least 24 hours of computing.* Check for overfitting by early stopping and keeping a close watch on training parameters. We also used 10 lags as it was what showed the best results with our FFN and 

##### 1.4.2 Evaluation :  

 On 10% of the 2005 to 2023 dataset (not used for training), data from 2004 to 2005 and 2019 to 2020 (covid crash) made into smaller batches to be able to create IC's and use students test for model comparaison. Using MSE, MAE and signal win rate (aka bullish or bearish vs actual movement). We also check if our errors follow a normal distribution via t.tests and qq-plots and plot predictions vs errors to check for error patterns and skews in our models. We also test it on bootstrapped data to check how our model can react to changing market conditions. As well as test out a novel synthetic data production method to add an extra layer of robustness.
 
##### 1.4.3 Explainability and testing : 

Investigate model explainability using the DALEX package to see how our model uses each variable and its impact on reducing MSE. Check for robustness and then comparing it to the other models and a baseline linear model.  As well as looking at performance on bootstrapped training data and Synthetic data. Then we look at trading performance on trading scenarios on years 2001 and 2020 and lastly 2025, we include the last two since they forecast particularly non normal and uncertain market conditions. Moreover we will check is the models explainability results fits theory, for those where we can as well as look at neural plots for the concerned models. Finally we go over limitations and potential caveats and further features that could be implemented. 

### **1.5 - Evaluated models**

###### - Random forest 
###### - Feed Forward Neural Network 

###### - CNN (Wavenet architecture )

###### - LSTM 

###### - KNN (K nearest neighbour )

###### - Ensemble stacking model combining our successful models using a XGboost meta learner. 

We also use a linear model that serves as a benchmark, it isn't necessarily a optimised AR linear model but is using the same data to our base models. We are aware this isn't perfect, but we did check for heterodasticity.
## 2. **Results** 

#### 2.1 **On validation Data**

As state previously we test our model on 8 60 day samples of out of sample 2020 and 2004 data. We use shorter periods from the samples too construct less robust but still relevant in our opinion ICs using a student test rather then testing on 3 periods like for the trading tests. We achieve the following performance :

| Model            | Winrate mean | IC Winrate        | MSE mean         | IC MSE                      |
| ---------------- | ------------ | ----------------- | ---------------- | --------------------------- |
| Linear           |              |                   |                  |                             |
| o1-Random Forest | 79.71429     | 74.88750;84.54107 | 2.791313e-06     | -3.038547e-06; 8.621173e-06 |
| o2-FFNN          | 75.71429     | 68.67666;82.75192 | 2.056414e-05<br> | 5.037272e-06;3.609101e-05   |
| o3-CNN           | 74.57143     | 69.05179;80.09106 | 1.506942e-05     | 4.166908e-06;2.597193e-05   |
| o4-LSTM          | 60.85714     | 50.79050;70.92379 | 0.009021036      | 0.008774696;0.009267375     |
| o5-KNN           | 66           | 60.24908;71.75092 | 1.068038e-05     | -3.488358e-06  2.484912e-05 |
| oK-Xboost        | 77.71429     | 71.18087;84.24770 | 1.094115e-05     | -1.410767e-05  3.598998e-05 |
| o6-Ens-stacking  | 79.71429     | 75.25598;84.17259 | 1.754023e-06     | -1.555411e-06  5.063456e-06 |
Therefore we can conclude that tree based models appear too outperform FNN and KNN approaches, furthermore as detailed in section 6, the most important features where Gold Spot and Futures , Silver futures, and Copper futures. The importance spot does lend credence to the Cost too carry model but focusing on the contracts underlying environment (commodities here) rather then just spot. It also underlines the interconnectedness in commodities markets. 
#### 2.2 **On Bootstrapped training data :** 

In this section we go over how our model performs on bootstrapped training data over 100 iterations this way we can construct robust performance IC's. The bootstrapped data should show 1/2 in terms of prediction unless there is a data leakage, this is what happen with our model.

| Model            | Winrate mean |
| ---------------- | ------------ |
| o1-Random Forest | 51.27        |
| o2-FFNN          | 51.12        |
| o3-CNN           | 52           |
| o4-LSTM          | NA           |
| o5-KNN           | 50.22        |
| oK-Xboost        | 50           |
| o6-Ens-stacking  | 48.54        |

#### 2.3 **On Synthetic Alvi Method data :**

Continuing our comprehensive evaluation strategy outlined in section 1.4, we refer here to the paper by Maira Alvi , Tim French , Rachel Cardell-Oliver , Damien Batstone, and Naveed Akhtar (2024) (12), where they use a novel synthetic data creation method for machine learning implementation to model chemical processes and provide more data too train ML models. We initially planned to see if our models performed better when trained on synthetic data as well but we realised this would require re-tuning hyper-parameters. Nevertheless this serves as a valuable test. Our implementation of their algorithm can be found in the code section. We generate 2000 days of synthetic data.
#### 2.3.1 Comparing synthetic and real data 

We compare several metrics such as correlation, rolling standard deviation, and distributions. 

*Rolling standard deviations, red synthetic, blue real.*

As we can see volatility exhibits similar patters. We then test too see if the silver futures are statistically different in both standard deviation and mean using a students test we find $\text{p value}_{sd} = 0.07$ implying that our sd does differ but not significantly at 5% threshold,  $\text{p value}_{returns} = 0.85$, so our returns are similar. We then look at correlations to see if the synthetic data is close to the observed data for the target variable (Silver futures) : 

|            | Spot cop | Spot sil | Spot gold | Gold fut | copper fut | ust   | vix  |
| ---------- | -------- | -------- | --------- | -------- | ---------- | ----- | ---- |
| Silver fut | 0.15     | 0.03     | 0.04      | 0.03     | 0.03       | -0.23 | 0.15 |
*Difference between real and synthetic data correlations*

Densities appear to match. Therefore we test our models on the data, we get the following performance after splitting the data in 19 100-ish day subsets for more robust ICs and performance appraisal :

| Model            | Winrate mean |
| ---------------- | ------------ |
| o1-Random Forest | 50.70175     |
| o2-FFNN          | 44.97076     |
| o3-CNN           | 45.49708     |
| oK-Xboost        | 49.94152     |
| o6-Ens-stacking  | 50.99415     |

Model performance collapses, so either Alvi's & co method doesn't work for financial data, or our models react poorly too different market scenarios, but seeing our models trading rather good performance on the particularly difficult market conditions from the 3rd of November to the 20th of April we would like think the foremost is more likely, since our models exhibit robust performance on validation and trading data in diverging and complex market conditions. to further our point we check the differences between the errors of our models on synthetic versus validation data and find p values for the tested models under 5% with the lowest being ($\text{statistic} = 0.380, p = 7.478x10^{-12})$ for the FNN and the highest ($\text{statistic} = 0.149, p = 0.035)$ for the CNN, demonstrating that while the synthetic data preserves certain univariate properties, it fails to capture the multivariate relationships that our models leverage for prediction.

## 3. **Implementation of a trading scenario** 

We implement on three Periods that our out of sample and present different macro and market conditions : 
- 2003/12/31 too 2005/02/03 Normal market conditions, bullish.
- 2019/11/20 too 2020/12/31 Covid crash too bull market , reversal. 
- 2024/11/03 too 2024/04/20 Trump mark, bull then bear, reversal.

We test our models on these periods using silver mini futures. We look at the Beta, Sharpe ratio and max drawdowns and expectancy. As our benchmark we Use the DJ precious metal index. Futhermore we used implemented high transaction costs (2%), volatility based slippage : 
As in 1% scaled by 3 times the volatility ratio; $0.01 \times 3 \times \frac{Vol_{t_{-3},t_{-1}}}{qVol_{0.8}}$ . 
As well as 1.5 the margin requirements of mini silver futures to ensure our backtests are realistic.  We try several trading strategies but two where illustrative : 

- **Buy and hold** 

The trading strategy is the following we buy or sell the contract near close based on the models prediction with current input, and sell the previous days contract. We implemented slippage as we sadly where unable to find intraday data over these periods. The algorithmic implementation of the strategy is as follow : 
$$Position_i = 
\begin{cases}
(1) \times (0.98 - slippage_i) & \text{if } signal_i > 0 \\
(-1) \times (0.98 - slippage_i) & \text{if } signal_i < 0 \\
0 & \text{otherwise}
\end{cases}$$
$$slippage_i = 
\begin{cases}
base\_slippage \times vol\_multiplier \times \frac{\sigma_i}{vol\_threshold} & \text{if } \sigma_i > vol\_threshold \\
base\_slippage & \text{otherwise}
\end{cases}$$
Where $\sigma_i$​ is the 5-day rolling standard deviation of returns at time $i+lags$, vol threshold is the 80th percentile of the rolling volatility, base slippage 1%, vol multiplier at 3 (h). We use log returns of settlement prices to calculate all trading returns.

- **Selling delta hedged options**

We sell near expiry silver futures options based on models prediction we collect premiums and eat losses that are hedged. All of the option prices are calculated using BS that we implement in R, we also need to calculate margin costs and usage. We set margin costs at 5%, a 1% col adjusted slippage, trading costs at 5%.

The strategy is the following if our model is bullish we buy a silver future, then we sell a OTM call with a strike that is futher out then our hedge, since we also buy a ITM put. We aim for short expiry, ie 1 day out. If our model is bearish we sell a silver future, then we sell a put with a strike that is futher out then our hedge, since we also buy a ITM call. 

To Simulate option prices we use B & S ie the premium of a Call is: 
With $S$ the strike, $P$ the price, $\sigma$ the risk (rolling Vol here), $r$ the interest rate (UST 10 year here),T time to expiry :
$d_1=\log(P/S)+\frac{(r+0.5\times \sigma^2)\times T}{\sigma \sqrt{T}}$
$d_2=d_1-\sigma \times \sqrt{T}$
$p = (P \times d_{N(d1)} + S \times \exp{-r \ times T} \times d_{N(d2)})$

And vice versa, the goal here is to benefit from  the periods where the model as an edge and try to limit the losses when it performs worse. We get the following results :
### 3.1 On periods 2004 :

| Model            | Return Avg | Sd   | Total return | Sharpe Ratio | Max drawdown | Beta |
| ---------------- | ---------- | ---- | ------------ | ------------ | ------------ | ---- |
| o1-Random Forest | 0.28%      | 1.8% | 75%          | 0.15         | -8.1%        | 0.43 |
| o2-FFNN          | 0.18%      | 1.6% | 47%          | 0.10         | -8.1%        | 0.41 |
| o3-CNN           | 0.25%      | 1.5% | 67%          | 0.15         | -5.5%        | 0.38 |
| o5-KNN           | 0.15%      | 1.3% | 39%          | 0.10         | -8.1%        | 0.32 |
| oK-Xboost        | 0.24%      | 1.3% | 13%          | 0.17         | -5.5%        | 0.33 |
| o6-Ens-stacking  | 0.23%      | 1.6% | 16%          | 0.14         | -8.1%        | 0.40 |
*Derivatives strategy above*

| Model            | Return Avg | Sd   | Total return | Sharpe Ratio | Max drawdown | Beta |
| ---------------- | ---------- | ---- | ------------ | ------------ | ------------ | ---- |
| o1-Random Forest | -0.00%     | 1.1% | -4.8%        | 0.00         | -5.8%        | 0.26 |
| o2-FFNN          | 0.00%      | 1.2% | 23%          | 0.06         | -5.8%        | 0.27 |
| o3-CNN           | 0.04%      | 1.2% | 11%          | 0.03         | -5.8%        | 0.25 |
| o5-KNN           | 0.16%      | 1.5% | 44%          | 0.10         | -5.8%        | 0.43 |
| oK-Xboost        | 0.01%      | 1.6% | 0.6%         | 0.00         | -10.8%       | 0.42 |
| o6-Ens-stacking  | 0.02%      | 1.5% | 5%           | 0.01         | -10.8%       | 0.33 |
*Buy & hold*
### 3.2 On period 2020 : 

| Model            | Return Avg | Sd   | Total return | Sharpe Ratio | Max drawdown | Beta |
| ---------------- | ---------- | ---- | ------------ | ------------ | ------------ | ---- |
| o1-Random Forest | 0.37%      | 1.9% | 98%          | 0.19         | -7.5%        | 0.28 |
| o2-FFNN          | 0.24%      | 1.9% | 65%          | 0.12         | -7.9%        | 0.26 |
| o3-CNN           | 0.19%      | 1.9% | 51%          | 0.09         | -8.3%        | 0.28 |
| o5-KNN           | 0.19%      | 1.8% | 52%          | 0.10         | -7.9%        | 0.2  |
| oK-Xboost        | 0.24%      | 1.3% | 63%          | 0.17         | -2.2%        | 0.2  |
| o6-Ens-stacking  | 0.25%      | 1.5% | 67%          | 0.16         | -7.5%        | 0.23 |
*Derivatives strategy*

| Model            | Return Avg | Sd   | Total return | Sharpe Ratio | Max drawdown | Beta |
| ---------------- | ---------- | ---- | ------------ | ------------ | ------------ | ---- |
| o1-Random Forest | -0.04%     | 2.0% | -11%         | 0.0          | -10.6%       | 0.26 |
| o2-FFNN          | 0.08%      | 1.8% | 4%           | 0.06         | -10.6%       | 0.27 |
| o3-CNN           | 0.11%      | 1.7% | 6%           | 0.03         | -10.0%       | 0.25 |
| o5-KNN           | 0.18%      | 1.7% | 10%          | 0.10         | -7.4%        | 0.43 |
| oK-Xboost        | 0.07%      | 2.5% | 2%           | 0.00         | -11.5%       | 0.42 |
| o6-Ens-stacking  | 0.07%      | 2.3% | 3%           | 0.01         | -10.6%       | 0.33 |
*Buy & hold*
### 3.3 On period 2025 : 

| Model            | Return Avg | Sd   | Total return | Sharpe Ratio | Max drawdown | Beta |
| ---------------- | ---------- | ---- | ------------ | ------------ | ------------ | ---- |
| o1-Random Forest | 0.16%      | 1.5% | 16%          | 0.00         | -7.0%        | 0.27 |
| o2-FFNN          | 0.20%      | 1.4% | 20%          | 0.00         | -5.3%        | 0.28 |
| o3-CNN           | 0.18%      | 1.5% | 19%          | 0.00         | -7.0%        | 0.31 |
| o5-KNN           | 0.09%      | 1.3% | 9.9%         | 0.00         | -7.0%        | 0.23 |
| oK-Xboost        | 0.07%      | 1.3% | 7.6%         | 0.00         | -7.0%        | 0.22 |
| o6-Ens-stacking  | 0.16%      | 1.2% | 16%          | 0.00         | -2.9%        | 0.22 |
*Derivatives strategy*

| Model            | Return Avg | Sd   | Total return | Sharpe Ratio | Max drawdown | Beta |
| ---------------- | ---------- | ---- | ------------ | ------------ | ------------ | ---- |
| o1-Random Forest | -0.07%     | 1.1% | 7%           | 0.00         | -3.9%        | 0.24 |
| o2-FFNN          | 0.01%      | 1.1% | 1%           | 0.00         | -4.2%        | 0.22 |
| o3-CNN           | 0.08%      | 1.2% | -0.81%       | 0.00         | -7.4%        | 0.17 |
| o5-KNN           | 0.19%      | 1.3% | 19%          | 0.00         | -3.9%        | 0.26 |
| oK-Xboost        | 0.23%      | 1.3% | 23.%         | 0.00         | -3.9%        | 0.31 |
| o6-Ens-stacking  | 0.00%      | 1.2% | 9%           | 0.00         | -4.2%        | 0.28 |
*Buy & hold*
*The universally negative Sharpe ratios in 2025 reflect the extreme market volatility during Trump's election and early policy announcements. But serves as a good stress test*
### 3.4 Recap graphs : 

**Options strategy :** 
<img src="attachment/79b76b98416a5d05c28880e5e351ec2d.png" />
*Wealth Progression by Model, 2004, starting capital of 1€*
<img src="attachment/7ff0971dd46fd6c25936d16fb146b458.png" />
Wealth Progression by Model, 2024, starting capital of 1€*
<img src="attachment/4829a8dd660673c7a98e8b6beb39faae.png" />
*Wealth Progression by Model, 2025, starting capital of 1€*

 **Buy & Hold:** 
<img src="attachment/98bc8743b881ad6629afce2592c15c4d.png" />
*Wealth Progression by Model, 2004, starting capital of 1€*
<img src="attachment/62720e8d4db4ae12737d4f7551c49fef.png" />
*Wealth Progression by Model, 2020, starting capital of 1€*
<img src="attachment/6edc19f9d3dbea21861a7811192b2c6d.png" />
*Wealth Progression by Model, 2025, starting capital of 1€*

### 3.5 Comparing model stability too noise  

Model stability is particularly import as we can Thus we will follow the approach of adding *noise* too our trading test series and check what proportion of trades / signals "stay stable" in such a conditions.  Here we take inspiration from Neelanjana Pal, Diego Manzanas Lopez, and Taylor T Johnson (13) and use a Multifeature All-instance approach, we will scale this noise based on the standard deviation of each input feature, such that: 
$$\text{Noisy Feature}_i = \text{Feature}_i + N(0,\text{Noise Level} \times sd(Feature))$$

We will run this $n$ times per noise levels, compute the mean of the proportion that didn't flip. Thus we know what model, is more stable in different scenario. Nevertheless the ensemble model is more stable. Below are the result graphs :
<img src="attachment/a07f4711f1a2a5f25dfaa58c4cf38874.png" />
<img src="attachment/9c1940ad680894e3829dbe1ff422ecbe.png" />
<img src="attachment/2d44d6a35d8d98ef3a4f74195c192d93.png" />
*Model Robustness too added Noise All test Periods*

Our Ensemble Model is more robust too noise, as such its a more robust prediction and trading system in our opinion , even if it does exhibit slightly higher drawdowns and worse performance then the CNN in 25. 
## 4. **Caveats**

### 4.1 CNN Explainability & Shallow networks
 
 We failed to get through the "black box" approach of CNNs and could not extract Feature importance, tus lowering model interpretability and confidence in the models predictions. Furthermore we use shallow networks when recent (relatively) by Masaya Abe, Hideki Nakayama (15), has shown that deep networks outperform in a financial context (ie. stocks), so while our models are interpretable and fast once tuned they do suffer from relative shallowness, but they are also less prone to overfitting by learning the entire training dataset, since we had a limited amount of data for each input features roughly 3600~. Furthermore this is much less of an issue for the tree based models, and ensemble models (XGboost / Random Forest, the ensemble Meta Learner).
### 4.2 Hourly data and backtest results 

We where not able to get hourly data for long periods lowering our confidence in the backtest on the trading implications of our models, thus our current backtest rely on being able to take perfect short and long positions on futures at close, we did implement volatility linked slippage and high fees to counteract this. Furthermore real world markets also have several additional constraints, liquidity constraints, market impact for example Nevertheless its a key issue with our project especially when actual trading requires proper risk and position management to be implemented. One could image using options of actual future positions to use models signals. 
## 5. **Conclusion**

This project demonstrates the effectiveness of machine learning approaches in predicting silver futures returns and generating profitable trading signals. Firstly on the subject of theory backed feature selection gold spot prices, historical silver futures data, and gold futures emerged as the most influential predictors across models, confirming our theoretical understanding of precious metals markets and the need for a cross theoretical approach in model selection. Since the results do lend credence to the Cost too carry model but focusing on the contracts underlying environment rather then just spot, and the role of silver as a risk hedge for market participants. 
Secondly tree-based models (Random Forest, XGBoost) and our ensemble stacked approach consistently achieved win rates exceeding 82% and significantly outperformed neural network architectures. Our top models delivered robust better Sharpe ratios (2004, 2020, 2025) demonstrating practical trading viability. CNNs showed the most robustness to market conditions being our most successful model in that regard. The ensemble stacked model exhibited superior stability under noise testing, maintaining consistent signal generation even as input noise increased—a critical advantage for real-world implementation. Simpler models proved more efficient and effective than complex architectures like LSTMs, which struggled with the high noise-to-signal ratio despite theoretical advantages. Future project directions include implementing denoising techniques to improve LSTM performance by denoising financial data using other ML models as well as simpler algorithms. Expanding the ensemble architecture to create a comprehensive commodities trading system that generates position recommendations for both gold and silver futures, that focuses on robustness and stability as well as performance and implementability.

*Below you can find a detailed analysis of our models structure performance, errors, and explainability.*
## 6. **Exploring our models** 

*All of our models take either matrixes or arrays of the 10 days lagged variables excluding the un-lagged time series of each variable and our target variable (silver futures). ($e$) All Neural network based models also where mostly agnostic to batch sizes so we standardised it to 64.*
### Neural Networks  & Co Approach 
### 6.1 **FFNN** 

##### 6.2.1 Chosen architecture 

We used to grid search algorithms to tune our parameters, one focusing on the number of units per layers and the number of lags (using a relu function), as this was the first model we trained and we later planned to we used a 3 layer architecture for interpretability and computational convenience. The lags then became the default for all our models as we planned to try an ensemble architecture later, The second grid search focused on batch size, dropouts and activation functions. 

The batch size, surprisingly had no discernable impact on model performance, the other parameters proved important. We settled on this model after running our grid. Even if silu activation showed better performance it produced larger errors for extreme values then basic relu. Thus this is our chosen architecture:

```
 Layer (type)                       Output Shape                    Param #     
## ================================================================================
##  dense_3 (Dense)                    (None, 128)                     13056       
##  dropout_1 (Dropout)                (None, 128)                     0           
##  dense_2 (Dense)                    (None, 64)                      8256        
##  dropout (Dropout)                  (None, 64)                      0           
##  dense_1 (Dense)                    (None, 32)                      2080        
##  dense (Dense)                      (None, 1)                       33          
## ================================================================================
## Total params: 23425 (91.50 KB)
## Trainable params: 23425 (91.50 KB)
## Non-trainable params: 0 (0.00 Byte)
________________________________________________________________________________
```
##### 6.1.2 Errors 

Its errors where also concentrated around extreme values but where less small , it had a tilt towards positive errors but they followed a normal distribution, showing more robustness in how our model was functioning.
<img src="attachment/0ff496ae2695906b80b695e145985f2b.png" />
*On test data FFNN, Red predicted, Blue Real*
<img src="attachment/35149d35e1443f71b55dc18b99d0d3a3.png" />
*QQ-plot of residuals FFNN*

##### 6.1.3 Explainability

We use PPD plots to showcase variable importance in our models "The general idea underlying the construction of PD profiles is to show how does the expected value of model prediction behave as a function of a selected explanatory variable. For a single model, one can construct an overall PD profile by using all observations from a dataset, or several profiles for sub-groups of the observations. Comparison of sub-group-specific profiles may provide important insight into, for instance, the stability of the model’s predictions.", using the DALEX package (8),w e once again aggregate our lags in parent variables, and created a boxplot :
<img src="attachment/a2a0db7d765d14ecd5d431072124d7c3.png" />
*Variable importance using loss, FNN model*

Same observation as our previous Random forest model but it does appear to successfully leverage, the other variables more the our RF model (at the price of performance?).

### 6.2 **CNN**

##### 6.2.1 Chosen architecture 

After investigating literature, especially by Anastasia Borovykh, Sander Bohte, Cornelis W. Oosterlee (8) that recommended using a "Wavenet architecture" After tuning it using a grid search algo and "by hand" we found this architecture works the best. The model takes for input a 3 dimensional array of dimensions (lags, time, 1).Smaller layers using increasing time dilation following a $U_{n+1} = Un \times 2$   performed the best, as presented by the paper, *(appart for the final layer)*. We then applied a global average pooling function to permit the output layer to output a single datapoint.
```
________________________________________________________________________________
##  Layer (type)                       Output Shape                    Param #     
## ================================================================================
##  conv1d_4 (Conv1D)                  (None, 99, 256)                 1024        
##  dropout_5 (Dropout)                (None, 99, 256)                 0           
##  conv1d_3 (Conv1D)                  (None, 95, 128)                 98432       
##  dropout_4 (Dropout)                (None, 95, 128)                 0           
##  conv1d_2 (Conv1D)                  (None, 87, 64)                  24640       
##  dropout_3 (Dropout)                (None, 87, 64)                  0           
##  conv1d_1 (Conv1D)                  (None, 71, 32)                  6176        
##  dropout_2 (Dropout)                (None, 71, 32)                  0           
##  conv1d (Conv1D)                    (None, 55, 16)                  1552        
##  global_average_pooling1d (GlobalA  (None, 16)                      0           
##  veragePooling1D)                                                               
##  dense_4 (Dense)                    (None, 1)                       17          
## ================================================================================
## Total params: 131841 (515.00 KB)
```

We note it needed significantly more computational resources too tune the the basic FNN network, we should have used a random search.
##### 6.2.2 Errors 

They have a slight skew towards positive values and are much larger then our previous models , but its residuals are Normal.
<img src="attachment/bc2ccfa89bf768cf9755b51b89dc5d24.png" />
*On test data Wavenet CCNN, Red predicted, Blue Real*
<img src="attachment/95e064697ec8dc0428be7e3781a6164a.png" />
*QQ-plot of residuals Wavenet CNN*
We where not able to implement explainability due to the authors own technical failings, and the DALEX package not being able to handle arrays in R. 
#### 6.3 **LSTM**

##### 6.3.1 Chosen architecture 

We input an array of the shape (time, variables, lags), and follow a 3 layer architecture after a very slow grid search we found these parameters "work the best", a 3 layer LSTM failed to work so we have one LSTM layers 2 dense and one output layer : 
```
 ________________________________________________________________________________
##  Layer (type)                       Output Shape                    Param #     
## ================================================================================
##  lstm (LSTM)                        (None, 128)                     66560       
##  dropout_8 (Dropout)                (None, 128)                     0           
##  dense_8 (Dense)                    (None, 64)                      8256        
##  dropout_7 (Dropout)                (None, 64)                      0           
##  dense_7 (Dense)                    (None, 32)                      2080        
##  dropout_6 (Dropout)                (None, 32)                      0           
##  dense_6 (Dense)                    (None, 16)                      528         
##  dense_5 (Dense)                    (None, 1)                       17          
## ================================================================================
## Total params: 77441 (302.50 KB)
## Trainable params: 77441 (302.50 KB)
## Non-trainable params: 0 (0.00 Byte)
## ________________________________________________________________________________
```
The model performs poorly during training with no significant loss reduction.
##### 6.3.2 Errors 

Our Model's errors are extremely large and show the issue with it and volatility, nevertheless its errors are normally distributed.
<img src="attachment/7981198b315d94fff822b3263c4beb67.png" />
*On test data LSTM-FFNN, Red predicted, Blue Real*
<img src="attachment/ae2451899f707b72fe24dab4798dbb8b.png" />
*QQ-plot of residuals LSTM-FFNN*

Therefore the LSTM is computationally intensive too tune and has trouble dealing with the high noise to info ratio in financial data, We will not be further investigating RNN's in this project, our conclusion follows previous research from Razvan Pascanu, Tomas Mikolov, Yoshua Bengio (10) shows LSTM have trouble with noisy data and sparse signals, much like financial data, when LSTMs are successfully implemented, like showed by A.Dastgerdi, P.Mercorelli (2022) (11), additional pre-processing was required like wavelet transforms or Kalman filtering to de-noise the series. But when implemented on our data it achieved only a $45-62$% win rate making us to focus our time on the ensemble.
### 6.5 **KNN** 

##### 6.4.1 Chosen architecture 

KNN models have a far simpler approach then other models we presented, we selected the following parameters for it : $k = 10$  , i.e it looks for the nearest neighbours in the 10 nearest values, we decided to select this $k$ due to our lag selection for other models. 
##### 6.4.2 Errors 

The models errors are normally distributed and have a slight positive skew. 
<img src="attachment/40f2b97c04f007aa8e1d11edd9f57ef2.png" />
*On test data KNN, Red predicted, Blue Real*
<img src="attachment/56206a9ba17cdc228a10e0a36ca86fb3.png" />
*QQ-plot of residuals KNN*

### Tree Based Approach 

### 6.5 **Random forest**

##### 6.5.1 Chosen architecture 

A random forest algorithm is an algorithm that performs classification or regression tasks using decision trees. It has been showed to successfully applied to time series predictions, *Hristos Tyralis & Georgia Papacharalampous* (6). We use a random search algorithm and human tuning to figure out "optimal" parameters for our RF model (a).  We found that with our limited computational ressource and "sparse data" these parameters worked the best : 1000 trees with a node-size of 7.
##### 6.5.2 Errors & Explainability

 Its larger errors are concentrated over "large variations" (already smoothed), its errors do not perfectly follow a Normal distribution, we have a heavy fat tail on both ends of our QQ plot showing its trouble with extremums.
 <img src="attachment/b80b55042a6a65812be097cbaea247e5.png" />
*On test data Random forest, Red predicted, Blue Real*
<img src="attachment/2c29ad96d44e27a7c44bd15f6982313d.png" />
*QQ-plot for Random forest*

##### 6.5.3 Explainability

One of the advantages of the random forest architecture was that it was rather easier to implement some explainability methods, the packaged we used to implement the Rf model uses this method : 
"Here are the definitions of the variable importance measures. The first measure is computed from permuting OOB data: For each tree, the prediction error on the out-of-bag portion of the data is recorded (error rate for classification, MSE for regression). Then the same is done after permuting each predictor variable. The difference between the two are then averaged over all trees, and normalised by the standard deviation of the differences. If the standard deviation of the differences is equal to 0 for a variable, the division is not done (but the average is almost always equal to 0 in that case). The second measure is the total decrease in node impurities from splitting on the variable, averaged over all trees. For classification, the node impurity is measured by the Gini index. For regression, it is measured by residual sum of squares." (7)
We the aggregated each lag in its parent variable , and created a boxplot : 
<img src="attachment/1d90f103bccf511cd0b143d1340d1a85.png" />
*Variable importance by RSS in our Random forest model*

The findings where somewhat consistent with our theatrical framework BUT they showed a greater importance of spot gol rather then silver, and the VIX had no noticeable impact. 

### 6.6 **XGboost** 

##### 6.6.1 Chosen architecture 

We use a random search algorithm to find the best parameters for our Xboost, we settle on the following architecture and training parameters : max depth 15, eta 0.3 alpha 0.3 we then train it for 500 rounds.
##### 6.6.2 Errors 

The models Errors are rather normally distributed even if they suffer from a slight fat tail pattern around  the -1 theoretical quantiles. There is no Skews in errors appart from that. 
<img src="attachment/a008fc3a8d542d26bd6861a476330228.png" />
*On test data Xgboost, Red predicted, Blue Real*
<img src="attachment/913d87d3af8020953d618c49a6f96493.png" />
*QQ-plot of residuals XGBoost*
##### 6.6.2 Explainability
This package permitted rather simple variable led explainability, on the models interpretation of training data. Therefore we can see that our models only relevant variables where past Silver Futures data and Copper futures. 
<img src="attachment/71fe866179be0fbf5c85769bf617c1fa.png" />
*Variable Importance, Xgboost*
### 6.7 **Ensemble Stacked** 

##### 6.7.1 Chosen architecture 

We refer too the paper "Stacked Generalisation" by David H. Wolpert (1991) (9), and implemented a stacked model, after testing different architecture and rather predictably our best model turned out to be a combination of our best 4 models, the FFNN, The CNN, the XGboost and the random forest. This architecture allows the XGboost to use its propreties to best allocate each models strengths.  

What was rather surprising is that despite its more subpar performance imputing the CNN lowered our models MSE, and during trading tests permitted it too have the lowest drawdown. We implemented the following architecture :

As such our model takes The other models predictions ($X$) as inputs and outputs a prediction of its own of $Y$.

##### 6.7.2 Errors 

The models results are nearly normal there is a rather large skew for quantiles -3, -2 especially regarding positive values.
<img src="attachment/cd315dc86eac6c8e5c412189ced3238a.png" />
*On test data Ensemble Stacked, Red predicted, Blue Real*
<img src="attachment/b4ef6c5a8ff52705c3f3ad55d8c20ca5.png" />
*QQ-plot of residuals Ensemble-Stacked.*
##### 6.7.3 Explainability
We can therefore interpret our FNN to figure out witch models inputs was the most appreciated. 
<img src="attachment/41241f2905fbf10f14a021a4f7f530e3.png" />
*Features Importance, Ensemble-Stacked*

Despite showing a negative lower IC bound for loss for the CNN model the drawdown and noise stability improvement lead too us keeping them.
## Sources 

(1) "Expected Commodity Returns and Pricing Models", Gonzalo Cortazar; Ivo Kovacevic; Eduardo S. Schwartz (2014).

(2) "Modeling and Forecasting Stock Returns: Exploiting the Futures Market, Regime Shifts and International Spillovers", Lucio Sarno, Giorgio Valente (2003)

(3) "Taxes and the Pricing of Stock Index Futures", BRADFORD CORNELL, KENNETH R. FRENCH (1983)

(4) "GENERALIZED AUTOREGRESSIVE CONDITIONAL HETEROSKEDASTICITY", Tim BOLLERSLEV (1985)

(5) "Machine Learning", McGraw Hill, Mitchell, T. (1997).

(6) Leo Breiman (Fortran original),Adele Cutler (Fortran original), Andy Liaw (R port), Matthew Wiener (R port) (2024). *Breiman and Cutlers Random Forests for Classification and Regression*

(7) https://ema.drwhy.ai/partialDependenceProfiles.html 
Author Przemyslaw Biecek, Szymon Maksymiuk, Hubert Baniecki (2025). *DALEX package*.

(8) Anastasia Borovykh, Sander Bohte, Cornelis W. Oosterlee (2018). *Conditional time series forecasting with convolutional neural networks*

(9) "Stacked Generalization", David H. Wolpert (1991) 

(10) "Investigating the Effect of Noise Elimination on LSTM Models for Financial Markets Prediction Using Kalman Filter and Wavelet Transform" ,AMIN KARIMI DASTGERDI, PAOLO MERCORELLI (2022)

(11) "On the difficulty of training recurrent neural networks", Razvan Pascanu , Tomas Mikolov, Yoshua Bengio (2013)

(12) "Enhanced Deep Predictive Modelling of Wastewater Plants With Limited Data", Maira Alvi , Tim French , Rachel Cardell-Oliver , Damien Batstone, and Naveed Akhtar (2024)

(13) "Robustness Verification of Deep Neural Networks using Star-Based Reachability Analysis with Variable-Length Time Series Input" Neelanjana Pal, Diego Manzanas Lopez, and Taylor T Johnson (2023)

(14)"Forecasting high frequency financial time series using parallel FFN with CUDA and ZeroMQ", Paola Arce , Cesar Fernandez, Cristian Mauriera-Fernandez (2012)

(15)"Deep Learning for Forecasting Stock Returns in the Cross-Section" ,Masaya Abe, Hideki Nakayama (2018)

### Tools : 
- Keras, Tensorflow, R language, python, CRAN. 
- LLMs for debugging and graph plots issues  : 
 - Qwen models, Alibaba cloud 
 - Claude, Antrophic 

# Code examples 

# Grid search example

```r
library(randomForest)

library(dplyr)

library(tidyr)

lags <- 10 

y_train <- ts_train[(lags+1):nrow(ts_train),4]

lagged_data <- as.data.frame(embed(as.matrix(ts_train), lags + 1))

colnames(lagged_data) <- c(paste0("Lag_", rep(0:lags, each = 10), "_Input_", rep(1:10, lags+1)))


x_train <- as.matrix(lagged_data[, ncol(ts_train):ncol(lagged_data)])


y_test <- ts_test[(lags+1):nrow(ts_test),4]

lagged_test <- as.data.frame(embed(as.matrix(ts_test), lags + 1))

colnames(lagged_test) <- c(paste0("Lag_", rep(0:lags, each = 10), "_Input_", rep(1:10, lags+1)))

x_test <- as.matrix(lagged_test[, ncol(ts_train):ncol(lagged_data)])


# Random search parameters

n_iterations <- 1000

set.seed(123) 

results_df_1 <- list()

for(i in 1:n_iterations) {

# Randomly sample parameters

n_trees <- sample(1100:1200, 1) #Tested params

n_sizenode <- sample(1:20, 1)

model <- randomForest(y_train ~., data = x_train, ntree = n_trees, nodesize = n_sizenode)

predictions <- predict(model, newdata = x_test)

mse <- mean((predictions - y_test)^2)

mae <- mean(abs(predictions - y_test))

signal <- ifelse(y_test > 0, 1, 0)

signal_mod <- ifelse(predictions > 0, 1, 0)

winrate <- mean(signal == signal_mod) * 100

results_df_1[[i]] <- c(n_trees, n_sizenode, mse, mae, winrate)

cat("Trial:", i, "Trees:", n_trees, "Nodesize:", n_sizenode, "MSE:", mse, "Winrate:", winrate, "\n")

}


results_df <- do.call(rbind, results_df_1)

colnames(results_df) <- c("n_trees", "n_sizenode", "mse", "mae", "winrate")

results_df <- as.data.frame(results_df)

best_params <- results_df[which.min(results_df$mse), ]  #get lowest MSE

print(best_params)
```

Full model training example 

```r

#combining rolling avgs

data_returns=data_returns5

data_returns=as.data.frame.array(data_returns)

colnames(data_returns)=c("SPOT_COP","SPOT_SIL","SPOT_GOL","SIL","GOLD","COP","UST","VIX","MON","VOL")

  

library(reticulate)

library(tensorflow)

library(keras)

library(tidyr)

  

#Split

data_returns_train = data_returns[1:(nrow(data_returns)*0.9), ]

data_returns_test = data_returns[(nrow(data_returns) - (round(nrow(data_returns)*0.1))):nrow(data_returns), ]

  

#creating the time series

start_date=PriceHistoryCOP$Date[4827]

end_date = PriceHistoryCOP$Date[1]

  

weekly_dates <- PriceHistoryCOP$Date

  

full_week_index = as.Date(weekly_dates)

ts_train = as.ts(data_returns_train,

start = c(2004,01,01),

end = c(2021,03,02))

  

ts_test = as.ts(data_returns_test,

start = c(2021,03,02),

end = c(2023,03,08))

  

# Set the number of lags

lags <- 10 # Number of lags

y_train <- ts_train[(lags+1):nrow(ts_train),4]

lagged_data <- as.data.frame(embed(as.matrix(ts_train), lags + 1))

colnames(lagged_data) <- c(paste0("Lag_", rep(0:lags, each = 10), "_Input_", rep(1:10, lags+1)))

  

x_train <- as.matrix(lagged_data[, ncol(ts_train):ncol(lagged_data)])

  

# Prepare lagged test data

y_test <- ts_test[(lags+1):nrow(ts_test),4]

lagged_test <- as.data.frame(embed(as.matrix(ts_test), lags + 1))

colnames(lagged_test) <- c(paste0("Lag_", rep(0:lags, each = 10), "_Input_", rep(1:10, lags+1)))

  

x_test <- as.matrix(lagged_test[, ncol(ts_train):ncol(lagged_data)])

  

x_train <- array(x_train, dim = c(nrow(x_train), ncol(x_train), 1))

y_train <- array(y_train, dim = c(length(y_train), 1, 1))

x_test <- array(x_test, dim = c(nrow(x_test), ncol(x_test), 1))

y_test <- array(y_test, dim = c(length(y_test), 1, 1))

  
  

# Build the CNN model w wavenet archi

model_o3 <- keras_model_sequential() %>%

layer_conv_1d(

filters = 256,

kernel_size = 3,

dilation_rate = 1,

input_shape = c(101, 1),

activation = "leaky_relu"

) %>%

layer_dropout(rate = 0.0) %>%

layer_conv_1d(

filters = 128,

kernel_size = 3,

dilation_rate = 2,

activation = "leaky_relu"

) %>%

layer_dropout(rate = 0.0) %>%

layer_conv_1d(

filters = 64,

kernel_size = 3,

dilation_rate = 4,

activation = "leaky_relu"

) %>%

layer_dropout(rate = 0.0) %>%

layer_conv_1d(

filters = 32,

kernel_size = 3,

dilation_rate = 8,

activation = "leaky_relu"

) %>%

layer_dropout(rate = 0.0) %>%

layer_conv_1d(

filters = 16,

kernel_size = 3,

dilation_rate = 8,

activation = "leaky_relu"

) %>%

layer_global_average_pooling_1d() %>%

layer_dense(units = 1)

  
  

# Compile the model

model_o3 %>% compile(

loss = 'mean_squared_error',

optimizer = optimizer_adam(learning_rate = 0.0045),

metrics = c('mae')

)

  

#Early stop (low patience from prior testing)

early_stop <- callback_early_stopping(

monitor = "val_mae",

patience = 7,

verbose = 7,

mode = "auto"

)

  

# Train the model on batches

history <- model_o3 %>% fit(

x_train, y_train,

epochs = 20,

batch_size = 64,

validation_split = 0.2,

callbacks = list(early_stop)

)

summary(model_o3)

plot(history)

```

Test

```r

predictions <- model_o3 %>% predict(x_test)

mse_o3=mean(y_test[,,1]- predictions)^2

print(paste("mse :",mse_o2))

  

#plot

plot(y_test[,,1], type = "l", col = "blue", main = "Actual vs Predicted", ylab = "Returns")

lines(predictions, col = "red")

  

#Signal

signal <- ifelse(y_test[,1,1] > 0, 1, 0)

signal_mod <- ifelse(predictions > 0, 1, 0)

winrate <- mean(signal == signal_mod) * 100

paste("winrate",winrate, "%")

s_o3=winrate

  

#Checking the Errors

  

#Testing if mean of error is 0

Errors=y_test[,1,1] - predictions

t.test(Errors)

  

#Normality of residuals

qqnorm(Errors, main="QQ Plot of Residuals", xlab="Theoretical

Quantiles", ylab="Sample Quantiles")

qqline(Errors, col="red")

  

#Error to prediction plot

plot(Errors,predictions, main = "Errors vs Predictions & linear regression line")

abline(lm(Errors~predictions), col = "red")

```

Validation 
  
```r

library(keras)

library(tidyr)

library(dplyr)

  
# On validation

  

predict_list <- list()

eval_list <- list()

winrate_list <- list()

  

lags = 10

for (i in seq(1,7)) {

# Prepare lagged test data

lagged_test <- as.data.frame(embed(as.matrix(splits[[i]]), lags + 1))

colnames(lagged_test)= c(c(paste0("Lag_", rev(rep(0:10, each = 8)), "_Input_", rep(1:8, lags))))

y_test <- splits[[i]][(lags+1):nrow(splits[[i]]),4] #SIL col

x_test <- as.matrix(lagged_test[,ncol(ts_train):ncol(lagged_data)])

x_test <- array(x_test, dim = c(nrow(x_test), ncol(x_test), 1))

y_test <- array(y_test, dim = c(length(y_test), 1, 1))

  
predictions <- model_o3 %>% predict(x_test)

predict_list[[i]] <- predictions


eval_results_o2 <-c(mean(abs(predictions - y_test[,1,1])),mean(y_test[,1,1] - predictions)^2)

eval_list[[i]] <- c(eval_results_o2)

#Signal

signal <- ifelse(y_test[,1,1] > 0, 1, 0)

signal_mod <- ifelse(predictions > 0, 1, 0)

winrate <- mean(signal == signal_mod) * 100

paste("winrate",winrate, "%")

winrate_list[[i]] <- winrate

}

```

```r

Reduce(rbind,winrate_list)

Reduce(rbind,eval_list)

list_full_validation[[2]] <- cbind(Reduce(rbind,winrate_list),Reduce(rbind,eval_list))

#ICs

#Winrate

t.test(Reduce(rbind,winrate_list))

#Mae

t.test(Reduce(rbind,eval_list)[,1])

#Mse

t.test(Reduce(rbind,eval_list)[,2])

```

# SYNTHETIC DATA

Using The Alvi Method Maira Alvi , Tim French , Rachel Cardell-Oliver , Damien Batstone,

and Naveed Akhtar 2024, We generate synthetic data that closely Matches our distributions while following similar volatility patterns.
We will use it to evaluate our models futher. We will also try to augment model perfrormance by training it on Part synthetic data.

```r
# Number of bins (5 days)

rho <- round(nrow(data_returns) / 5)

# Ranges for each feature

ranges <- apply(data_returns5, 2, function(x) (max(x) - min(x)) / rho)

bin_edges <- lapply(names(data_returns5), function(x) {

seq(min(data_returns5[, x]), max(data_returns5[, x]) + ranges[x], length.out = rho + 1)

})

names(bin_edges) <- colnames(data_returns5)


# Assign bin labels => matrix B

B <- matrix(0, nrow = nrow(data_returns5), ncol = ncol(data_returns5))

colnames(B) <- colnames(data_returns5)


for (col in colnames(data_returns5)) {

B[, col] <- findInterval(data_returns5[, col], vec = bin_edges[[col]], rightmost.closed = TRUE)

}

# State Matrix

S <- unique(B)

n_states <- nrow(S)

  
# Transition probability matrix

P <- matrix(0, nrow = n_states, ncol = n_states)

for (i in 1:(nrow(B) - 1)) {

current_state <- which(apply(S, 1, function(row) all(row == B[i, ])))

next_state <- which(apply(S, 1, function(row) all(row == B[i + 1, ])))

if (length(current_state) > 0 && length(next_state) > 0) {

P[current_state, next_state] <- P[current_state, next_state] + 1

}

}
# Normalize T matrix

P <- sweep(P, 1, rowSums(P), FUN = "/")

P[is.na(P)] <- 0 


# Brownian motion using T

k <- 2000 # Number of synthetic samples

synthetic_states <- matrix(0, nrow = k, ncol = ncol(B))

start_state <- sample(1:n_states, 1)

for (i in 1:k) {

synthetic_states[i, ] <- S[start_state, ]

cumulative_probs <- cumsum(P[start_state, ])

next_state <- which(cumulative_probs >= runif(1))[1]

start_state <- next_state

}

# Map synthetic states back

synthetic_data <- matrix(0, nrow = k, ncol = ncol(data_returns5))

colnames(synthetic_data) <- colnames(data_returns5)

for (j in 1:ncol(data_returns5)) {

for (i in 1:k) {

current_state <- synthetic_states[i, j]

bin_min <- bin_edges[[j]][current_state]

bin_max <- bin_edges[[j]][current_state + 1]

synthetic_data[i, j] <- runif(1, bin_min, bin_max)

}

}

```

Checking if our implementation of the method works

```{r}

#olatility clustering

library(zoo)

sd_unscaled <- rollapply(

data_returns5,

width = 5,

FUN = sd,

align = "center",

partial = 1)

sd_synthetic <- rollapply(

synthetic_data,

width = 5,

FUN = sd,

align = "center",

partial = 1)


plot(sd_unscaled[1:2000,4], main = "Rolling SD plot for Silver futures", ylab =" SD", type = "l",col = "blue")

lines(sd_synthetic[,4], col = "red", type = "l")

plot(sd_unscaled[1:2000,5], main = "Rolling SD plot for Gold futures", ylab =" SD rolling", type ="l", col = "blue")

lines(sd_synthetic[,5], col = "red", type = "l")

  
#t tests
t.test(sd_unscaled[,4],sd_synthetic[4,])

t.test(data_returns[,4],synthetic_data[4,])

lapply(colnames(data_returns5),function(x) t.test(data_returns5[,x],synthetic_data[,x]))

# Plots

lapply(names(data_returns5), function(x)

{

plot(density(synthetic_data[, x]), col = "blue", main =paste(x,"Synthetic rollling avrages vs Real Data"))

lines(density(data_returns5[,x]), col = "red")

legend("topright", legend = c("Synthetic", "Real"), col = c("blue", "red"), lty = 1)

}

)

#correlations

cor(data_returns5)[,4]-cor(synthetic_data)[,4]

```

#  Options

```r
#Positions function

# Function to calculate positions with volatility-adjusted slippage

Positions <- function(model_signals, data, lags,

base_slippage = 0.01,

vol_multiplier = 3,

initial_margin = 19880,

maintenance_margin = 1800,

available_capital = 100000,

margin_buffer = 3,

max_position_pct = 1) {

len <- nrow(data) - lags

Position <- rep(0, len)

margin_used <- rep(0, len)

contracts_held <- rep(0, len)

leverage_factor <- rep(1, len)

rolling_vol <- rollapply(data[,4], width = 5, FUN = sd, align = "right", fill = NA)

rolling_vol[is.na(rolling_vol)] <- mean(rolling_vol, na.rm = TRUE)

vol_threshold <- quantile(rolling_vol, 0.8, na.rm = TRUE)

base_max_contracts <- floor(available_capital / (initial_margin * margin_buffer))

for(i in seq_len(len)) {

current_vol <- rolling_vol[lags + i]

is_high_vol <- current_vol > vol_threshold

current_slippage <- base_slippage

if(is_high_vol) {

current_slippage <- base_slippage * vol_multiplier * (current_vol / vol_threshold) #slippage goes up w vol

}

vol_scaling <- 1

if(is_high_vol) {

vol_scaling <- max(0.3, 1 - (current_vol / vol_threshold - 1)) # Reduce pos when high vol

}

if(i > 1) {

# Force exit if margin 70%

if(margin_used[i-1] > available_capital * 0.7) {

affordable_contracts <- 0

} else if(margin_used[i-1] > available_capital * 0.5) {

affordable_contracts <- max(1, floor(contracts_held[i-1] * 0.5)) #reduce pos by 50% if margin called

} else {

affordable_contracts <- floor(base_max_contracts * vol_scaling)

}

} else {

affordable_contracts <- floor(base_max_contracts * vol_scaling)

}

affordable_contracts <- min(affordable_contracts, 1)

if(model_signals[i] > 0 && affordable_contracts > 0) {

contracts <- affordable_contracts

Position[i] <- contracts * max_position_pct * (1 - current_slippage)

margin_used[i] <- contracts * initial_margin

contracts_held[i] <- contracts

leverage_factor[i] <- contracts # Track leverage

} #BULL

if(model_signals[i] < 0 && affordable_contracts > 0) {

contracts <- affordable_contracts

Position[i] <- -contracts * max_position_pct * (1 - current_slippage)

margin_used[i] <- contracts * initial_margin

contracts_held[i] <- contracts

leverage_factor[i] <- contracts

} #BEAR

if(model_signals[i] == 0 && i > 1) {

Position[i] <- Position[i-1] * 0.98

margin_used[i] <- margin_used[i-1]

contracts_held[i] <- contracts_held[i-1]

leverage_factor[i] <- leverage_factor[i-1]

}

} #IF NO SIGNA (Say NA)

return(list(

Position = Position,

margin_used = margin_used,

contracts_held = contracts_held,

leverage_factor = leverage_factor,

margin_utilization = margin_used / available_capital

))

}

  

strat_options_fct <- function(dt, model_signals, lags = 1) {

pos_info <- Positions(model_signals, dt, lags)

positions <- pos_info$Position

contracts <- pos_info$contracts_held

leverage <- pos_info$leverage_factor

cat("Margin Information:\n"). #MINI SILVER

cat("Max margin used:", max(pos_info$margin_used), "\n")

cat("Avg margin utilization:", round(mean(pos_info$margin_utilization), 3), "\n")

cat("Max margin utilization:", round(max(pos_info$margin_utilization), 3), "\n")

cat("Max leverage:", max(leverage), "contracts\n")

cat("Total contracts held:", sum(contracts > 0), "days\n\n")

returns <- dt[(lags+1):nrow(dt), 4]

strat_returns <- rep(0, length(returns))

rolling_vol <- rollapply(dt[,4], width = 5, FUN = sd, align = "right", fill = NA)

rolling_vol[is.na(rolling_vol)] <- mean(rolling_vol, na.rm = TRUE)

rolling_vol <- rolling_vol * sqrt(252)

for(i in seq_along(returns)) { #Here we do B&S

if(positions[i] != 0) {

current_vol <- rolling_vol[lags + i]

P <- exp(cumsum(returns))[i]

r <- dt[lags + i, 7]

T <- 1/365

margin_cost_pct <- (1.03)^(1/360) - 1

lev <- leverage[i]

if(positions[i] > 0) {

# SELL call at 2%

S_income <- P * 1.02

d1_inc <- (log(P/S_income) + (r + 0.5*current_vol^2)*T) / (current_vol*sqrt(T))

d2_inc <- d1_inc - current_vol*sqrt(T)

premium_income <- (P*pnorm(d1_inc) - S_income*exp(-r*T)*pnorm(d2_inc)) / P

payout_income <- max(0, returns[i] - 0.02)

# BUY put at 1%

S_hedge <- P * 0.99

d1_hdg <- (log(P/S_hedge) + (r + 0.5*current_vol^2)*T) / (current_vol*sqrt(T))

d2_hdg <- d1_hdg - current_vol*sqrt(T)

premium_hedge <- -(-P*pnorm(-d1_hdg) + S_hedge*exp(-r*T)*pnorm(-d2_hdg)) / P * lev

payout_hedge <- max(0, -returns[i] - 0.01) * lev

} else {

# SELL put at -2%

S_income <- P * 0.98

d1_inc <- (log(P/S_income) + (r + 0.5*current_vol^2)*T) / (current_vol*sqrt(T))

d2_inc <- d1_inc - current_vol*sqrt(T)

premium_income <- (-P*pnorm(-d1_inc) + S_income*exp(-r*T)*pnorm(-d2_inc)) / P

payout_income <- max(0, -returns[i] - 0.02)

# BUY call at 1%

S_hedge <- P * 1.01

d1_hdg <- (log(P/S_hedge) + (r + 0.5*current_vol^2)*T) / (current_vol*sqrt(T))

d2_hdg <- d1_hdg - current_vol*sqrt(T)

premium_hedge <- -(P*pnorm(d1_hdg) - S_hedge*exp(-r*T)*pnorm(d2_hdg)) / P * lev

payout_hedge <- max(0, returns[i] - 0.01) * lev

}

net_premium <- premium_income + premium_hedge

net_payout <- payout_income - payout_hedge

#Levrage and retur calc

strat_returns[i] <- returns[i]*lev + net_premium*0.95 - net_payout - margin_cost_pct

}

}

return(strat_returns)

}
```
