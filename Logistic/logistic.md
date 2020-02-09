# Native Bayes

```
https://zhuanlan.zhihu.com/p/46928319
https://zhuanlan.zhihu.com/p/44591359
https://zhuanlan.zhihu.com/p/34325602
https://zhuanlan.zhihu.com/p/33543849
```

| Algorithm description | note |
| --------------------- | ---- |
|                       |      |

# 

| Algorithm evaluation | complexity |
| -------------------- | ---------- |
| time complexity      | O()        |
| Space complexity     | O()        |

# 

## Algorithm math description:

#### Logistic Regression model:



$$  \tag{1.0} sigmoid(X) = \frac { exp(X) } { 1 + exp(X) } = \frac { 1 } { 1 + exp(-X) }  $$

$  \tag{1.1} P(y_i=0|X_i) = 1 - sigmoid( W^{T} X_i) $

$$ \tag{1.2} P(y_i=1|X_i)=sigmoid( W^{T} X_i) $$ 

$$ \tag{1.3} L(W) = \prod_{i=1}^n [ P(y_i=1|X_i) ]^{y_i} [ P(y_i=0|X_i) ]^{1 - y_i} $$ 

$$ \tag{1.4} \begin{align} ln[L(W)] &= \sum^n_{i=1} [ y_i * ln(P(y_i=1|X_i) + (1 - y_i) * ln(P(y_i=0|X_i) ] \\ &= \sum^n_{i=1} [ y_i * ln(\frac { P(y_i=1|X_i) } { P(y_i=0|X_i) } ) + ln( P(y_i=0|X_i) ) ] \\ & = \sum^n_{i=1} [ y_i * (W^{T} X_i) - ln(1 + exp(W^{T} X_i)) ]  \end{align} $$

$$ \tag{1.5} \begin{align} [y_i * (W^{T} X_i) - ln(1 + exp(W^{T} X_i))]^{'} &= y_i * X_i - \frac { exp(W^{T} X_i) * X_i } { 1 + exp(W^{T} X_i) } \\ &= [y_i - sigmoid(W^{T} X_i)] * X_i \end{align} $$

 $$ \tag{1.6} ln[L(W)]^{'}*W = [[Y - sigmoid(W^{T} X)]X^{T}]^{T}, Y \in R*{1 * m}, W \in R_{k * 1}, X \in R_{k * m} $$ $$ \tag{1.6} ln[L(W)]^{'}*W = X [Y - sigmoid(X W)], Y \in R*{m * 1}, X \in R_{m * k}, W \in R_{k * 1} $$



#### Multi-Nominal Logistic Regression model:

$$\tag{2.1} P(y_i=k|X_i) = \frac { exp(W^{T}_k X_i) } { 1 + \sum^{K-1}_k exp(W^{T}_k X_i) } $$ $$\tag{2.2} P(y_i=K|X_i) = \frac { exp(W^{T}_k X_i) } { 1 + \sum^{K-1}_k exp(W^{T}_k X_i) } $$

# 

### Algorithm extension: