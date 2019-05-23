# Native Bayes
```
```

Algorithm description|note
:--|:--

#
Algorithm evaluation|complexity
:--|:--
time complexity|O()
Space complexity|O()
#
## Algorithm math description:
#### Single-Nominal Logistic Regression model:
$$\tag{1.0}
sigmoid(X)
= \frac { exp(X) } { 1 + exp(X) }
= \frac { 1 } { 1 + exp(-X) }
$$
$$\tag{1.1}
P(y_i=0|X_i) = 1 - sigmoid(W^{T} X_i)
$$
$$\tag{1.2}
P(y_i=1|X_i) = sigmoid(W^{T} X_i)
$$
$$
\tag{1.3}
L(W) = \prod_{i=1}^n
[ P(y_i=1|X_i) ]^{y_i}
[ P(y_i=0|X_i) ]^{1 - y_i}
$$
$$
\tag{1.4}
ln(L(W)) = \sum^n_{i=1} [
    y_i * ln(P(y_i=1|X_i) +
    (1 - y_i) * ln(P(y_i=0|X_i)
]
$$
$$
= \sum^n_{i=1} [
    y_i * ln(\frac
        { P(y_i=1|X_i) }
        { P(y_i=0|X_i) }
    ) + ln( P(y_i=0|X_i) )
]
$$
$$
= \sum^n_{i=1} [
    y_i * (W^{T} X_i) -
    ln(1 + exp(W^{T} X_i))
]
$$
$$
\tag{1.5}
ln(L(W))^{'}_W = 
YX_i - \frac
    { exp(W^{T} X_i) X_i }
    { 1 + exp(W^{T} X_i) }
= [Y - sigmoid(W^{T} X_i)] X_i
$$

#### Multi-Nominal Logistic Regression model:
$$\tag{2.1}
P(y_i=k|X_i) = \frac
{ exp(W^{T}_k X_i) }
{ 1 + \sum^{K-1}_k exp(W^{T}_k X_i) }
$$
$$\tag{2.2}
P(y_i=K|X_i) = \frac
{ exp(W^{T}_k X_i) }
{ 1 + \sum^{K-1}_k exp(W^{T}_k X_i) }
$$
#
### Algorithm extension: