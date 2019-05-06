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
#### Native Bayes:
$$\tag{1.0} \vec X = (\vec x_1, ..., \vec x_n)$$
$$\tag{1.1} \vec x_i = (a^{(1)}_i, ..., a^{(m)}_i)$$
$$
\tag{1.2}
\boldsymbol P(\vec x_i | y_i=c_k) =
\boldsymbol P(\vec x^{(1)}, ..., \vec x^{(n)}_i | y_i=c_k) =
\prod_{j=1}^m \boldsymbol P(\vec x^{(j)}_i | y_i=c_k)
$$
$$
\tag{2.0}
\boldsymbol P(y_i=c_k | \vec x_i) =
\frac{
    \boldsymbol P(y_i=c_k)
    \boldsymbol P(\vec x_i | y_i=c_k)
}{
    \sum^m_{i=1} (
        \boldsymbol P(y_i=c_k)
        \boldsymbol P(\vec x_i | y_i=c_k)
    )
}
$$
#### Likelihood:
$$
\tag{2.1}
\boldsymbol P(y_i=c_k) =
\frac{\sum^N_{i=1} \boldsymbol I(y_i=c_k)}{N}
$$
$$
\tag{2.2}
\boldsymbol P(x^{(j)}_i = a^{(j)}_i | y_i=c_k) =
\frac{
    \sum^N_{i=1} \boldsymbol I(\boldsymbol x^{(j)}_i , y_i=c_k)
}{\sum^N_{i=1} \boldsymbol I(y_i=c_k)}
$$
$$
\tag{3.0}
Y = \boldsymbol {argmax} (\boldsymbol P(y_i=c_k | \vec x_i)
= \boldsymbol {argmax} (
    \boldsymbol P(y_i=c_k) \prod_{j=1}^n
    \boldsymbol P(x^{(j)}_i | y_i=c_k)
)
$$
#### Laplace Smooth:
$$Laplace Smooth: \lambda = 1$$
$$
\tag{4.0}
\boldsymbol P(y_i=c_k) =
\frac {\sum^N_{i=1} \boldsymbol I(y_i=c_k) + \lambda}
      {N + K \lambda}
$$
$$
\tag{4.1}
\boldsymbol P(x^{(j)}_i = a^{(j)}_i | y_i=c_k) =
\frac {\sum^N_{i=1}\boldsymbol I(\boldsymbol x^{(j)}_i , y_i=c_k) + \lambda}
      {\sum^N_{i=1} \boldsymbol I(y_i=c_k) + S_j \lambda}
$$
$$
\tag{4.2}
\sum^{S_j}_{i=1} \boldsymbol P(x^{(j)}_i = a^{(j)}_i | y_i=c_k) = 1
$$
#
### Algorithm extension: