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
$$
\tag{1.0}
\boldsymbol P(\boldsymbol x_j | y=c_k) =
\boldsymbol P(\boldsymbol x^{(1)}, ..., \boldsymbol x^{(n)} | y=c_k) =
\prod_{i=1}^n \boldsymbol P(\boldsymbol x^{(i)} | y=c_k)
$$
$$
\tag{1.1}
\boldsymbol X = (\boldsymbol x_1, ..., \boldsymbol x_n)
$$
$$
\tag{1.2}
Likelihood:
\boldsymbol x_i = (\boldsymbol x^{(1)}_i, ..., \boldsymbol x^{(n)}_i)
$$
$$
\tag{2.0}
\boldsymbol P(y=c_k | \boldsymbol X) =
\frac{
    \boldsymbol P(y=c_k)
    \boldsymbol P(\boldsymbol X | y=c_k)
}{
    \sum_k (
        \boldsymbol P(y=c_k)
        \boldsymbol P(\boldsymbol X|y=c_k)
    )
}
$$
$$
\tag{2.1}
\boldsymbol P(\boldsymbol X | y=c_k) =
\prod_{i=1}^n \boldsymbol P(\boldsymbol x^{(i)} | y=c_k)
$$
$$
\tag{2.2}
Likelihood:
\boldsymbol P(y=c_k) =
\frac{
    \sum^n_{j=1} \boldsymbol I(y_j=c_k)
}{
    \sum^n_{j=1} \boldsymbol I(y_j=c_k)
}
$$
$$
\tag{2.3}
Likelihood:
\boldsymbol P(\boldsymbol x^{(i)} | y=c_k) =
\frac{
    \sum^n_{j=1} \boldsymbol I(\boldsymbol x^{(i)}_j , y_j=c_k)
}{
    \sum^n_{j=1} \boldsymbol I(y_j=c_k)
}
$$

$$
\tag{3.0}
y^{'} = \boldsymbol f(\boldsymbol X)
= \boldsymbol {argmax} (
    \boldsymbol P(y=c_k | \boldsymbol X = \boldsymbol x_{(i)})
)
= \boldsymbol {argmax} (
    \boldsymbol P(y=c_k) \prod_{i=1}^n
    \boldsymbol P(\boldsymbol x^{(i)} | y=c_k)
)
$$
### Native Bayes with Laplace Smooth:
#
### Algorithm extension: