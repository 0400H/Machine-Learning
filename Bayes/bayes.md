# Native Bayes
```
```

Algorithm description|note
:--|:--

#
Algorithm evaluation|h testcase
:--|:--
time complexity|O()
Space complexity|O()
#
### Algorithm math description:
$$
\boldsymbol P(\boldsymbol X | \boldsymbol Y=y_k) =
\boldsymbol P(\boldsymbol x^{(1)}, \boldsymbol x^{(2)},..., \boldsymbol x^{(n)} | \boldsymbol Y=y_k) =
\prod_{i=1}^n \boldsymbol P(\boldsymbol X = \boldsymbol x^{(i)} | \boldsymbol Y=y_k)
$$
$$
\boldsymbol P(\boldsymbol Y=y_k | \boldsymbol X)
=
\frac{
    \boldsymbol P(\boldsymbol Y=y_k)
    \boldsymbol P(\boldsymbol X | \boldsymbol Y=y_k)
}{
    \sum_k (
        \boldsymbol P(\boldsymbol Y=y_k)
        \boldsymbol P(\boldsymbol X|\boldsymbol Y=y_k)
    )
}
=
\frac{
    \boldsymbol P(\boldsymbol Y=y_k)
    \prod_{i=1}^n
    \boldsymbol P(\boldsymbol X = \boldsymbol x^{(i)} | \boldsymbol Y=y_k)
}{
    \sum_k (
        \boldsymbol P(\boldsymbol Y=y_k)
        \prod_{i=1}^n \boldsymbol P(\boldsymbol X = \boldsymbol x^{(i)} | \boldsymbol Y=y_k)
    )
}
$$
$$
y = \boldsymbol f(\boldsymbol X)
= \boldsymbol {argmax} (
    \boldsymbol P(\boldsymbol Y=y_k | \boldsymbol X = \boldsymbol x_{(i)})
)
= \boldsymbol {argmax} (
    \boldsymbol P(Y=y_k) \prod_{i=1}^n
    \boldsymbol P(\boldsymbol X = \boldsymbol x^{(i)} | \boldsymbol Y=y_k)
)
$$
#
### Algorithm extension: