# KNN: k-Nearest Neighbor Algorithm

```
https://www.wikiwand.com/zh/K-%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95
https://www.cnblogs.com/pinard/p/6164214.html
```

### Algorithm math description:

$$
L_{p(i)}
= (\sum_{j=1}^n |
    ~test_{(j)}-val_{(i,j)}~
| ^ p )^{\frac{1}{p}}
, ~i\in[:m], ~j\in[:n]
$$

$$
Predict
= \text{Max}(
    ~\text{Mode}(
        ~\text{Sort}(
            ~L_{p(i)}
        )~[:k]~
    )~
)
$$

| Algorithm description | note                                                         |
| :-------------------- | :----------------------------------------------------------- |
| type                  | supervised learning                                          |
| validation dataset    | the number of case and param is m, n                         |
| test case             | the number of case and param is n                            |
| step 1                | get m Lp (p = 2) distances of entire validation dataset with the case of test dataset |
| step 2                | find out the top k minimum of every Lp distance and it's corresponding labels |
| step 3                | find out largest number of duplicate label in top k labels   |
| step 4                | we get it                                                    |

| Algorithm evaluation | complexity     |
| :------------------- | :------------- |
| time complexity      | O( 8 * m * n ) |
| Space complexity     | O( m * n )     |

### Algorithm extension: kd tree