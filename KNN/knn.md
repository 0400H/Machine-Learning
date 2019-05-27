# KNN: k nearest neighbor algorithm
```
https://www.wikiwand.com/en/K-nearest_neighbors_algorithm
https://www.wikiwand.com/zh/%E6%9C%80%E8%BF%91%E9%84%B0%E5%B1%85%E6%B3%95
```

### Algorithm math description:
$$
l2\_distance_{(l,i)}
= \sum_{j=0}^n (
    ~test_{(l,j)}-validation_{(i,j)}~
)
, ~l\in[:h], ~i\in[:m], ~j\in[:n]
$$
$$
prediction_l
= \text{max}(
    ~\text{common}(
        ~\text{sort}(
            ~l2\_distance_{(l,)}
        )~[:k]~
    )~
)
$$

| Algorithm description | note                                                         |
| :-------------------- | :----------------------------------------------------------- |
| type                  | supervised learning                                          |
| validation dataset    | the number of case and param is m, n                         |
| test case             | the number of case and param is n                            |
| step 1                | get m L2-distances of entire validation dataset with the case of test dataset |
| step 2                | find out the top k minmum of every l2-distance and it's corresponding labels |
| step 3                | find out largest number of duplicate label in top k labels   |
| step 4                | we get it                                                    |

| Algorithm evaluation | complexity         |
| :------------------- | :----------------- |
| time complexity      | O( 8 * h * m * n ) |
| Space complexity     | O( (m + h) * n )   |

### Algorithm extension: kd tree