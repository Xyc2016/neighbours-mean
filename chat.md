

Polars代码优化与性能分析
image.png
PNG 402.97KB
把这一段改写成polars
以下是使用Polars重写的代码，采用笛卡尔积连接（cross join）后进行范围过滤的方法，适用于中小型数据集。对于大型数据集，可能需要考虑优化或使用空间索引库（如GeoPandas）。

```python
import polars as pl

# 假设输入数据为small_block (pl.DataFrame)，包含列: X, Y, Value
# distance 是给定的距离参数

# 添加唯一行ID用于后续分组
small_block = small_block.with_row_count(name="row_id")

# 创建辅助点集（重命名列避免冲突）
points = small_block.select([
    pl.col("X").alias("X_point"),
    pl.col("Y").alias("Y_point"),
    pl.col("Value").alias("Value_point")
])

# 执行笛卡尔积连接并进行范围过滤
result = (
    small_block.join(points, how="cross")  # 笛卡尔积
    .filter(
        (pl.col("X_point") >= pl.col("X") - distance) &  # X下限
        (pl.col("X_point") <= pl.col("X") + distance) &  # X上限
        (pl.col("Y_point") >= pl.col("Y") - distance) &  # Y下限
        (pl.col("Y_point") <= pl.col("Y") + distance)    # Y上限
    )
    .group_by("row_id")  # 按原始行分组
    .agg(pl.col("Value_point").mean().alias("Value_new"))  # 计算均值
)

# 合并结果并清理
small_block = small_block.join(result, on="row_id").drop("row_id")
```
优化说明：
笛卡尔积处理：通过交叉连接创建所有点对组合

范围过滤：使用四边界条件快速筛选邻域点

分组聚合：按原始行分组计算邻域平均值

内存优化：使用Polars的延迟执行和并行处理优化性能

替代方案（适用于大型数据集）：
如果数据集非常大（>10万行），建议使用空间索引加速查询。以下是使用RTree的备选方案：

```python
from rtree import index
import numpy as np

# 创建空间索引
idx = index.Index()
for i, (x, y) in enumerate(zip(small_block["X"], small_block["Y"])):
    idx.insert(i, (x, y, x, y))  # 点坐标插入索引

# 计算每个点的邻域均值
values_new = []
for (x, y) in zip(small_block["X"], small_block["Y"]):
    # 查询邻域点ID
    neighbors = list(idx.intersection((x-distance, y-distance, 
                                      x+distance, y+distance)))
    # 提取邻域值并计算均值
    if neighbors:
        mean_val = small_block[neighbors]["Value"].mean()
    else:
        mean_val = None
    values_new.append(mean_val)

# 添加结果列
small_block = small_block.with_columns(pl.Series(name="Value_new", values=values_new))
```
性能比较：
方法	时间复杂度	适用场景
笛卡尔积法	O(n²)	中小数据集（<5万行）
RTree索引法	O(n log n)	大型数据集
原始Pandas法	O(n²)	小数据集（不推荐）
提示：实际使用时请根据数据规模选择合适方法。对于地理空间数据，建议先进行坐标系统一和单位标准化。

