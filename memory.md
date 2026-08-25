# Memory

## Context

当前目录：`D:\dabaidanzi\组内项目\weixing项目\异常检测 - 副本\AE_上实验`

用户提供的图片是代码截图，要求先读取所有图片内容，再把截图里的代码写入当前目录的 `pre.py`。后续要求把本次处理内容整理到 `memory.md`。

## 图片读取结果

当前目录下共读取到 9 张图片：

- `微信图片_20260710175549_261_3.jpg`
- `微信图片_20260710175551_262_3.jpg`
- `微信图片_20260710175553_263_3.jpg`
- `微信图片_20260710175556_264_3.jpg`
- `微信图片_20260710175558_265_3.jpg`
- `微信图片_20260710175600_266_3.jpg`
- `微信图片_20260710175601_267_3.jpg`
- `微信图片_20260710175604_268_3.jpg`
- `微信图片_20260710175605_269_3.jpg`

图片内容主要来自 `pre_nouplink.py` 和 `pre.py`，二者大致相同。另有一张图片是 `train_data` 文件夹内容截图，展示 `.dat` 文件命名格式。

示例文件名：

```text
20260425_104526_01_0.5_1_1_1_train.dat
20250624_103015_18_0.5_2_1_1_train.dat
```

## 已写入 pre.py 的内容

`pre.py` 已根据截图还原并补齐为可运行的数据预处理脚本，主要包含：

- `DEFAULT_KEEP_SIGNAL_IDS`
- `DEFAULT_ROOT`
- `DEFAULT_SAVE_ROOT`
- `parse_filename(filename)`
- `get_sat_id(tx_id, signal_id)`
- `load_dat(path)`
- `power_normalize(iq)`
- `split_segment(iq, segment_length=1280)`
- `save_dataset(save_dir, data, label, sat)`
- `_list_dat_files(root)`
- `_build_split(files, segment_length)`
- `build_dataset(...)`
- 命令行参数入口 `_parse_args()`

### 文件名解析逻辑

`parse_filename()` 按 `_` 拆分文件名，要求拆分后长度为 8。

当前逻辑：

- `signal_id = int(parts[2])`
- `uplink = int(parts[4])`
- 只保留 `uplink == 2`
- `tx = int(parts[5])`
- `sat = get_sat_id(tx, signal_id)`
- 若格式不匹配、不是下行信号、或无法映射卫星，则返回 `None`

### 卫星映射逻辑

`get_sat_id()` 当前映射：

| tx | signal_id 范围 | sat |
| --- | --- | --- |
| 1 | 1-20 | 1 |
| 1 | 21-40 | 2 |
| 2 | 41-60 | 1 |
| 2 | 61-80 | 2 |
| 3 | 81-100 | 3 |
| 3 | 101-120 | 4 |
| 4 | 121-140 | 3 |
| 4 | 141-160 | 4 |

不匹配返回 `None`。

### IQ 读取和处理逻辑

`load_dat()`：

- 使用 `np.fromfile(path, dtype=np.float64)` 读取原始数据。
- 要求长度为偶数，否则抛出 `ValueError`。
- 偶数索引作为 I，奇数索引作为 Q。
- 输出形状为 `(N, 2)`。

`power_normalize()`：

- 将 IQ 组成复数：`x = iq[:, 0] + 1j * iq[:, 1]`
- 计算平均功率：`np.mean(np.abs(x) ** 2)`
- 若功率小于 `1e-12`，直接返回原始 IQ。
- 否则除以 `sqrt(power)` 做功率归一化。
- 再拆回 `(N, 2)` 的实部和虚部。

`split_segment()`：

- 默认按 `1280` 点切片。
- 丢弃不足一个完整片段的尾部。
- 输出形状为 `(num, 1280, 2)`。

### 数据集保存逻辑

`save_dataset(save_dir, data, label, sat)` 保存：

```text
save_dir/
  data.npy
  label.npy
  sat.npy
```

其中：

- `data.npy`：切片后的 IQ 数据。
- `label.npy`：当前按 `signal_id` 保存。
- `sat.npy`：卫星编号。

### build_dataset 补齐逻辑

截图没有完整覆盖 `build_dataset()` 中间部分，因此当前实现是按截图上下文补齐的最小可运行版本：

1. 遍历 `root` 下所有 `.dat` 文件。
2. 用 `parse_filename()` 过滤和提取元信息。
3. 如果提供 `keep_signal_ids`，只保留指定信号序号。
4. 使用 `seed` 固定随机顺序。
5. 按文件划分：
   - `train_ratio` 默认为 `0.6`
   - 剩余文件平均分给 `val` 和 `test`
6. 每个 split 内部再读取 `.dat`、功率归一化、按 1280 切片。
7. 保存到：
   - `save_root/train_data/data.npy`
   - `save_root/train_data/label.npy`
   - `save_root/train_data/sat.npy`
   - `save_root/val_data/...`
   - `save_root/test_data/...`

## 默认参数

`DEFAULT_KEEP_SIGNAL_IDS` 来自截图：

```python
[
    3, 4, 8, 10, 11,
    25, 26, 37, 39,
    44, 47, 49, 51, 52,
    65, 66, 70, 71, 72,
]
```

默认路径来自截图：

```text
/home/ubuntu/[260603]西安采集数据/UESTC_train_data/train_data
/home/ubuntu/[260603]西安采集数据/UESTC_train_data/dr_data
```

## 验证记录

已运行：

```powershell
python -m py_compile .\pre.py
```

结果：退出码为 0。

已运行文件名解析样例：

```powershell
python -c "import pre; assert pre.parse_filename('20250624_103015_18_0.5_2_1_1_train.dat') == {'tx': 1, 'sat': 1, 'signal_id': 18}; assert pre.get_sat_id(4, 150) == 4; print('ok')"
```

结果：输出 `ok`。

## 已发现并修正的问题

整理 `memory.md` 前重新读取 `pre.py` 时，发现当前文件缺少：

- `import argparse`
- `import numpy as np`

这两个导入已补齐。

## 仍需注意的不确定点

- 截图没有拍完整 `build_dataset()`，当前实现不是逐字符还原，而是根据已看到的函数和调用方式补齐。
- `label.npy` 当前按 `signal_id` 保存；如果后续模型训练需要 `0=normal, 1=anomaly`，需要另写转换逻辑。
- 当前项目的 `dataset.py` 读取的是 `train_data.npy / train_label.npy / test_data.npy / test_label.npy`，而 `pre.py` 当前保存的是子目录内的 `data.npy / label.npy / sat.npy`。如果要直接接入现有训练流程，需要统一数据文件命名和形状。
- `build_dataset()` 当前按文件划分 `train/val/test`，不是按切片后样本划分。
- 当前只根据截图保留 `uplink == 2`。如果 `pre.py` 和 `pre_nouplink.py` 的真实区别是是否保留上下行，需要再根据原始代码确认。

## 建议下一步

1. 用真实 `.dat` 数据运行一次 `pre.py`，检查输出 shape 和保存目录。
2. 决定 `label.npy` 是继续表示 `signal_id`，还是改成异常检测需要的二分类标签。
3. 决定是否把 `pre.py` 输出格式改成当前 `dataset.py` 可直接读取的 `train_data.npy / train_label.npy / test_data.npy / test_label.npy`。
