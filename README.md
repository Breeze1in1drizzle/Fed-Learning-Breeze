# Fed-Learning-Breeze
 The fedrated learning framework I tried.

### 2022.03.07
#### 联邦学习 Simulation，需要有以下指标
#### ①收敛性指标（Loss & Accuracy non-iid case）
用高斯分布指定每个设备的数据量 datasize，
再设置某个百分比的主类 main-label
#### ②公平性指标（基于各设备出现频率）
用一个列表统计每个设备出现的频率即可
#### ③性能指标（训练耗时）
用一个随机分布，模拟每个设备每个 round 的执行时间
#### 多线程实现 or 串行
多线程更符合联邦学习并行特点，但是如果线程太多，
容易导致线程切换频繁而带来的数据吞吐量大增，
反而导致实验速度降低？
#### 策略
可以分别试一下多线程和串行的 Simulation。
上述指标，再这两种 Simulation 的情况下都可以执行