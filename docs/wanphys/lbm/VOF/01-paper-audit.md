# 论文依据审计

本文档对照论文印刷内容审计当前讨论中的实现框架。页码使用 PDF 页码，公式编号
使用论文编号。

## 1. 总览

| 设计项 | 状态 | 论文依据或问题 |
|---|---|---|
| L/I/G 三类节点 | `PAPER` | Sec. 3.1, p.5 |
| Interface 必须位于 Liquid 与 Gas 之间 | `PAPER` | Sec. 3.1, p.5 |
| 气相不模拟，只施加界面压力 | `PAPER` | Sec. 2.2、3.1 |
| 格链质量平流 | `PAPER` | Eq. (9)-(10), p.5 |
| 仅中心 `INTERFACE` 更新质量 | `CONFLICT` | Eq. (10) 明列中心 `L/G/I` 三种权重 |
| 按邻居类型取 `1 / avg / 0` | `CONFLICT` | 与本文印刷版 Eq. (10) 的条件不一致 |
| 持久保存 `mass=rho*phi` | `DERIVED` | 文中说明 `phi=mass/rho` 且内存表列出 mass，但 Eq. (9) 直接更新 `phi` |
| 等待 streamed `rho*` 后才更新 `phi` | `CONFLICT` | Eq. (9) 分母明确是 `rho(x,t)`，不是 `rho*` |
| 钳制并重分配越界质量 | `PAPER` | Sec. 3.1, p.5 |
| 重分配的并行算法 | `REFERRED` | 只引用 Lehmann 2019 |
| 自由表面 population 重构 | `PAPER` | Eq. (11), p.5 |
| 使用气泡压力与曲率求 `rho_g` | `PAPER` | Eq. (12), Eq. (41) |
| `u_pred=(j+F/2)/rho` | `DERIVED` | 可由 Eq. (7) 推出；论文 Eq. (11) 直接使用存储的 `u` |
| `u_pred=u+F/(2rho)` | `CONFLICT` | 若 `u` 已按 Eq. (7) 闭合，会重复加入半力 |
| 用 `Pi` 表示重构应力 | `CONFLICT` | 本文 `Pi` 专指 disjoining pressure |
| PLIC 用于界面质量输运 | `CONFLICT` | Eq. (9) 负责质量平流；PLIC 用于曲率 |
| PLIC 用于法向/曲率 | `PAPER` | Eq. (12) 后说明，Sec. 5.1 |
| 几何场持久保存在 `I^n` | `UNSPECIFIED` | 论文未规定持久化；内存表未列出 |
| HOME `rho,u,S` 10 标量 | `PAPER` | Sec. 4.1, Eq. (16)-(21) |
| D3Q27 高阶 NOCM-MRT | `PAPER` | Eq. (13)-(21)，central moment 索引至 26 |
| D3Q19/SRT/TRT 作为等价论文实现 | `CONFLICT` | 论文将其作为稳定性较弱的对比或背景 |
| FullF 与 HOME 共用 VOF | `UNSPECIFIED` | 项目兼容性扩展 |
| 气泡理想气体压力 | `PAPER` | Eq. (22)-(25) |
| CCL 只在类型变化时触发 | `PAPER` | Sec. 4.2, p.7 |
| cut-cell 不参与 bubble CCL | `PAPER` | Sec. 4.2, p.7 |
| moving-solid fresh/dead | `PAPER` | Sec. 4.3, p.9 |
| 把一般 VOF 新界面也称为 fresh/dead | `CONFLICT` | 论文 fresh/dead 专指移动固体覆盖变化 |
| 通用 `DOMAIN` 路由 | `UNSPECIFIED` | 论文没有设计通用入口/出口框架 |
| 单一融合 kernel | `UNSPECIFIED` | Algorithm 1 未规定 kernel 粒度 |
| `R` 与质量交换 `X` 并行 | `DERIVED` | 可由二者只读旧状态推导，论文未明确调度 |

## 2. 质量平流：当前最大的公式差异

### 2.1 论文印刷公式

论文用格链 population 通量更新体积分数：

\[
\phi(\mathbf{x},t+1)
=
\phi(\mathbf{x},t)
+
\frac{1}{\rho(\mathbf{x},t)}
\sum_{i=0}^{q-1}
\theta(\mathbf{x})
\left[
f_{\bar i}(\mathbf{x}+\mathbf{c}_i,t)
-
f_i(\mathbf{x},t)
\right].
\tag{9}
\]

权重按论文 Eq. (10) 写为：

\[
\theta(\mathbf{x})=
\begin{cases}
1,
&\mathbf{x}\in L,
\\
0,
&\mathbf{x}\in G,
\\
\dfrac{
\phi(\mathbf{x},t)+
\phi(\mathbf{x}+\mathbf{c}_i,t)
}{2},
&\mathbf{x}\in I.
\end{cases}
\tag{10}
\]

这意味着本文印刷版不是“只处理中心 Interface”，也不是“根据邻居类型取
`1 / average / 0`”。

### 2.2 当前讨论公式

当前讨论提出：

\[
\Delta m_i(\mathbf{x})=
\begin{cases}
q_i,& type(\mathbf{x}+\mathbf{c}_i)=L,\\
\frac{\phi_x+\phi_{x+c_i}}{2}q_i,& type(\mathbf{x}+\mathbf{c}_i)=I,\\
0,& type(\mathbf{x}+\mathbf{c}_i)\in\{G,S\},
\end{cases}
\]

且只在中心节点为 `INTERFACE` 时执行。这不是论文 Eq. (9)-(10) 的同一写法。

### 2.3 实现前决策

必须二选一：

1. **严格按本文实现：**采用 Eq. (9)-(10)，用测试验证均匀液体、界面和平衡状态；
2. **采用经典 FSLBM/Lehmann 变体：**另行核验 Körner 2005、Lehmann 2019 或
   参考实现，并在代码和文档中标明来源，不引用本文 Eq. (10) 为依据。

在决策完成前，不应把当前分段公式固化为公共 API。

另外，论文 Eq. (9) 使用旧时间层：

\[
\rho(\mathbf{x},t),
\]

因此论文质量线不需要等待 streamed populations 的 `rho*`。当前图中的
`M -> PT` 同步边不受本文支持；如果保留，必须作为新的数值格式单独推导和验证。

## 3. 自由表面补全

### 3.1 论文公式

从气体方向流入界面的未知 population 按 Eq. (11) 重构：

\[
f_i^*(\mathbf{x},t)
=
f_i^{eq}(\rho_g,\mathbf{u}(\mathbf{x},t))
+
f_{\bar i}^{eq}(\rho_g,\mathbf{u}(\mathbf{x},t))
-
f_{\bar i}(\mathbf{x},t).
\tag{11}
\]

气体密度为：

\[
\rho_g
=
\frac{p_g-2\gamma\kappa(\mathbf{x})}{c_s^2}.
\tag{12}
\]

泡沫扩展加入 disjoining pressure：

\[
\rho_g
=
\frac{p_g-2\gamma\kappa(\mathbf{x})-\Pi}{c_s^2}.
\tag{41}
\]

### 3.2 对当前抽象的影响

- Eq. (11) 读取的是旧状态的 `u(x,t)` 与对向 `f_bar(x,t)`，不是当前已经
  streamed 的 `f_bar*`。
- HOME 路径中的 `f_bar(x,t)` 应由 `rho,u,S` 按 Eq. (16)-(17) 重构。
- 因为全部输入来自旧状态，论文这条自由表面规则可以逐 link 独立求值。
- 本文没有把非平衡应力记作 `Pi`；持久二阶速度矩是 `S`，而 `Pi` 是泡沫排斥压。
- 如果项目持久保存原始动量 `j` 而不是物理速度 `u`，可由 Eq. (7) 计算：

\[
\mathbf{u}
=
\frac{\sum_i \mathbf{c}_i f_i+\mathbf{F}/2}{\rho}.
\]

这属于存储表示适配。若状态已经保存物理 `u`，不得再次加 `F/2rho`。

## 4. HOME 与碰撞

### 4.1 论文支持

论文不持久保存全部 D3Q19/D3Q27 populations，而保存：

\[
\rho,\qquad
\mathbf{u},\qquad
\mathbf{S},
\]

合计：

\[
1+3+6=10
\]

个标量。Streaming 时按 Eq. (16)-(17) 做三阶 filtered Hermite reconstruction。

论文采用 non-orthogonal central-moment MRT：

\[
\boldsymbol{\Omega}
=
-\mathbf{M}^{-1}
\left[
\mathbf{R}(\mathbf{m}-\mathbf{m}^{eq})
+
\left(\mathbf{I}-\frac12\mathbf{R}\right)\mathbf{K}
\right].
\tag{13}
\]

Eq. (18)-(21) 给出碰撞后 HOME moments。Eq. (19) 明确：

\[
u_\alpha(\mathbf{x},t+1)
=
u_\alpha^*
+
\frac{F_\alpha}{2\rho^*}.
\tag{19}
\]

### 4.2 与当前仓库的差异

当前仓库主格子是 D3Q19，并支持 FullF/HOME、SRT/TRT/Raw-MRT/NOCM-MRT。
这些路径可以共享 VOF 的体积分数和压力边界，但只有与论文相同的高阶 HOME
重构和 central-moment collision 才能主张复现论文的稳定性设计。

因此文档与测试应区分：

```text
VOF-compatible
    能运行 Eq. (9)-(12) 的自由表面算法

HOME-FREE-paper-fidelity
    同时满足论文的 HOME moment、filtered reconstruction 和高阶碰撞
```

## 5. 类型转换与质量重分配

### 5.1 论文直接说明

论文 Sec. 3.1 说明：

- `I` 节点满足 `0 < phi < 1`；
- `I` 应位于 `L` 和 `G` 之间；
- 当 `phi >= 1 + epsilon_phi`，`I -> L`；
- 当 `phi <= 0 - epsilon_phi`，`I -> G`；
- 论文使用 `epsilon_phi = 1e-4`；
- 转换后将 `phi` 钳制为 1 或 0；
- 被钳掉的体积守恒地重分配到邻居。

### 5.2 论文没有说明

本文没有给出：

- GPU 上的转换候选数据结构；
- 如何并行解决相邻节点的冲突转换；
- 如何创建新的界面层；
- 重分配使用 scatter atomic 还是 gather；
- 新 interface population 的一般初始化公式；
- 重分配后是否再次分类；
- DOMAIN 入口/出口的体积分数通量。

这些不能标成本文结论。论文明确引用 Lehmann 2019 作为重分配依据，应在实现该
模块前继续核验原始算法或参考代码。

## 6. PLIC 与几何

### 6.1 论文支持

- Eq. (9) 的质量平流“不需要界面重构、法向或曲率”。
- PLIC 用于从 `phi` 估计曲率 `kappa`，供 Eq. (12) 的压力边界使用。
- Sec. 5.1 指明使用 Lehmann 2019 的 PLIC，法向采用 Parker-Youngs 近似。

### 6.2 结论

本文的质量输运器是格链 population mass exchange，不是 geometric PLIC
swept-volume advection。两者不能同时更新同一份 `mass/phi`。

论文没有规定 `normal/PLIC/kappa` 是否持久化。把它们存入 `I^n` 并在每步末尾
更新，是避免下一步重复计算的一种工程策略，但需要自行评估内存与失效管理。

## 7. 气泡

### 7.1 压力与体积

等温理想气体模型：

\[
p(b_i,t)
=
p_{atmos}\frac{V(b_i,0)}{V(b_i,t)}.
\tag{22}
\]

气泡体积：

\[
V(b_i,t)
=
\sum_{\mathbf{x}\in b_i}
\left(1-\phi(\mathbf{x},t)\right).
\tag{23}
\]

### 7.2 并行重标号

论文在 `F -> G/I` 或 `G/I -> F` 事件出现时触发 CCL。每个 `G/I` 节点保存
当前标签 `i` 和旧标签 `i_old`，并用双精度 atomic add：

\[
V_i \mathrel{+}=1-\phi(\mathbf{x},t),
\qquad
V_i^0
\mathrel{+}=
\left(1-\phi(\mathbf{x},t)\right)
\frac{p_{i_{old}}^{old}}{p_{atmos}}.
\tag{24}
\]

最后：

\[
p_i=p_{atmos}\frac{V_i^0}{V_i}.
\tag{25}
\]

论文还明确：

- CCL 对三维相邻 27 节点连通区域操作；
- cut-cell 节点不参与 CCL；
- cut-cell 区域气体按大气压力处理；
- 气泡体积累计使用双精度。

## 8. 固体、fresh/dead 与泡沫

### 8.1 固体

Sec. 4.3 支持：

- cut-cell link；
- 只在 fluid/interface cut-cell 上做反弹，跳过 gas；
- Eq. (30)-(32) 的双向力矩交换；
- thin-shell 与非闭合物体。

### 8.2 fresh/dead

论文的 fresh/dead 专指移动固体：

- `dead`：被固体新覆盖的节点，标为 gas；
- `fresh`：固体刚离开的节点；
- fresh 节点先邻域平均 `phi`；
- `phi < theta` 时为 gas；
- 否则建立 fluid 节点，邻域插值 `rho`，使用固体表面速度 `u_s`，
  并令 `S_ab = u_a u_b`；
- 阈值 `theta` 随固体速度从约 0.3 到 0.95 调整。

一般的 `G/L/I` VOF 类型变化不应与这个 moving-solid fresh/dead 模块混为一谈。

### 8.3 泡沫

泡沫是可选扩展，包括：

- D3Q7 dissolved-gas advection-diffusion，Eq. (33)-(39)、(42)；
- Henry 定律，Eq. (38)；
- disjoining pressure，Eq. (40)-(41)；
- 泡沫附近局部黏度；
- Eq. (43) 的 foam-ossifying surface tension。

第一阶段自由表面实现不需要同时实现这一整套扩展，但变量名必须为其预留清晰语义，
尤其不要占用 `Pi` 表示其他应力。

## 9. Algorithm 1 与建议模块顺序

论文 Algorithm 1 的显式顺序是：

```text
1 ResetCutCell
2 ComputeDisjoinPressure
3 Streaming + free-surface boundary
4 Compute two-way force
5 HOME collision
6 HOME-FREE bubble update
7 Fresh/dead nodes update
8 Advection-Diffusion
```

论文没有在 Algorithm 1 中单列 mass advection、重标记、重分配和 PLIC 曲率；
这些来自 Sec. 3.1 的 FSLBM 基线，具体融合位置没有完全说明。

因此项目可以为了职责清晰拆成多个模块，但应把新增时序标为 `DERIVED`，并用守恒
和时间层测试证明不会改变 Eq. (9)-(12) 的语义。
