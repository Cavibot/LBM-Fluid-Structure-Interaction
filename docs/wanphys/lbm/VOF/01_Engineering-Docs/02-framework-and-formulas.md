# 实现框架与重点公式

## 1. 实现范围分层

为避免一次性引入论文全部功能，框架分为四层。

### Level 1：核心自由表面

必须包含：

- L/I/G 节点；
- HOME 或 FullF 的 population 提供器；
- Eq. (9)-(10) 格链质量平流；
- Eq. (11)-(12) 自由表面压力边界；
- 类型转换、钳制和守恒重分配；
- PLIC 法向与曲率；
- 外力闭合与现有碰撞；
- 静态固体边界。

### Level 2：封闭气泡

增加：

- G/I 连通分量；
- current/old bubble labels；
- Eq. (22)-(25) 气泡体积和压力；
- merge/split 的并行继承。

### Level 3：移动固体与双向耦合

增加：

- cut-cell link；
- double-sided bounce-back；
- Eq. (30)-(32) 流固力矩交换；
- moving-solid fresh/dead。

### Level 4：泡沫

增加：

- D3Q7 dissolved-gas transport；
- Henry 边界；
- disjoining pressure；
- 局部 eddy/foam viscosity；
- foam-ossifying surface tension。

后一级不得成为前一级质量守恒和稳定性测试的前置条件。

## 2. 变量与时间语义

| 符号 | 含义 | 时间层 |
|---|---|---|
| `fMom^n` | 持久 kinetic 状态；HOME 为 `rho,u,S` | 碰撞后 |
| `f_i^n` | 从 `fMom^n` 读取或重构的 logical population | 碰撞后 |
| `f_i*` | streaming 和边界处理后的 population | 碰撞前 |
| `rho*, j*, S*` | 从完整 `f*` 收集的临时矩 | 碰撞前 |
| `F*` | 本步 collision 使用的力密度 | 碰撞前 |
| `u*` | 从 population 得到的临时速度量 | 碰撞前 |
| `phi^n` | 液体体积分数 | 当前状态 |
| `mass^n` | 液体质量；若持久保存，必须与 `phi/rho` 一致 | 当前状态 |
| `type^n` | `LIQUID/INTERFACE/GAS` | 当前状态 |
| `kappa^n` | 当前界面平均曲率 | 当前状态或临时缓存 |
| `p_b^n` | 当前 bubble pressure | 当前状态 |

论文 Eq. (7) 中：

$$
\rho=\sum_i f_i,
\qquad
\rho\mathbf{u}
=
\sum_i\mathbf{c}_if_i+\frac12\mathbf{F}.
\tag{7}
$$

所以必须区分：

$$
\mathbf{j}_{raw}=\sum_i\mathbf{c}_if_i
$$

和：

$$
\mathbf{u}_{physical}
=
\frac{\mathbf{j}_{raw}+\mathbf{F}/2}{\rho}.
$$

论文 HOME 状态保存的是物理 `u`。如果项目适配层保存 `j_raw`，只允许在明确的
closure 中加一次 `F/2`。

## 3. 初始化

### 3.1 必需输入

- 初始 `phi^0`；
- 初始 `type^0`，或由 `phi^0` 构造；
- 初始 `rho^0, u^0, S^0` 或 FullF populations；
- 固体和 domain boundary；
- 外力参数；
- surface tension `gamma`；
- atmosphere pressure `p_atmos`；
- 可选初始 bubble labels。

### 3.2 初始化步骤

```text
1 校验 0 <= phi <= 1
2 建立 L/I/G 类型，并保证 L 与 G 之间有 I
3 令 mass 与 phi、有效液体密度一致
4 初始化 kinetic equilibrium / HOME moments
5 计算 normal、PLIC、curvature
6 对 G/I 执行初始 CCL，初始化 bubble V、V0、p
7 初始化双缓冲输出和 scratch
```

初始几何完成前，不得进入第一次自由表面 reconstruction。

## 4. LBM population 主流程

### 4.1 HOME population 重构

论文 Eq. (16)-(17)：

$$
f_i
=
\rho w_i
\left[
1+
\frac{\mathbf{c}_i\cdot\mathbf{u}}{c_s^2}
+
\frac{\mathbf{H}^{[2]}(\mathbf{c}_i):\mathbf{S}}{2c_s^4}
+
\frac{
\sum_{\alpha\beta\gamma}
H_{\alpha\beta\gamma}^{[3]}(\mathbf{c}_i)
T_{\alpha\beta\gamma}
}{2c_s^6}
\right],
\tag{16}
$$

$$
T_{\alpha\beta\gamma}
=
S_{\alpha\beta}u_\gamma
+
S_{\alpha\gamma}u_\beta
+
S_{\beta\gamma}u_\alpha
-
2u_\alpha u_\beta u_\gamma.
\tag{17}
$$

FullF 适配路径直接读取持久 population。无论编码为何，下游统一消费 logical
`f_i^n`。

### 4.2 格链解析

对目标 fluid/interface 节点 `x` 和方向 `i`，来源为：

$$
\mathbf{y}=\mathbf{x}-\mathbf{c}_i.
$$

工程层统一解析：

```text
periodic source       -> 映射后普通 pull
fluid/interface       -> 从 y 读取或重构 f_i^n(y)
solid/cut-cell        -> solid link law
gas to interface      -> Eq. (11)
domain boundary       -> 项目 domain BC
```

该统一路由是项目抽象，不是论文枚举。实现必须保证每个 `f_i*(x)` 有唯一写入者。

### 4.3 自由表面 reconstruction

论文 Eq. (11)：

$$
f_i^*(\mathbf{x},t)
=
f_i^{eq}(\rho_g,\mathbf{u}(\mathbf{x},t))
+
f_{\bar i}^{eq}(\rho_g,\mathbf{u}(\mathbf{x},t))
-
f_{\bar i}(\mathbf{x},t).
\tag{11}
$$

其中对向 `f_bar(x,t)`：

- HOME：按 Eq. (16)-(17) 从 `rho^n,u^n,S^n` 重构；
- FullF：从持久 population 读取。

核心自由表面密度：

$$
\rho_g
=
\frac{p_b-2\gamma\kappa}{c_s^2}.
\tag{12}
$$

无封闭 bubble 时：

$$
p_b=p_{atmos}.
$$

泡沫模式：

$$
\rho_g
=
\frac{p_b-2\gamma\kappa-\Pi_{disj}}{c_s^2}.
\tag{41}
$$

必须验证：

$$
\rho_g>0
$$

并定义超出稳定范围时的 fail-fast 或 limiter 策略。论文未给出该保护策略。

### 4.4 宏观量收集

完整 `f*` 后计算：

$$
\rho^*=\sum_i f_i^*,
\qquad
\mathbf{j}^*=\sum_i\mathbf{c}_i f_i^*.
$$

项目 collision closure 使用：

$$
\mathbf{u}_{collision}
=
\frac{\mathbf{j}^*+\mathbf{F}^*/2}{\rho^*}.
$$

论文 HOME 表达中则按 Eq. (18)-(21) 直接更新持久 moments。

### 4.5 外力与碰撞

项目中建议明确：

$$
\mathbf{F}^*
=
\rho^*\mathbf{g}+\mathbf{F}_{user}.
$$

这是当前工程外力接口；论文只使用抽象 external force `F`，没有规定 provider
组合方式。

论文高阶碰撞是：

$$
\boldsymbol{\Omega}
=
-\mathbf{M}^{-1}
\left[
\mathbf{R}(\mathbf{m}-\mathbf{m}^{eq})
+
\left(\mathbf{I}-\frac12\mathbf{R}\right)\mathbf{K}
\right].
\tag{13}
$$

论文持久 HOME 输出：

$$
\rho^{n+1}=\rho^*,
\tag{18}
$$

$$
u_\alpha^{n+1}
=
u_\alpha^*
+
\frac{F_\alpha}{2\rho^*}.
\tag{19}
$$

Eq. (20)-(21) 更新二阶速度矩 `S`。完整实现应直接参照论文，不在本概览中重新
抄写所有分量闭式式子，避免索引转录错误。

## 5. 质量平流与类型重标记

### 5.1 论文质量更新

$$
\phi^{n+1}(\mathbf{x})
=
\phi^n(\mathbf{x})
+
\frac{1}{\rho^n(\mathbf{x})}
\sum_i
\theta_i^n(\mathbf{x})
q_i^n(\mathbf{x}),
\tag{9}
$$

$$
q_i^n(\mathbf{x})
=
f_{\bar i}^n(\mathbf{x}+\mathbf{c}_i)
-
f_i^n(\mathbf{x}).
$$

按论文印刷 Eq. (10)：

$$
\theta_i^n(\mathbf{x})=
\begin{cases}
1,&\mathbf{x}\in L,\\
0,&\mathbf{x}\in G,\\
\dfrac{\phi^n(\mathbf{x})+
\phi^n(\mathbf{x}+\mathbf{c}_i)}{2},
&\mathbf{x}\in I.
\end{cases}
\tag{10}
$$

如果实现持久 `mass`，必须证明它与 Eq. (9) 等价，并在每步输出维持：

$$
mass=\rho_{full}\phi.
$$

不能同时再使用 PLIC swept-volume 更新同一份 `mass/phi`。

### 5.2 类型阈值

论文给出：

$$
\epsilon_\phi=10^{-4}.
$$

对 interface：

```text
phi >= 1 + epsilon_phi -> 候选 LIQUID
phi <= 0 - epsilon_phi -> 候选 GAS
otherwise              -> INTERFACE
```

转换时：

```text
LIQUID -> phi clamp 到 1
GAS    -> phi clamp 到 0
```

被钳制的体积必须守恒地分配给邻近 interface。

### 5.3 论文缺失的工程步骤

以下步骤是为了实现论文高层要求，但具体算法不由本文给出：

```text
候选转换
→ 维护 L/I/G 拓扑，避免 L 直接邻接 G
→ 解决并行转换冲突
→ 确定重分配接收者
→ 守恒重分配
→ 最终 phi/type 校验
→ 初始化新 interface kinetic 状态
```

这些步骤在编码前应继续核验 Lehmann 2019 或 HOME-FREE 参考实现。

## 6. 几何

核心关系：

```text
最终 phi
→ Parker-Youngs normal
→ PLIC
→ mean curvature kappa
→ 下一步 Eq. (12)
```

论文说明 PLIC 只用于曲率估计；质量更新 Eq. (9) 不依赖 PLIC。

实现必须固定：

- normal 朝向；
- curvature 正负号；
- Eq. (12) 的 `-2 gamma kappa` 与该约定一致；
- solid contact angle 的处理；
- domain 边缘 stencil。

这些符号约定应由平面、球形气泡和液滴 Laplace pressure 测试锁定。

## 7. 气泡更新

### 7.1 触发

检测：

```text
fluid -> gas/interface
gas/interface -> fluid
```

事件出现时运行 3D CCL。没有事件时可复用现有 labels；这是论文 Sec. 4.2 的优化。

### 7.2 体积和压力

$$
V_i
=
\sum_{\mathbf{x}\in b_i}(1-\phi_\mathbf{x}),
\tag{23}
$$

$$
V_i^0
\mathrel{+}=
(1-\phi_\mathbf{x})
\frac{p_{i_{old}}^{old}}{p_{atmos}},
\tag{24}
$$

$$
p_i=p_{atmos}\frac{V_i^0}{V_i}.
\tag{25}
$$

累计 `V` 和 `V0` 使用双精度 atomic add。每个 G/I 节点保存新旧 bubble label。
cut-cell 不参加 CCL，其气压按大气压力处理。

## 8. 可选固体与泡沫步骤

### 8.1 固体

论文 Algorithm 1 在每步最先刷新 cut-cell。之后：

- streaming 对 cut link 使用固体表面重构；
- fluid/interface cut-cell 参与反弹；
- gas cut-cell 跳过；
- collision 前计算 Eq. (32) 双向力；
- collision 后更新 moving-solid fresh/dead。

### 8.2 泡沫

每步在 streaming 前计算：

$$
\Pi_{disj}=
\begin{cases}
0,&d>d_{max},\\
k_\pi(1-d/d_{max}),&d\le d_{max}.
\end{cases}
\tag{40}
$$

然后由 Eq. (41) 进入自由表面密度。流体 collision 后运行 D3Q7
advection-diffusion，相关公式为 Eq. (33)-(39)、(42)。

## 9. 必须保持的验收不变量

### 9.1 Population

- 每个活动节点的每个 `f_i*` 恰好写一次；
- gas 节点不执行液体 collision；
- surface reconstruction 只处理 gas-to-interface links；
- `rho*`、population 和输出 moments 全部有限；
- 零力平衡界面不产生净动量。

### 9.2 质量与类型

- 封闭域总液体质量只在浮点容差内变化；
- interface-interface 链路交换成对守恒；
- 重分配前后总质量一致；
- 最终 `LIQUID` 的 `phi=1`；
- 最终 `GAS` 的 `phi=0`；
- 最终 `INTERFACE` 满足合法范围；
- 不存在直接相邻的 L/G 链路，除非算法明确允许并有边界处理。

### 9.3 几何与压力

- 平面界面 `kappa≈0`；
- 半径为 `R` 的球面满足选定符号下的 `|kappa|≈1/R`；
- Eq. (12) 产生预期 Laplace pressure；
- `rho_g` 始终为正且处于 collision 可接受范围。

### 9.4 气泡

- 静止气泡 `pV` 保持常量；
- merge/split 前后继承的 `V0` 守恒；
- 标签改变不改变总气泡体积；
- cut-cell 不进入 bubble CCL；
- 开放气相使用 `p_atmos`。

## 10. 实现前待决策

1. Eq. (9)-(10) 严格按本文印刷版，还是采用另一个经典 FSLBM 变体？
2. 第一版只支持 HOME，还是允许 FullF 作为调试/对照路径？
3. D3Q19 先做 VOF 兼容，还是直接实现论文 D3Q27 高阶 HOME-FREE？
4. `normal/PLIC/kappa` 持久保存还是按需 scratch？
5. 新 interface 的 population 初始化采用哪一来源？
6. 并行质量重分配采用 atomic scatter 还是两阶段 gather？
7. 第一版是否限制为封闭域，暂不支持 VOF 开口入口/出口？
8. bubble、moving solid、foam 分别在哪个里程碑启用？

这些问题必须作为项目决策记录，不能用“论文默认如此”代替。
