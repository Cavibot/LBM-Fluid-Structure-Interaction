# CompositeSimulation 重构说明

**背景**：`feature/collision-pipeline` 分支上完成，分两个阶段提交：
1. `230a676f` — 将 `CompositeSimulation` 改为 ABC，新增 `SimpleSimulation`
2. `cc61db39` — 完全移除 `SimpleSimulation`，示例改用显式步进循环

---

## 我们做了什么

### 阶段一：拆分职责

原始的 `CompositeSimulation` 是一个具体类，承担了三件不同的事：

- **编排**：维护 `_domains` 字典，调用每个域的 `pre_step / step / post_step`
- **耦合注册**：`add_coupling(Coupling)` — 管理跨域耦合列表
- **碰撞代理**：`set_collision_pipeline(pipeline)` — 持有 `CollisionPipeline` 并在内部调用 `pipeline.run()`

第一次重构将这个类拆成两层：

```
CompositeSimulation (ABC)      ← 只剩 _time / time / step() / reset() 三件事
    └── SimpleSimulation       ← 继承 ABC，保留原有编排逻辑
```

同步删除了 `Coupling` ABC 及 `coupling.py`（当时无任何真实子类，是死代码）。

### 阶段二：删除 SimpleSimulation

第二次重构将 `SimpleSimulation` 完全移除，所有示例改为**显式步进循环**：

```python
# 之前 (SimpleSimulation)
sim = SimpleSimulation()
sim.add_domain(domain)
sim.step(dt)          # 黑盒，内部做了什么不知道

# 之后 (显式)
domain.state.clear_forces()
domain.pre_step(dt)
contacts = CollisionPipeline.collide_rigid(domain)
domain.step(dt, contacts=contacts)
domain.post_step(dt)
```

同步新增了 `CollisionPipeline.collide_particles()` classmethod，让粒子流体域与刚体域有对称的公开 API。

---

## 为什么要这样做

### 旧架构的问题

**1. 通用编排器掩盖了物理逻辑**

`SimpleSimulation.step()` 把所有域一视同仁地循环执行：

```python
for name, domain in self._domains.items():
    domain.step(dt, contacts=contacts)
```

但物理仿真中域的执行顺序往往**有物理意义**。流固耦合中流体必须先于刚体更新（或反之），WCSPH 和 PBF 的子步数也不同。通用编排器迫使开发者把这些物理约束藏进 `Coupling.apply()` 的副作用里，语义隐晦。

**2. 碰撞检测的路径是间接的**

旧架构中碰撞检测流程：

```
CompositeSimulation.step()
  └─ pipeline.run(states_dict)        ← 通过字符串 key 查 domain
       └─ CollisionResult.intra["rigid"]
            └─ domain.step(contacts=...)
```

`CollisionPipeline` 需要一个 `states` 字典，由 `CompositeSimulation` 在运行时构建。这产生了循环依赖：`Pipeline` 要依赖 `CompositeSimulation` 来喂数据，而 `CompositeSimulation` 又要持有 `Pipeline`。

新架构中：

```python
contacts = CollisionPipeline.collide_rigid(domain)   # 直接传 domain 对象
domain.step(dt, contacts=contacts)
```

调用链扁平，无中间状态字典，无双向依赖。

**3. 外力注册机制是不必要的间接层**

```python
sim.register_external_forces(lambda states, dt: viewer.apply_forces(states["rigid"].as_newton_state()))
```

这个 callback 机制的唯一目的是让 `viewer.apply_forces()` 在正确的时机被调用。但调用时机完全由调用方控制，无需注册：

```python
viewer.apply_forces(domain.state.as_newton_state())   # 直接写在这里
domain.step(dt, ...)
```

**4. `Coupling` ABC 是提前抽象**

`Coupling` 类在仓库中从未有任何实际子类被创建和使用。WCSPH 双向耦合示例（`fluid_wcsph_twoway_coupling.py`）的耦合逻辑直接写在 `WCSPHRigidTwoWayCoupling(CompositeSimulation)` 的 `step()` 里，证明了 `Coupling` 这层抽象是冗余的。

---

## 新架构的优势

### 1. 显式优于隐式

步进循环写在示例文件里，开发者一眼就能看到物理操作的完整顺序：清力、预处理、碰撞、积分、后处理。调试时无需进入 `SimpleSimulation.step()` 查找实际发生了什么。

### 2. CompositeSimulation 仍然存在，用于真正的耦合场景

对于需要跨域协调的真实耦合算法，仍然应该子类化 `CompositeSimulation`：

```python
class WCSPHRigidTwoWayCoupling(CompositeSimulation):
    def step(self, dt):
        # 1. 流体预测步
        self.fluid.pre_step(dt)
        # 2. 流体→刚体力传递
        self._transfer_fluid_to_rigid()
        # 3. 刚体积分
        contacts = CollisionPipeline.collide_rigid(self.rigid)
        self.rigid.step(dt, contacts=contacts)
        # 4. 刚体→流体边界更新
        self._update_fluid_boundary()
        # 5. 流体压力求解 + 积分
        self.fluid.step(dt)
        self._time += dt
```

这是 ABC 的正确使用方式：**有稳定接口、有明确语义、有真实子类**。

### 3. CollisionPipeline 职责清晰

`CollisionPipeline` 不再需要被注入到 `SimpleSimulation` 里才能工作。它暴露的 classmethod 可以在任何地方直接调用，与调用方的类层次结构无关：

| classmethod | 用途 |
|---|---|
| `collide_rigid(domain)` | 刚体域内碰撞 |
| `collide_particles(domain)` | 粒子流体域与边界形状碰撞 |
| `collide_rigid_fluid(rigid, fluid)` | 刚体-流体跨域碰撞（构建桥接 Newton 模型，缓存复用） |

### 4. 单域示例不再需要"模拟多域"的框架

`fluid_pbf_dam_break.py` 等都是单域示例。把它们包进 `SimpleSimulation` 是强迫单域场景适配多域 API，多此一举。

---

## 设计准则总结

> **能用三行写清楚的步进循环，不要藏进通用编排器。**
> 通用编排器只有在"通用"有实际价值时才有意义——即多个完全不同的场景可以共用同一套编排逻辑。
> WanPhys 中每个耦合场景的物理顺序都是独特的，通用编排器只是掩盖了这种独特性。
>
> `CompositeSimulation` 作为 ABC 保留，是为了给**真实耦合算法**提供统一的类型标识和时间追踪接口，
> 而不是为了让所有仿真都通过它运行。
