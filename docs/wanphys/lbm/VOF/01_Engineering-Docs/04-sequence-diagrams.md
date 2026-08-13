# 时序图

为便于阅读，本文件不画一张包含全部功能的总图，而按论文模块切分。图中：

- `paper` 表示论文明确时序；
- `project` 表示本项目为模块化加入的时序；
- 可选模块不阻塞核心自由表面。

## 1. 论文 Algorithm 1 顶层时序

这张图只复述论文 Algorithm 1，不加入质量重标记和几何缓存等推断步骤。

```mermaid
sequenceDiagram
    participant Sim as Simulation
    participant Solid as CutCell
    participant Surface as Surface/Foam BC
    participant Fluid as HOME Fluid
    participant Extra as Bubble/Fresh/Foam

    loop each step
        Sim->>Solid: ResetCutCell
        Sim->>Surface: ComputeDisjoinPressure
        Sim->>Fluid: Streaming + free-surface BC
        Sim->>Solid: Compute two-way force
        Sim->>Fluid: HOME collision
        Sim->>Extra: Bubble update
        Sim->>Extra: Fresh/dead update
        Sim->>Extra: D3Q7 advection-diffusion
    end
```

说明：

- `ComputeDisjoinPressure` 和 D3Q7 仅在 foam 模式启用。
- `ResetCutCell`、two-way force 和 fresh/dead 仅在移动固体/耦合启用。
- 论文没有在 Algorithm 1 中单列 Eq. (9) 质量更新、重分配和 PLIC 更新。

## 2. LBM population 时序

该图覆盖 streaming、gas surface reconstruction、moment collection、外力和 collision。

```mermaid
sequenceDiagram
    participant Step as LbmSolver
    participant In as State n
    participant Provider as PopulationProvider
    participant Resolve as LinkResolver
    participant Collide as Collector/Collision

    Step->>Resolve: resolve target fluid/interface cells
    loop each target link
        Resolve->>In: read type, solid/domain relation
        alt fluid or interface source
            Resolve->>Provider: f_i from source state
            Provider-->>Resolve: FullF read or HOME Eq.16
        else gas source to interface
            Resolve->>In: read rho, u, S, kappa, bubble pressure
            Resolve->>Provider: reconstruct opposite f_bar at x
            Provider-->>Resolve: Eq.16 value
            Resolve->>Resolve: Eq.11 and Eq.12 or Eq.41
        else solid/cut-cell source
            Resolve->>Resolve: solid link law
        else domain boundary
            Resolve->>Resolve: project domain BC
        end
        Resolve-->>Step: write exactly one f_i*
    end
    Step->>Collide: collect rho*, j*, required moments
    Step->>Collide: compose F* and physical velocity
    Step->>Collide: collision + forcing
    Collide-->>Step: temporary kinetic state n+1
```

论文支持：

- HOME Eq. (16)-(17) 重构；
- gas-to-interface Eq. (11)；
- Eq. (12)/(41) 的自由表面密度；
- collision 的 Eq. (13)、(18)-(21)。

项目适配：

- 统一 `LinkResolver`；
- FullF provider；
- 通用 domain BC；
- `f_star` scratch；
- 把现有多 collision backend 接在统一 collector 后。

## 3. 质量平流与 TYPE 重标记时序

论文 Eq. (9) 使用旧状态 `rho(x,t)`，因此这一模块不等待 LBM collector 的 `rho*`。

```mermaid
sequenceDiagram
    participant Step as VofStepper
    participant In as State n
    participant Mass as MassAdvection
    participant Type as TypeTransition
    participant Redist as MassRedistribution

    Step->>Mass: advect phi/mass
    Mass->>In: read rho^n, phi^n, logical f^n
    loop each cell and lattice direction
        Mass->>Mass: q_i = f_bar(x+c_i)-f_i(x)
        Mass->>Mass: apply paper Eq.10 weight
    end
    Mass-->>Step: temporary phi or mass
    Step->>Type: threshold with epsilon_phi
    Type-->>Step: L/G conversion candidates
    Step->>Type: maintain interface topology
    Step->>Redist: clamp and redistribute excess
    Redist-->>Step: final phi, mass and type
```

其中只有以下部分由本文直接说明：

```text
Eq. (9)-(10)
epsilon_phi = 1e-4
I -> L/G 阈值
clamp 到 1/0
守恒重分配到邻居
```

`maintain interface topology`、并行冲突解决和具体重分配时序是项目待定设计，论文只
引用 Lehmann 2019。

## 4. Kinetic 类型转换时序

这张图是项目必须补充的适配，不是论文 Algorithm 1 中的独立模块。

```mermaid
sequenceDiagram
    participant Step as VofStepper
    participant Fluid as CollisionOutput
    participant Type as FinalTypeState
    participant Init as KineticInitializer
    participant Out as State n+1

    Step->>Init: apply finalized transitions
    Init->>Fluid: read neighboring valid liquid/interface state
    Init->>Type: read old type, new type, final phi/mass
    alt gas becomes interface
        Init->>Init: initialize rho, u, S or FullF
    else interface becomes gas
        Init->>Init: invalidate kinetic state
    else interface becomes liquid
        Init->>Init: retain/correct collision output
    else unchanged active fluid
        Init->>Init: copy collision output
    end
    Init-->>Out: finalized kinetic state
```

论文没有给出一般 VOF `gas -> interface` 的完整初始化公式。不能把 Sec. 4.3 的
moving-solid fresh-node 方法直接当作通用 VOF 初始化，二者物理事件不同。

## 5. 几何更新时间序

论文确认 PLIC 用于 curvature，但没有规定它在 Algorithm 1 中的精确调度位置。
以下是保证下一步 Eq. (12) 可直接读取一致几何的项目时序。

```mermaid
sequenceDiagram
    participant Step as VofStepper
    participant State as Final phi/type
    participant Normal as NormalEstimator
    participant Plic as PlicReconstruction
    participant Curv as CurvatureEstimator

    Step->>Normal: compute from final phi
    Normal->>State: read local phi stencil
    Normal-->>Step: normal n+1
    Step->>Plic: reconstruct interface plane
    Plic->>State: read phi and normal
    Plic-->>Step: PLIC n+1
    Step->>Curv: compute mean curvature
    Curv->>State: read phi/normal/PLIC stencil
    Curv-->>Step: kappa n+1
```

下一步 surface reconstruction 使用同一 epoch 的：

```text
phi, type, normal, PLIC, kappa
```

## 6. 气泡时序

```mermaid
sequenceDiagram
    participant Step as BubbleTracker
    participant Grid as G/I Grid Labels
    participant CCL as Parallel CCL
    participant Table as Bubble Table
    participant Out as State n+1

    Step->>Grid: detect F to G/I or G/I to F
    alt topology changed
        Step->>CCL: label 27-neighbor G/I components
        CCL-->>Grid: new bubble_id
    else topology unchanged
        Step->>Grid: reuse labels
    end
    Step->>Table: clear new V and V0
    loop each G/I node except cut-cell
        Step->>Table: double atomic adds, Eq.24
    end
    Step->>Table: pressure p = p_atmos V0/V, Eq.25
    Table-->>Out: bubble labels, volumes and pressure
```

该时序由论文 Sec. 4.2 直接支持。CCL 的具体 GPU 数据布局依项目选择。

## 7. 移动固体 fresh/dead 时序

```mermaid
sequenceDiagram
    participant Step as SolidVofCoupling
    participant Old as Old CutCell State
    participant New as New CutCell State
    participant Init as FreshDeadHandler
    participant Out as State n+1

    Step->>Old: read previous solid coverage
    Step->>New: rebuild current cut links
    Step->>Init: compare old and new coverage
    alt newly covered, dead
        Init->>Out: mark gas
    else newly exposed, fresh
        Init->>Init: average neighboring phi
        alt phi below velocity-dependent theta
            Init->>Out: mark gas
        else phi above theta
            Init->>Init: interpolate rho
            Init->>Init: use solid surface velocity
            Init->>Init: set S_ab = u_a u_b
            Init->>Out: initialize fluid node
        end
    end
```

这是论文 Sec. 4.3 的 fresh/dead，不能与一般 VOF 类型重标记合并命名。

## 8. 泡沫扩展时序

```mermaid
sequenceDiagram
    participant Step as FoamModel
    participant Surface as Interface Geometry
    participant Fluid as Fluid Velocity
    participant Gas as D3Q7 Gas Solver
    participant Bubble as Bubble Table

    Step->>Surface: trace nearby bubble interface
    Surface-->>Step: distance and disjoining pressure, Eq.40
    Step->>Gas: stream with Henry boundary, Eq.38-Eq.39
    Gas->>Fluid: read local fluid velocity
    Step->>Gas: D3Q7 CMR collision, Eq.42
    Gas-->>Step: concentration phi_gas
    Step->>Bubble: diffuse gas contribution, Eq.37
```

该模块使用的 dissolved-gas concentration 应采用不同变量名，避免与液体体积分数
`phi` 混淆。
