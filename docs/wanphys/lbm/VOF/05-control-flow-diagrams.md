# 控制流图

## 1. 项目集成总览

这是把论文基线拆成项目模块后的建议控制流，属于 `DERIVED` 集成时序，不是论文
Algorithm 1 的逐行复刻。

```mermaid
flowchart LR
    IN["状态 n"]
    PRE["可选预处理<br/>cut-cell / disjoining pressure"]

    subgraph FLUID["LBM 线"]
        R["格链解析"]
        M["收集 rho* / j* / moments"]
        C["外力闭合与 collision"]
    end

    subgraph VOF["质量与类型线"]
        X["Eq.9-Eq.10<br/>质量/phi 平流"]
        T["转换候选与拓扑"]
        D["钳制和守恒重分配"]
    end

    JOIN["应用最终类型转换<br/>完成 kinetic state"]
    NEXT["几何与气泡更新"]
    OUT["状态 n+1"]

    IN --> PRE
    PRE --> R --> M --> C --> JOIN
    PRE --> X --> T --> D --> JOIN
    D --> NEXT
    JOIN --> OUT
    NEXT --> OUT
```

两条主线的数据关系：

```text
LBM 线读取旧 kinetic state，输出临时碰撞后 kinetic state
质量线读取旧 kinetic/phi/rho，输出最终 mass/phi/type
应用转换时两条线合流
```

严格按论文 Eq. (9) 时，质量线使用 `rho(x,t)`，不等待 `M` 的 `rho*`。

## 2. LBM 格链解析

每个活动目标节点和方向必须得到唯一 population。

```mermaid
flowchart TD
    A["目标节点 x、方向 i"]
    ACTIVE{"type(x) 是<br/>LIQUID 或 INTERFACE?"}
    SKIP["跳过液体 LBM"]
    MAP["求来源 y=x-c_i<br/>先处理 periodic 映射"]
    DOMAIN{"映射后仍在域外?"}
    CUT{"link 穿过固体?"}
    SRC{"type(y)"}
    DB["DOMAIN BC<br/>项目扩展"]
    SB["solid / cut-cell law"]
    PULL["读取或 Eq.16 重构<br/>f_i^n(y)"]
    SURF_OK{"目标是 INTERFACE?"}
    SURF["Eq.11<br/>surface reconstruction"]
    ERROR["L/G 直接相邻<br/>拓扑不变量失败"]
    WRITE["写唯一 f_i*(x)"]

    A --> ACTIVE
    ACTIVE -- 否 --> SKIP
    ACTIVE -- 是 --> MAP --> DOMAIN
    DOMAIN -- 是 --> DB --> WRITE
    DOMAIN -- 否 --> CUT
    CUT -- 是 --> SB --> WRITE
    CUT -- 否 --> SRC
    SRC -- LIQUID/INTERFACE --> PULL --> WRITE
    SRC -- GAS --> SURF_OK
    SURF_OK -- 是 --> SURF --> WRITE
    SURF_OK -- 否 --> ERROR
```

论文支持 `PULL`、gas-to-interface `SURF` 和 solid/cut-cell 概念。统一
`DOMAIN/CUT/SRC` 路由及 fail-fast 是项目控制层。

## 3. 自由表面 reconstruction

```mermaid
flowchart TD
    A["gas-to-interface link"]
    B["读取 rho^n、u^n、S^n"]
    C["读取 kappa^n 和 bubble_id^n"]
    D{"封闭 bubble?"}
    E["p_g = bubble pressure"]
    F["p_g = p_atmos"]
    FOAM{"foam 启用?"}
    P0["rho_g=(p_g-2 gamma kappa)/cs^2"]
    P1["rho_g=(p_g-2 gamma kappa-Pi_disj)/cs^2"]
    VALID{"rho_g 合法?"}
    OPP["Eq.16 重构旧状态对向 f_bar"]
    EQ11["Eq.11 计算未知 f_i*"]
    FAIL["fail-fast 或明确 limiter<br/>论文未规定"]

    A --> B
    A --> C --> D
    D -- 是 --> E --> FOAM
    D -- 否 --> F --> FOAM
    FOAM -- 否 --> P0 --> VALID
    FOAM -- 是 --> P1 --> VALID
    VALID -- 否 --> FAIL
    VALID -- 是 --> OPP
    B --> OPP
    OPP --> EQ11
```

注意：本文 Eq. (11) 使用旧状态 `f_bar(x,t)`，不是另一个线程刚写出的
`f_bar*(x,t)`。

## 4. Moment、外力和碰撞

```mermaid
flowchart TD
    A["完整 f*"]
    M["收集 rho*、j*、所需 moments"]
    F["项目 ForceProvider<br/>F*=rho*g+F_user"]
    U["u_collision=(j*+F*/2)/rho*"]
    BACKEND{"collision backend"}
    HOME["论文 HOME NOCM-MRT<br/>Eq.13、Eq.18-Eq.21"]
    OTHER["FullF / SRT / TRT / MRT<br/>项目兼容路径"]
    TMP["临时 kinetic state n+1"]

    A --> M
    M --> F --> U --> BACKEND
    M --> BACKEND
    F --> BACKEND
    BACKEND -- paper fidelity --> HOME --> TMP
    BACKEND -- VOF compatible --> OTHER --> TMP
```

只有 HOME 高阶路径可以主张复现论文的碰撞稳定性设计；其他 backend 只表示共享
VOF 边界和质量模块。

## 5. 论文 Eq. (9)-(10) 质量平流

这张图严格表达本文印刷版公式。

```mermaid
flowchart TD
    A["节点 x"]
    INIT["delta_phi=0"]
    LOOP["遍历方向 i"]
    Q["q_i=f_bar^n(x+c_i)-f_i^n(x)"]
    TYPE{"type^n(x)"}
    WL["theta_i=1"]
    WG["theta_i=0"]
    WI["theta_i=(phi^n(x)+phi^n(x+c_i))/2"]
    ACC["delta_phi += theta_i*q_i/rho^n(x)"]
    MORE{"还有方向?"}
    OUT["phi_tmp=phi^n+delta_phi"]

    A --> INIT --> LOOP --> Q --> TYPE
    TYPE -- LIQUID --> WL --> ACC
    TYPE -- GAS --> WG --> ACC
    TYPE -- INTERFACE --> WI --> ACC
    ACC --> MORE
    MORE -- 是 --> LOOP
    MORE -- 否 --> OUT
```

solid link、domain boundary 和开放边界的质量通量不在 Eq. (9)-(10) 中定义，必须
作为额外策略单独设计。

## 6. TYPE 重标记与质量重分配

实线是论文明确要求；虚线是为并行实现和拓扑一致性补充的项目阶段。

```mermaid
flowchart TD
    A["临时 phi / mass"]
    ISI{"旧类型是 INTERFACE?"}
    HIGH{"phi >= 1+epsilon_phi?"}
    LOW{"phi <= -epsilon_phi?"}
    KEEP["保持 INTERFACE"]
    CL["候选 LIQUID<br/>phi clamp 到 1"]
    CG["候选 GAS<br/>phi clamp 到 0"]
    EXCESS["计算被钳制的<br/>多余/残余质量"]
    TOPO["维护一层 INTERFACE<br/>解决转换冲突"]
    REDIST["守恒重分配到<br/>邻近 INTERFACE"]
    CHECK["校验最终 phi/type<br/>与总质量"]
    OUT["最终 mass/phi/type"]

    A --> ISI
    ISI -- 否 --> TOPO
    ISI -- 是 --> HIGH
    HIGH -- 是 --> CL --> EXCESS
    HIGH -- 否 --> LOW
    LOW -- 是 --> CG --> EXCESS
    LOW -- 否 --> KEEP --> TOPO
    EXCESS -.-> TOPO
    TOPO -.-> REDIST
    REDIST -.-> CHECK
    CHECK --> OUT
```

论文只明确阈值、clamp 和守恒重分配，没有给出 `TOPO/REDIST/CHECK` 的 GPU
算法。实现前应核验被引用的 Lehmann 2019。

## 7. 最终 kinetic 类型转换

```mermaid
flowchart TD
    A["旧 type、新 type<br/>临时 collision output"]
    CHANGE{"类型变化?"}
    SAME["复制有效 collision output"]
    CASE{"转换类型"}
    GI["GAS -> INTERFACE<br/>初始化 kinetic state"]
    IG["INTERFACE -> GAS<br/>标记 kinetic 无效"]
    IL["INTERFACE -> LIQUID<br/>保留并校正"]
    LI["LIQUID -> INTERFACE<br/>保留并设置 VOF 状态"]
    VALID["检查 rho/u/S 或 FullF 有效"]
    OUT["最终 kinetic state n+1"]

    A --> CHANGE
    CHANGE -- 否 --> SAME --> OUT
    CHANGE -- 是 --> CASE
    CASE -- G to I --> GI --> VALID
    CASE -- I to G --> IG --> OUT
    CASE -- I to L --> IL --> VALID
    CASE -- L to I --> LI --> VALID
    VALID --> OUT
```

一般 `GAS -> INTERFACE` 初始化公式论文未说明，是实现阻塞决策。不要直接套用
moving-solid fresh-node 公式。

## 8. 几何

```mermaid
flowchart TD
    A["最终 phi/type"]
    N["Parker-Youngs normal"]
    P["PLIC reconstruction"]
    K["mean curvature"]
    SIGN["球面/平面测试<br/>锁定法向与曲率符号"]
    CACHE{"持久缓存?"}
    STORE["写 normal/PLIC/kappa<br/>geometry_epoch++"]
    TEMP["保留为下一步可重建状态"]

    A --> N --> P --> K --> SIGN --> CACHE
    CACHE -- 是 --> STORE
    CACHE -- 否 --> TEMP
```

`phi -> normal -> PLIC -> curvature` 有论文依据；更新时间点、缓存和 epoch 是项目
设计。

## 9. 气泡

```mermaid
flowchart TD
    A["最终 phi/type"]
    EVENT{"F 与 G/I 之间<br/>发生类型变化?"}
    CCL["对 G/I 运行 27 邻域 CCL<br/>跳过 cut-cell"]
    REUSE["复用当前标签"]
    CLEAR["清零新 bubble 的 V、V0"]
    LOOP["遍历所有非 cut-cell G/I 节点"]
    ATOMIC["按 Eq.24 双精度 atomic add"]
    PRESS["按 Eq.25 更新压力"]
    OUT["bubble state n+1"]

    A --> EVENT
    EVENT -- 是 --> CCL --> CLEAR
    EVENT -- 否 --> REUSE --> CLEAR
    CLEAR --> LOOP --> ATOMIC --> PRESS --> OUT
```

## 10. 移动固体 fresh/dead

```mermaid
flowchart TD
    A["比较旧/新固体覆盖"]
    DEAD{"新覆盖?"}
    FRESH{"新暴露?"}
    DG["标为 GAS"]
    AVG["邻域平均 phi"]
    TH{"phi < theta(u_s)?"}
    FG["标为 GAS"]
    FF["建立 FLUID<br/>插值 rho<br/>u=u_s<br/>S_ab=u_a*u_b"]
    NONE["无 fresh/dead 变化"]

    A --> DEAD
    DEAD -- 是 --> DG
    DEAD -- 否 --> FRESH
    FRESH -- 否 --> NONE
    FRESH -- 是 --> AVG --> TH
    TH -- 是 --> FG
    TH -- 否 --> FF
```

该流程只表示论文 Sec. 4.3 的移动固体覆盖变化。
