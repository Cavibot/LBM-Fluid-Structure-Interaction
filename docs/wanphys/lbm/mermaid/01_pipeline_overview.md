# 01 · LBM 统一一步流水线（post-collision 时间语义）

> 对应代码：`LbmSolver.step()`、`streaming.py`、`boundaries.py`、
> `forcing.py`、`moments.py`、`collisions.py`、`encoding.py`。

## 数据流

```mermaid
flowchart TB
    SI["EncodedStateⁿ<br/>post-collision"]

    subgraph PROVIDER["StateProvider"]
        PF["FullF：直接读取 fPostᵢ"]
        PH["HOME：Hermite 重构 fPostᵢ"]
    end

    subgraph BOUNDARY["Part 3：迁移与边界补全"]
        ST["Pull Streaming"]
        TL["TransportLinkLaw<br/>periodic / bounce-back / moving-wall / cut-link"]
        OC["OpenBoundaryCompletion<br/>Zou-He / pressure / convective"]
        FS["完整 f*[Q]"]
        SURF["SurfaceCompletion<br/>仅接口；选择时 fail-fast"]
    end

    subgraph COLLECT["统一 Collector"]
        PC["PopulationCollector"]
        MA["MomentAccumulator<br/>ρ* / j* / stress / required moments"]
    end

    subgraph FORCE["Part 2：统一外力"]
        FP["ForceProvider<br/>F = ρg + Fsc"]
        HC["HydroClosure<br/>u = (j* + F/2) / ρ*"]
    end

    subgraph COLLIDE["Part 1：碰撞算子"]
        CTX["CollisionContext<br/>f* / moments / ρ / j / u / F"]
        EPC["SRT / TRT<br/>population or even/odd source"]
        RAW["Raw MRT<br/>raw-moment source"]
        NOCM["FullF NOCM MRT<br/>19D central-moment source"]
        HOME["HOME NOCM MRT<br/>retained-moment closure source"]
    end

    EN["Encoder<br/>FullF 或 HOME"]
    SO["EncodedStateⁿ⁺¹<br/>post-collision"]

    SI --> PF
    SI --> PH
    PF --> ST
    PH --> ST
    TL --> ST
    ST --> OC
    OC --> FS
    SURF -. "不进入当前执行流" .-> OC

    FS --> PC
    FS --> MA
    PC --> FP
    MA --> FP
    MA --> HC
    FP --> HC

    FS --> CTX
    PC --> CTX
    MA --> CTX
    FP --> CTX
    HC --> CTX

    CTX --> EPC
    CTX --> RAW
    CTX --> NOCM
    CTX --> HOME
    EPC --> EN
    RAW --> EN
    NOCM --> EN
    HOME --> EN
    EN --> SO
    SO -. "Domain.step 交换双缓冲" .-> SI
```

## 强制时序

```text
post-collision state n
→ streaming / transport link law
→ open-boundary population completion
→ 完整 f*[Q]
→ population / moment collection
→ force computation
→ hydro closure
→ collision-space force injection + collision
→ encode post-collision state n+1
```

因此边界只负责补齐 population；它不查询 CollisionSpace。Zou-He、pressure、
convective 的结果先进入 Collector，随后才计算宏观量、外力和碰撞。

## 模块职责

| 模块 | 负责 | 不负责 |
|---|---|---|
| StateProvider | FullF 读取或 HOME 重构 | 边界与碰撞 |
| TransportLinkLaw | 周期、反弹、移动壁面、cut-link | 开口宏观条件 |
| OpenBoundaryCompletion | 补齐未知 incoming population | 碰撞基底 |
| Collector | 由完整 `f*[Q]` 统计状态 | 修改边界 |
| ForceProvider | 计算统一力密度 `F` | 决定注入基底 |
| HydroClosure | 计算半步物理速度 | 重复施加外力 |
| CollisionOperator | 把 `F` 翻译到自身基底并碰撞 | 选择持久编码 |
| Encoder | 保存为 FullF 或 HOME | 重新解释碰撞公式 |
