# 02 · FullF/HOME × EPC/EMC 组合矩阵

> 对应代码：`LbmModel.encoding/collision`、`streaming.py` 四特化 kernel、`collisions.py`、`encoding.py`  
> 目的：说明两个正交选择如何决定 streaming 特化、scratch 分配与 encode 路径。

## 图含义

持久编码与碰撞形态是 **正交** 的：

1. **encoding**：`fullf | home` —— 决定持久状态布局与 population provider；
2. **collision**：`srt | trt`（EPC） / `home_nocm`（EMC） —— 决定 streaming 后 collector 需要什么。

```mermaid
flowchart LR
    subgraph Config["LbmModel 配置"]
        E["encoding<br/>fullf | home"]
        C["collision<br/>srt | trt | home_nocm"]
    end

    E --> PROV
    C --> COLL_KIND

    subgraph Provider["Population Provider（由 encoding 决定）"]
        PROV{"encoding?"}
        PROV -->|fullf| PF["直接读 f_post[q]"]
        PROV -->|home| PH["reconstruct_home_population<br/>按方向从 10 矩重构"]
    end

    subgraph Collector["Streaming Collector（由 collision.input_kind 决定）"]
        COLL_KIND{"input_kind?"}
        COLL_KIND -->|population| CP["收集完整 f_star[Q]"]
        COLL_KIND -->|moments| CM["累计 ρ*, j*, ρS*<br/>不分配 f_star"]
    end

    PF --> STREAM_MAT
    PH --> STREAM_MAT
    CP --> STREAM_MAT
    CM --> STREAM_MAT

    subgraph STREAM_MAT["2×2 Streaming 特化 Kernel"]
        direction TB
        S11["stream_fullf_to_populations<br/>正式 EPC 主路径"]
        S12["stream_fullf_to_moments<br/>实验 EMC on FullF"]
        S21["stream_home_to_populations<br/>实验 EPC on HOME"]
        S22["stream_home_to_moments<br/>正式 EMC 主路径"]
    end

    STREAM_MAT --> PATHS

    subgraph PATHS["运行时路径"]
        direction TB
        P11["FullF + SRT/TRT<br/>f_post → f* → EPC → f_post"]
        P12["FullF + home_nocm<br/>f_post → moments* → NOCM → Hermite → f_post"]
        P21["HOME + SRT/TRT<br/>moments → f* → EPC → project HOME"]
        P22["HOME + home_nocm<br/>moments → moments* → NOCM → moments"]
    end

    style S11 fill:#1a5c2d,color:#fff
    style S22 fill:#1a5c2d,color:#fff
    style S12 fill:#5c4a1a,color:#fff
    style S21 fill:#5c4a1a,color:#fff
    style P11 fill:#1a5c2d,color:#fff
    style P22 fill:#1a5c2d,color:#fff
    style P12 fill:#5c4a1a,color:#fff
    style P21 fill:#5c4a1a,color:#fff
```

## 组合地位矩阵

```mermaid
quadrantChart
    title 编码 × 碰撞 正式/实验地位
    x-axis FullF --> HOME
    y-axis EPC 显式分布 --> EMC 矩碰撞
    quadrant-1 正式主路径 HOME-NOCM
    quadrant-2 实验 FullF+EMC
    quadrant-3 正式主路径 FullF+SRT/TRT
    quadrant-4 实验 HOME+EPC
    FullF+SRT/TRT: [0.25, 0.25]
    FullF+home_nocm: [0.25, 0.75]
    HOME+SRT/TRT: [0.75, 0.25]
    HOME+home_nocm: [0.75, 0.75]
```

> 注：部分 Mermaid 渲染器对 `quadrantChart` 支持有限；上表逻辑与下表一致，以下表为准。

| 持久编码 \\ 碰撞 | EPC：SRT / TRT | EMC：HOME-NOCM-MRT |
|---|---|---|
| **FullF** | **正式主路径** | 实验：闭式矩碰撞后 Hermite 重构 FullF |
| **HOME** | 实验：显式碰撞后投影 HOME | **正式主路径** |

## Provider × Collector 数据契约

```mermaid
sequenceDiagram
    participant State as EncodedState (post)
    participant Prov as Provider
    participant Coll as Collector
    participant Out as Collision 输入

    loop 每个方向 q = 0..18
        State->>Prov: 请求上游格点 post 信息
        alt encoding = fullf
            Prov->>Prov: 读 f_post[q, source]
        else encoding = home
            Prov->>Prov: Hermite 重构 f_q(ρ,ρu,ρS)
        end
        Note over Prov: 含周期回绕 / halfway BB
        Prov->>Coll: f_q*
        alt EPC (population)
            Coll->>Out: 写入 f_star[q]
        else EMC (moments)
            Coll->>Out: ρ+=f; j+=c·f; ρS+=(cc-cs²I)f
        end
    end
```

## 配置与校验（fail-fast）

| 规则 | 行为 |
|------|------|
| `encoding ∉ {fullf, home}` | 构造 Model 时报错 |
| `collision ∈ {raw_mrt, nocm_mrt}` | `NotImplementedError` |
| `home + G ≠ 0` | 暂不支持 |
| `use_regularization` | 仅 TRT EPC |
| HOME + rigid/moving-wall | coupling / legacy step 拒绝 |
| `collision is None` | `λ_TRT==0 → srt`，否则 `trt` |
