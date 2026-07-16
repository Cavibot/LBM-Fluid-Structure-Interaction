# 01 · LBM 核心一步流水线（post-collision 时间语义）

> 对应代码：`LbmSolver.step()`、`streaming.py`、`collisions.py`、`encoding.py`  
> 目的：说明重整后「持久状态 → stream → physics → collide → encode」的时间语义与数据流。

## 图含义

旧实现把 collision 与 streaming 融合成单一 kernel，活动缓冲语义接近 *pre-collision*。  
新架构统一持久化 **post-collision** 状态，把一步拆成四个可组合阶段。

```mermaid
flowchart TB
    subgraph Persist["持久状态（双缓冲）"]
        SI["state_in<br/>post-collision EncodedStateⁿ"]
        SO["state_out<br/>post-collision EncodedStateⁿ⁺¹"]
    end

    SI --> COPY["复制边界场<br/>solid_phi / solid_body_id / vel_solid_*"]
    COPY --> STREAM

    subgraph STAGE1["① Streaming"]
        STREAM["pull-stream + 半程反弹回<br/>provider 读上游 post 分布"]
        STREAM --> KIND{collision.input_kind?}
        KIND -->|"population (EPC)"| FSTAR["scratch: f_star[Q·N]"]
        KIND -->|"moments (EMC)"| MSTAR["scratch: ρ*, j*, ρS*<br/>（边读边累计，不建全域 f）"]
    end

    FSTAR --> PHYS
    MSTAR --> EMC

    subgraph STAGE2["② Physics（仅 EPC 路径完整执行）"]
        PHYS["compute_moments → ρ, u"]
        PHYS --> SC{"G ≠ 0 ?"}
        SC -->|是| FORCE["Shan-Chen 力<br/>+ velocity shift"]
        SC -->|否| HYDRO["hydro velocity 就绪"]
        FORCE --> HYDRO
        HYDRO --> REG{"TRT + regularization?"}
        REG -->|是| REGF["reg_trt_kernel"]
        REG -->|否| COLL
        REGF --> COLL
    end

    subgraph STAGE3["③ Collision"]
        COLL["EPC: SRT / TRT<br/>f_star → f_post"]
        COLL --> FORCE2["可选 Guo 体力 / 恢复物理速度"]
        EMC["EMC: HOME-NOCM-MRT<br/>moments* → moments_post"]
    end

    FORCE2 --> ENC1
    EMC --> ENC2

    subgraph STAGE4["④ Encode（由 model.encoding 决定）"]
        ENC1{"encoding?"}
        ENC1 -->|fullf| OUTF["直接写入 state_out.f_post"]
        ENC1 -->|home| OUTH1["populations_to_home<br/>→ 10 个 HOME 矩"]
        ENC2{"encoding?"}
        ENC2 -->|home| OUTH2["直接写入 state_out.kinetic_fields"]
        ENC2 -->|fullf| OUTF2["home_to_populations<br/>Hermite 重构 → f_post"]
    end

    OUTF --> OBS
    OUTH1 --> OBS
    OUTH2 --> OBS
    OUTF2 --> OBS

    subgraph OBSERVE["观测量写出"]
        OBS["density / velocity_* / force_* / MAC 面速度"]
    end

    OBS --> SO
    SO -.->|"Domain.step 交换缓冲"| SI

    style SI fill:#1a3a5c,color:#fff
    style SO fill:#1a3a5c,color:#fff
    style STREAM fill:#2d5a3d,color:#fff
    style COLL fill:#5c3a1a,color:#fff
    style EMC fill:#5c3a1a,color:#fff
    style OBS fill:#4a3a5c,color:#fff
```

## 阅读要点

| 阶段 | 负责 | 不负责 |
|------|------|--------|
| Streaming | 上游寻址、周期回绕、bounce-back、population 获取 | 碰撞、SC 力、编码 |
| Physics | 矩、力、水动力学速度、可选正则化 | 刚体栅格化 |
| Collision | EPC 输出 `f_post`；EMC 输出 HOME 矩 | 持久编码选择 |
| Encode | 把碰撞结果写成 `model.encoding` 指定的持久格式 | 观测量以外的副作用 |

时间语义对照：

```text
旧:  fⁿ (≈ pre-collision)  --[collide+stream]-->  fⁿ⁺¹
新:  EncodedStateⁿ (post-collision)
       --stream--> f* / moments*
       --collide--> collision_output
       --encode--> EncodedStateⁿ⁺¹ (post-collision)
```
