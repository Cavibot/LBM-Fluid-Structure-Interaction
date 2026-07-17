# 03 · 模块与类结构（Domain / Model / State / Solver）

> 对应代码：`lbm/__init__.py`、`domain.py`、`model.py`、`state.py`、`solver.py` 及新增子模块  
> 目的：说明生命周期归属、状态类型拆分、Solver 如何缓存 backend 与 scratch。

## 图含义

本轮重整 **保留单一 Domain / Model / Solver**，只把 State 按内存布局拆成两个具体类型；  
新增 `streaming` / `collisions` / `encoding` 三个正交子模块，由 Solver 在 `__init__` 中根据配置装配。

```mermaid
classDiagram
    direction TB

    class LbmDomain {
        +LbmModel model
        +LbmSolver solver
        -LbmStateBase state_in
        -LbmStateBase state_out
        +create_state() LbmStateBase
        +step(dt)
    }

    class LbmModel {
        +str encoding
        +str|None collision
        +float tau
        +float lambda_trt
        +float G
        +bool use_regularization
        +resolved_collision
        +omega / omega_plus / omega_minus
    }

    class LbmStateBase {
        <<abstract-ish base>>
        +density
        +velocity_x/y/z
        +vel_u/v/w  MAC
        +vel_solid_*
        +solid_phi / solid_body_id
        +force_x/y/z
        +clear()
        +clone()
    }

    class FullFLbmState {
        +f_post : Q*N
        +f 兼容别名
    }

    class HomeLbmState {
        +rho
        +rho_u_x/y/z
        +rho_s_xx/yy/zz/xy/xz/yz
        +kinetic_fields()
    }

    class LbmSolver {
        -collision_backend
        -collision_kind
        -f_star / f_post scratch
        -moments / post_moments scratch
        -_rho, _ux/_uy/_uz, _fx/_fy/_fz
        +create_state()
        +step(state_in, state_out, dt)
        -_stream_to_populations()
        -_stream_to_moments()
        -_prepare_population_physics()
        -_encode_populations_to_home()
        -_write_observables()
        -_step_legacy() 移动壁兼容
    }

    class SrtCollision {
        +input_kind = population
        +relaxation_rates()
    }
    class TrtCollision {
        +input_kind = population
        +relaxation_rates()
    }
    class HomeNocmMrtCollision {
        +input_kind = moments
    }

    LbmDomain *-- LbmModel
    LbmDomain *-- LbmSolver
    LbmDomain o-- LbmStateBase : double buffer
    LbmSolver --> LbmModel
    LbmSolver --> LbmStateBase : create / step
    LbmStateBase <|-- FullFLbmState
    LbmStateBase <|-- HomeLbmState
    LbmSolver --> SrtCollision
    LbmSolver --> TrtCollision
    LbmSolver --> HomeNocmMrtCollision

    note for FullFLbmState "LbmState = FullFLbmState\n默认兼容别名"
    note for LbmSolver "无独立 ExecutionPlan 类\nbackend 只是选择元数据\nkernel 启动仍属 Solver"
```

## 模块依赖与职责

```mermaid
flowchart TB
    subgraph PublicAPI["公开 API（lbm/__init__.py）"]
        D[LbmDomain]
        M[LbmModel]
        S[LbmSolver]
        ST[FullFLbmState / HomeLbmState / LbmState]
    end

    subgraph CoreModules["核心子模块"]
        STR["streaming.py<br/>4 个 pull-stream kernel<br/>+ moment_velocity"]
        COL["collisions.py<br/>EPC kernel + HOME-NOCM kernel<br/>Srt/Trt/HomeNocm backends"]
        ENC["encoding.py<br/>Hermite 重构 / 矩投影<br/>方向与平衡态 wp.func"]
        KER["kernels.py<br/>遗留矩/力/正则化/MAC<br/>+ legacy collide-stream"]
    end

    subgraph External["外部耦合（本轮不重构）"]
        COUP["GridLbmRigidCoupling<br/>仅 FullF + legacy step"]
    end

    D --> M
    D --> S
    D --> ST
    S --> STR
    S --> COL
    S --> ENC
    S --> KER
    STR --> ENC
    COL --> ENC
    COUP --> D
    COUP -.->|"HOME 拒绝"| ST

    style STR fill:#1a4a3a,color:#fff
    style COL fill:#4a3a1a,color:#fff
    style ENC fill:#1a3a4a,color:#fff
    style KER fill:#3a3a3a,color:#ccc
```

## 对象生命周期

```mermaid
sequenceDiagram
    participant User
    participant Domain as LbmDomain
    participant Model as LbmModel
    participant Solver as LbmSolver
    participant State as state_in / state_out

    User->>Model: LbmModel(encoding=..., collision=..., ...)
    Note over Model: fail-fast 校验 encoding/collision<br/>解析 resolved_collision
    User->>Domain: LbmDomain(model)
    Domain->>Solver: LbmSolver(model)
    Note over Solver: 选 collision backend<br/>按 input_kind 分配 scratch

    User->>Domain: create_state()
    Domain->>Solver: create_state() × 2
    Solver->>State: FullF 或 Home（由 encoding）
    Domain->>Domain: state_in, state_out

    User->>Solver: initialize_equilibrium(...)
    loop 每个时间步
        User->>Domain: step(dt)
        Domain->>Solver: step(state_in, state_out, dt)
        alt has_moving_walls
            Solver->>Solver: _step_legacy (FullF only)
        else 新流水线
            Solver->>Solver: stream → physics → collide → encode
            Solver->>Solver: _write_observables
        end
        Domain->>Domain: swap state_in ↔ state_out
    end
```

## Solver scratch 分配规则

| collision.input_kind | encoding | 分配的 scratch |
|---|---|---|
| `population` | `fullf` | `_f_star`；碰撞直接写 `state_out.f_post` |
| `population` | `home` | `_f_star` + `_f_post`；再投影到 HOME 矩 |
| `moments` | `home` | `_moments`；碰撞直接写 `state_out.kinetic_fields` |
| `moments` | `fullf` | `_moments` + `_post_moments`；再 Hermite 重构 `f_post` |

## 设计决策摘要

1. **不拆 Domain/Model/Solver 两套实现** —— 配置驱动单一路径。  
2. **State 必须拆类型** —— FullF 的 `Q·N` 与 HOME 的 10 矩布局不同。  
3. **无 `output_encoding`** —— 输出编码始终等于 `model.encoding`。  
4. **无独立 ExecutionPlan 类** —— `Solver.__init__` 缓存调用约定即可。  
5. **coupling 暂走 legacy** —— 移动壁 / MEM 仍用旧 FullF 融合路径。
