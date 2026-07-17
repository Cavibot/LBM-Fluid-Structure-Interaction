# 03 · LBM 模块与契约结构

```mermaid
classDiagram
    class LbmModel {
        +encoding
        +collision
        +force_model
        +boundary_models
        +bc_types
        +bc_velocity
        +bc_density
        +bc_convective_speed
        +has_moving_walls
        +use_cut_link
    }

    class Contracts {
        +Encoding
        +Collision
        +CollisionSpace
        +ForceModel
        +BoundaryModel
        +CapabilityStatus
        +CollisionContract
        +CollisionContext
    }

    class LbmSolver {
        +step()
        -stream_to_populations()
        -complete_open_boundaries()
        -compute_force_density()
        -collide_raw_mrt()
        -collide_fullf_nocm()
        -encode_populations_to_home()
    }

    class StateProvider {
        <<module streaming.py>>
        +FullF pull
        +HOME reconstruct-and-pull
        +moving-wall transport
        +cut-link interpolation
    }

    class BoundaryCompletion {
        <<module boundaries.py + kernels.py>>
        +BoundaryResolver
        +Zou-He
        +pressure
        +history convective
        +corner static-wall fallback
        +SurfaceCompletion interface
    }

    class ForcePipeline {
        <<module forcing.py>>
        +ForceProvider
        +compose F = rho*g + Fsc
        +HydroClosure
    }

    class MomentPipeline {
        <<module moments.py>>
        +19D raw transform
        +raw MRT
        +19D NOCM transform
        +raw/NOCM force source
    }

    class Collisions {
        <<module collisions.py>>
        +SRT
        +TRT
        +HOME NOCM MRT
    }

    class Encoder {
        <<module encoding.py>>
        +FullF
        +HOME projection
        +HOME reconstruction
    }

    class LbmStateBase
    class FullFLbmState
    class HomeLbmState

    LbmStateBase <|-- FullFLbmState
    LbmStateBase <|-- HomeLbmState
    LbmModel --> Contracts
    LbmSolver --> Contracts
    LbmSolver --> StateProvider
    LbmSolver --> BoundaryCompletion
    LbmSolver --> ForcePipeline
    LbmSolver --> MomentPipeline
    LbmSolver --> Collisions
    LbmSolver --> Encoder
    LbmSolver --> LbmStateBase
```

## 依赖方向

```text
LbmModel / contracts
        ↓
StateProvider → BoundaryCompletion → Collector
                                      ↓
                            ForceProvider → HydroClosure
                                      ↓
                              CollisionOperator
                                      ↓
                                   Encoder
```

边界模块只输出完整 population，不依赖 `CollisionSpace`。碰撞模块消费统一
`CollisionContext`，并负责把同一个物理力 `F` 翻译为 population、raw moment、
NOCM moment 或 HOME closure source。
