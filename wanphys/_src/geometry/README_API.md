# 课题一 碰撞检测：阶段一新增接口

> **迁移说明 / Migration Note**:
>
> 此模块已迁移至 `wanphys/_src/geometry/` 以遵循 WanPhys 模块模式。
> This module has been migrated to `wanphys/_src/geometry/` to follow WanPhys module pattern.
>
> **推荐导入方式 / Recommended imports**:
>
> - 外部用户 / External users: `from wanphys import CollisionTriMeshStyle3D`
> - 模块特定 / Module-specific: `from wanphys.geometry import CollisionTriMeshStyle3D`
> - 内部开发 / Internal dev: `from wanphys._src.geometry import CollisionTriMeshStyle3D`

**本文档提供了碰撞检测部分的第一阶段的两部分新增接口**

* **阶段一碰撞检测接口**
* **点云采样接口**

## 1 碰撞检测接口

### 1.1 碰撞域后端

**每类碰撞域实现一个 Backend。Backend 负责候选生成与窄相计算**

**Backend 作为基类定义抽象方法接口供具体的碰撞类方法实现**

**相关文件路径**：

* `wanphys/_src/geometry/collision_backend.py`

**CollisionBackend 抽象类**

```
from __future__ import annotations
from abc import ABC, abstractmethod

class CollisionBackend(ABC):
    """Base class for collision backends (candidate generation + narrowphase)."""

    name: str = "base"

    @abstractmethod
    def build(self, model, device):
        """Initialize backend-specific data structures."""

    @abstractmethod
    def refit(self, state):
        """Update acceleration structures with new state."""

    @abstractmethod
    def generate_candidates(self, state, params, dt: float, out_pairs, out_pair_count) -> None:
        """Generate candidate pairs for narrowphase."""

    @abstractmethod
    def narrow_phase(self, state, params, dt: float, pairs, pair_count, out_hits, mode: str) -> None:
        """Compute contact geometry for candidate pairs."""
```

### 1.2 可变形体碰撞检测接口

#### 1.2.1 相关说明

**相关 Solver 中内置粒子类型组成的可变形体布料之间的碰撞**

该碰撞内窄相耦合接触信息与力求解，现将其初步解耦并集成到 `geometry` 模块中

**相关实现文件路径**：

* `wanphys/_src/geometry/collision_trimesh_style3d.py`
* `wanphys/_src/geometry/bvh.py`
* `wanphys/_src/geometry/collision_backend.py`
* `wanphys/_src/geometry/tests/collision_trimesh_style3d`（测试用例）

#### 1.1.2 新增结构

在原本 Style3D Solver 内置的碰撞类上新增字段并继承碰撞后端类，当前命名为 `CollisionTriMeshStyle3D`

`CollisionTriMeshStyle3D` 使用示例

```
from wanphys.geometry import CollisionTriMeshStyle3D

collision = CollisionTriMeshStyle3D(model) # 初始化
collision.generate_candidates(state, params=None, dt=dt, out_pairs=None, out_pair_count=None) # 生成候选对
collision.narrow_phase(state, params=None, dt=dt, pairs=None, pair_count=None, out_hits=None, mode="discrete") # 生成接触
```

**碰撞类内新增三个字段用于存储窄相接触信息**

1. `ContactVertexTriangle` 用于存储 Vertex-Triangle 接触信息

```
@dataclass
class ContactVertexTriangle:
    max_contacts: int # 最多候选数量
    contact_count: wp.array # 实际碰撞数
    contact_fid: wp.array # 碰撞面 id
    contact_normal: wp.array # 法线
    contact_dist: wp.array # 距离
    contact_bary: wp.array # 重心坐标
    contact_point: wp.array # 碰撞点
    contact_penetration: wp.array # 穿透距离

    @classmethod
    def allocate(cls, particle_count: int, max_contacts: int, device: Devicelike) -> "ContactVertexTriangle":
        return cls(
            max_contacts=max_contacts,
            contact_count=wp.array(shape=(particle_count), dtype=int, device=device),
            contact_fid=wp.array(shape=(max_contacts, particle_count), dtype=int, device=device),
            contact_normal=wp.array(shape=(max_contacts, particle_count), dtype=wp.vec3, device=device),
            contact_dist=wp.array(shape=(max_contacts, particle_count), dtype=float, device=device),
            contact_bary=wp.array(shape=(max_contacts, particle_count), dtype=wp.vec3, device=device),
            contact_point=wp.array(shape=(max_contacts, particle_count), dtype=wp.vec3, device=device),
            contact_penetration=wp.array(shape=(max_contacts, particle_count), dtype=float, device=device),
        )
```

2. `ContactEdgeEdge` 用于存储 Edge-Edge 接触信息

```
@dataclass
class ContactEdgeEdge:
    max_contacts: int # 最多候选数量
    contact_count: wp.array # 实际碰撞数
    contact_eid: wp.array # 碰撞边 id
    contact_s: wp.array # 最近点参数
    contact_t: wp.array # 最近点参数
    contact_dir: wp.array # 单位方向向量
    contact_dist: wp.array # 边边距离
    contact_limit: wp.array # 厚度上限
    contact_point: wp.array # 碰撞点
    contact_penetration: wp.array # 穿透深度

    @classmethod
    def allocate(cls, edge_count: int, max_contacts: int, device: Devicelike) -> "ContactEdgeEdge":
        return cls(
            max_contacts=max_contacts,
            contact_count=wp.array(shape=(edge_count), dtype=int, device=device),
            contact_eid=wp.array(shape=(max_contacts, edge_count), dtype=int, device=device),
            contact_s=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_t=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_dir=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_dist=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_limit=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
            contact_point=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_penetration=wp.array(shape=(max_contacts, edge_count), dtype=float, device=device),
        )
```

3. `ContactEdgeFace` 用于存储 Edge-Face 接触信息

```
@dataclass
class ContactEdgeFace:
    max_contacts: int
    contact_count: wp.array
    contact_fid: wp.array
    contact_dir: wp.array
    contact_bary: wp.array
    contact_edge_bary: wp.array
    contact_point: wp.array

    @classmethod
    def allocate(cls, edge_count: int, max_contacts: int, device: Devicelike) -> "ContactEdgeFace":
        return cls(
            max_contacts=max_contacts,
            contact_count=wp.array(shape=(edge_count), dtype=int, device=device),
            contact_fid=wp.array(shape=(max_contacts, edge_count), dtype=int, device=device),
            contact_dir=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_bary=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
            contact_edge_bary=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec2, device=device),
            contact_point=wp.array(shape=(max_contacts, edge_count), dtype=wp.vec3, device=device),
        )
```

#### 1.1.3 接口1：`generate_candidate`

**函数签名：**

```
def generate_candidates(self, 
                        state: State, 
                        params, 
                        dt: float, 
                        out_pairs, 
                        out_pair_count)
```

**输入规则：**

* `state`：当前的状态
* `params`：额外参数
* `dt`：物理帧时间
* `out_pairs`：输出候选对
* `out_pair_count`：输出候选对数量

**输出规则：**

**分别针对 Vertex-Triangle，Edge-Edge，Edge-Face 生成三类候选对**

计划是可通过 `out_pairs` 拿到，目前在内部的数据结构如下

```
self.broad_phase_ee = wp.array(shape=(self.max_contacts_count, model.edge_count), dtype=int, device=self.model.device)

self.broad_phase_ef = wp.array(shape=(self.max_contacts_count, model.edge_count), dtype=int, device=self.model.device)

self.broad_phase_vf = wp.array(shape=(self.max_contacts_count, model.particle_count), dtype=int, device=self.model.device)
```

#### 1.1.4 接口2：`narrow_phase`

**函数签名：**

```
def narrow_phase(self, 
                state: State, 
                params, 
                dt: float, 
                pairs, 
                pair_count, 
                out_hits, 
                mode: str)
```

**输入规则：**

* `state`：当前的状态
* `params`：额外参数
* `dt`：物理帧时间
* `pairs`：输入候选对
* `pair_count`：输入候选对数量
* `out_hits`：输出接触
* `mode`：离散或连续碰撞模式

**输出规则：**

**分别针对 Vertex-Triangle，Edge-Edge，Edge-Face 生成三类接触信息**

**存储格式见 **`ContactVertexTriangle` ，`ContactEdgeEdge`，`ContactEdgeFace` 定义

**计划是可通过 **`out_hits` 拿到，目前作为碰撞类内信息存储

```
self.vf_contacts = ContactVertexTriangle.allocate(
    model.particle_count,
    self.max_contacts_count,
    self.model.device,
)

self.ee_contacts = ContactEdgeEdge.allocate(
    model.edge_count,
    self.max_contacts_count,
    self.model.device,
)

self.ef_contacts = ContactEdgeFace.allocate(
    model.edge_count,
    self.max_contacts_count,
    self.model.device,
)
```

## 2 点云采样接口

### 2.1 依赖与文件组织

**模块说明：**

* **原 Newton 中没有该模块，属于新增功能**
* **抽取并封装 mesh -> pts（点云采样）能力，提供两个可调用接口**

**依赖：**

* **Python 3.10+**
* **warp-lang（NVIDIA Warp）**
* **numpy**

**文件目录：**

* `wanphys/_src/geometry/mesh_to_points_warp.py`（原实现）
* `wanphys/_src/geometry/mesh_to_points_api.py`（接口封装）
* `wanphys/_src/geometry/examples/mesh_to_points_demo.py` (调用示例)

### 2.2 返回结构

**部分接口新增返回结构**： `MeshToPointsResult`

**对应字段含义**：

* `points: (N,3) float32`，输出点坐标（若 layers>1 则为多层拼接）
* `normals: (N,3) float32`，点法线
* `face_id: (N,) int32`，每个点所属三角面索引
* `layer_id: (N,) int32`，层编号 0..layers-1
* `radius_used`: 实际用于采样的半径（`normalize=True` 时为 `radius * scale`）
* `layer_gap_used`: 实际层间距（`normalize=True` 时为 `layer_gap * scale`）
* `layers`: 生成的层数
* `scale`: normalize 的缩放系数（`normalize=False` 时为 `1.0`）
* `normalized`: 是否做了 normalize

### 2.3 多层采样语义（`layers + layer_gap`）

**API 的多层采样是：对每一层独立做一次表面 Poisson 采样（不同 seed），然后把该层的点沿法线方向偏移：**

`layer k` 的偏移量 = `k * layer_gap`（`inward=True` 时沿 -normal，`inward=False` 时沿 +normal

### 2.4 接口 1：`mesh_to_points`（内存转换）

**函数签名：**

```
def mesh_to_points(
    V: np.ndarray,
    F: np.ndarray,
    radius: float,
    point_count: int,
    *,
    device: str = "cuda:0",
    seed: int = 0,
    normalize: bool = True,
    # multilayer controls
    layers: int = 1,
    layer_gap: float = 0.0,
    inward: bool = True,
    # sampler knobs
    max_rounds: int = 200,
    batch_factor: int = 8,
    batch_min: int = 65536,
    batch_max: int = 1_048_576,
) -> MeshToPointsResult:
```

**输入规则：**

* `V: (nV, 3) float32`,  输入Mesh的顶点坐标
* `F: (nF, 3) float32`,  输入Mesh的三角形顶点索引（指向 V 的行号）
* `radius: float`,  **表面采样**的 Poisson-like 最小距离阈值
* `point_count：int`，**每一层**期望采样点数（表面层 + 每个内层各自的目标点数）
* `device：str`, Warp/GPU 设备字符串，例如 `"cuda:0"`。warp 采样函数就是以 device 作为参数
* `seed：int`, 随机种子基准。多层时通常会对不同层做 seed 偏移以避免层间完全同分布
* `normalize：bool`, 是否对 mesh 做归一化
* `layers：int`, 生成层数。语义：
  * `layer_id=0`：表面层（原始实现表面点 `LayerID=0`）
  * `layer_id=k (>=1)`：第 k 个内层（原始实现就是这么写入的）
* `layer_gap：float`, 相邻层之间的间距
* `inward：bool`, `True` 表示“向内”（沿 -normal 方向偏移），`False` 表示“向外”（沿 +normal）

**输出规则：**

**见 **`MeshToPointsResult` 定义

**调用示例：**

**1 层向内采样：**

```
import numpy as np
from wanphys._src.geometry.mesh_to_points_api import mesh_to_points

res = mesh_to_points(
    V, F,
    radius=0.01,
    point_count=200000,   # 每层点数
    layers=1,
    layer_gap=0.003,
    inward=True,
    device='cuda:0',
)
P, N, fid, lid = res.points, res.normals, res.face_id, res.layer_id
```

### 2.5 接口 2：`obj_to_outputs`（OBJ -> 可选输出）

**函数签名：**

```
def obj_to_outputs(
    obj_path: str,
    *,
    ply_path: Optional[str] = None,
    txt_path: Optional[str] = None,
    radius: float,
    point_count: int,
    device: str = "cuda:0",
    seed: int = 0,
    normalize: bool = True,
    # multilayer controls
    layers: int = 1,
    layer_gap: float = 0.0,
    inward: bool = True,
    # sampler knobs
    max_rounds: int = 200,
    batch_factor: int = 8,
    batch_min: int = 65536,
    batch_max: int = 1_048_576,
) -> MeshToPointsResult:
```

**输入规则：**

* `obj_path` 为输入的 OBJ 文件路径

**输出规则：**

* `ply_path` 不为空（非 `None` 且非 `''`）则输出 PLY（ASCII）文件：`x y z nx ny nz layer_id face_id`
* `txt_path` 不为空（非 `None` 且非 `''`）则输出 TXT 文件：`header + x y z layer_id face_id`
* **两者都提供则同时输出两种格式**
* **两者都为空则不写文件，但仍返回内存结果**

**调用示例：**

**OBJ 文件 -> 同时输出 PLY + TXT 文件，生成 1 层向内点云：**

```
from wanphys._src.geometry.mesh_to_points_api import obj_to_outputs

obj_to_outputs(
    'input.obj',
    ply_path='output.ply',
    txt_path='output.txt',
    radius=0.01,
    point_count=200000,   # 每层点数
    layers=1,
    layer_gap=0.0025,
    inward=True,
    device='cuda:0',
)
```
