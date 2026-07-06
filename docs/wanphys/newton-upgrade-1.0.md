# Newton 0.2.0 → 1.0.0 升级总结

**分支：** `upgrade/newton-1.0`（基于 `dev`）

---

## 做了什么

1. **替换 newton/ 目录**：用 `git read-tree` 将 Newton upstream `release-1.0` 分支的 `newton/` 子树直接覆盖到本仓库的 `newton/` 目录，共涉及 549 个文件的新增、修改和删除。

2. **对齐 pyproject.toml**：将 `pyproject.toml` 替换为 Newton 1.0 官方版本（仅保留 `warp-lang = { index = "nvidia" }` 这一条 WanPhys 专属的 index 覆盖），并重新生成 `uv.lock`。

3. **修复 WanPhys 适配层**：针对 Newton 1.0 的多项破坏性 API 变更，逐一修复 `wanphys/_src/` 下的适配代码（详见下方各条）。

---

## API 破坏性变更清单

### ① `key` 参数全面改名为 `label`（Builder 接口）

Newton 1.0 将 `ModelBuilder` 中所有方法的 `key` 参数统一重命名为 `label`，涉及：

```
add_body / add_link / add_articulation
add_shape_box / sphere / capsule / cylinder / mesh
add_joint_revolute / add_joint_fixed
add_ground_plane
```

**影响范围**：任何直接调用 Newton `ModelBuilder` 的代码。
**WanPhys 修复**：`wanphys/_src/rigid/builder.py` 的公开构建接口已对齐为 `label=`，内部继续转发到 Newton 的 `label` 字段。
**各团队注意**：若在自己的 `_src/[topic]/` 目录下有直接调用 Newton builder 的代码，须自行排查。

---

### ② `Model.shape_key` / `joint_key` / `body_key` 改名为 `shape_label` / `joint_label` / `body_label`

`Model` 对象上原有的属性名（用于按名称查找 shape/joint/body 索引）也同步改名。

```python
# 旧
model.shape_key["/env/Flap"]
model.joint_key["/env/Hinge"]

# 新
model.shape_label["/env/Flap"]
model.joint_label["/env/Hinge"]
```

---

### ③ `populate_contacts` 接口移除

`newton.sensors.populate_contacts(contacts, solver)` 在 Newton 1.0 中已删除。
替代方法：在 `domain.step()` 之后调用 `domain.update_contacts(contacts)`，通过 WanPhys 域接口将接触数据写入 `Contacts` 对象。

```python
# 旧
populate_contacts(self.contacts, self.solver)

# 新
self.domain.update_contacts(self.contacts)
```

---

### ④ `SensorContact` 接口变更

**构造函数**：所有参数改为关键字参数（不再接受位置参数）；移除了 `match_fn` 和 `prune_noncolliding` 参数。

**`update` 替代 `eval`**：原来的 `sensor.eval(contacts)` 改为 `sensor.update(contacts)`。

```python
# 旧
sensor.eval(contacts)

# 新
sensor.update(contacts)
```

**`sensing_objs` / `counterparts` 结构变更**：原来是 `list[int]`（形状索引的列表），现在是 `list[list[tuple[int, ObjectType]]]`（按"世界"分层，每个元素是 `(shape_index, ObjectType)` 元组）。

```python
# 旧
for i, shape in enumerate(sensor.sensing_objs):
    ...

# 新
for i, (shape, obj_type) in enumerate(sensor.sensing_objs[0]):  # [0] = world 0
    ...
```

---

### ⑤ `Contacts` 必须在 `SensorContact` 之后创建

Newton 1.0 要求：先创建所有 `SensorContact`（传感器会通过 `model.request_contact_attributes()` 注册对 `force` 字段的需求），再创建 `Contacts`（此时才会分配 force 缓冲区）。顺序反了会报错。

```python
# 正确顺序
self.flap_sensor = SensorContact(self.model, ...)
self.plate_sensor = SensorContact(self.model, ...)

self.contacts = Contacts(
    rigid_contact_max=njmax,
    soft_contact_max=0,
    requested_attributes=self.model.get_requested_contact_attributes(),
    device=self.model.device,
)
```

另外，`rigid_contact_max` 必须 `>= MuJoCo solver 的 njmax`，否则 MuJoCo 接触数据会被截断，传感器读不到力。

---

### ⑥ `mujoco-warp` 版本必须为 `3.5.0.2`（不是 `3.5.0`）

Newton 1.0 内部调用了 `mujoco_warp.set_length_range()`，该函数只存在于 `3.5.0.2`（patch 版本），`3.5.0` 无此接口，运行时会 `AttributeError`。

`pyproject.toml` 中已正确固定：`mujoco-warp==3.5.0.2`

---

### ⑦ `pyproject.toml` 主要变化

| 项目 | 0.2.0 | 1.0.0 |
|------|-------|-------|
| `authors` email | `warp-python@nvidia.com` | `developers@newton-physics.org` |
| `Development Status` | `4 - Beta` | `5 - Production/Stable` |
| Python 分类器 | 3.10–3.12 | 3.10–3.13 |
| CUDA 分类器 | `:: 12` | `:: 12`, `:: 13` |
| `sim` extra | 含 `cbor2` | 不含 `cbor2` |
| `importers` | 少 `requests`, `meshio`, `newton-usd-schemas` | 含以上三项 |
| `examples` | 少 `cbor2`, `Pillow`；pyglet 无上界 | 含两项；pyglet `<3` |
| 新增 `remesh` extra | — | `open3d`, `pyfqmr` |
| 新增 `torch-cu13` extra | — | ✓ |
| `docs` extra | 少 notebook 相关 | 含 `nbsphinx`, `ipykernel`, `pypandoc`, `viser` |
| `[tool.uv]` conflicts | — | torch-cu12 与 cu13 互斥 |
| Ruff rule | 无 `TID253` | 含 `TID253` + banned imports 列表 |

---

## 其他注意事项

### YAML 传感器配置格式变更

传感器配置文件（`cfg/*.yaml`）中的 shape 匹配模式由正则表达式改为 fnmatch（通配符）格式，且多个 pattern 改用列表：

```yaml
# 旧
sensing_obj_shapes: ".*Plate.*"
counterpart_shapes: ".*Cube.*|.*Sphere.*"
match_fn: re_match

# 新
sensing_obj_shapes: "*Plate*"
counterpart_shapes: ["*Cube*", "*Sphere*"]
```

### warp-lang 升至 1.12.0

旧版本依赖的是 dev 构建（`1.11.0.dev20251205`），1.0 起要求正式版 `>=1.12.0`。Warp 新版本有若干 API 标记为即将废弃（`warp.types.float32`、`warp.context.Device` 等），目前只是警告，暂不影响运行。

### Newton 1.0 新增内容（不影响现有代码）

- `kamino` 求解器（全新关节动力学）
- `viser` 查看器后端
- `newton.math` 模块
- 显式 MPM 模型
- `torch-cu13` 支持

### Windows 下偶发测试报错

`test_free_fall_velocity` 和 `test_torque_angular_velocity` 在 Windows 下偶尔 ERROR，根因是 Warp NVRTC 编译缓存文件锁冲突（`.pch` 文件无法删除），与 Newton API 无关。Linux/CI 环境下预期不会复现。

---

## 合并前建议

- 各团队检查自己目录下有无直接调用 Newton builder，或直接访问 `shape_key`/`joint_key` 的地方
- 确认 `SensorContact` 创建顺序在 `Contacts` 之前
- 在 CI（Linux + GPU）上完整跑一遍 `uv run --extra dev -m newton.tests` 和 WanPhys 测试，排除 Windows 环境噪音
