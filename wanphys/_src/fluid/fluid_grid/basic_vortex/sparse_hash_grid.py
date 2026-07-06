import warp as wp
import numpy as np

# =========================================================================== #
# 核心常量定义
# =========================================================================== #
BLOCK_SIZE = 8
BLOCK_VOL = BLOCK_SIZE ** 3  # 512

# 哈希表大小 (必须是 2 的幂，2^18 = 262144)
HASH_BITS = 18
HASH_SIZE = 1 << HASH_BITS
HASH_MASK = HASH_SIZE - 1

# 哈希槽状态标志
EMPTY_KEY = -1  #用于hash_keys
UNALLOCATED = -1  #用于hash_vals
REQUESTED = -2  #用于hash_vals

# 邻居方向枚举 (左, 右, 下, 上, 后, 前)
DIR_NEG_X = 0
DIR_POS_X = 1
DIR_NEG_Y = 2
DIR_POS_Y = 3
DIR_NEG_Z = 4
DIR_POS_Z = 5


# =========================================================================== #
# 1. 底层哈希与寻址算法 (Device Functions)
# =========================================================================== #


#atomic_cas仅支持int32的向量加锁，因此我们需要将3D坐标压缩成一个整数作为某个Block的key
@wp.func
def pack_key(bx: int, by: int, bz: int) -> int:
    """将 3D Block 坐标压缩进一个 32 位整数 (假设各轴坐标在 0~1023 之间)"""
    x = bx & 1023
    y = by & 1023
    z = bz & 1023
    return (x << 20) | (y << 10) | z

#还原key原来的block物理坐标
@wp.func
def unpack_key(key: int) -> wp.vec3i:
    """从 32 位整数还原 3D Block 坐标"""
    bx = (key >> 20) & 1023
    by = (key >> 10) & 1023
    bz = key & 1023
    return wp.vec3i(bx, by, bz)

#将Block的坐标映射到哈希表的位置中
@wp.func
def hash_func(bx: int, by: int, bz: int) -> int:
    """空间哈希函数 (Cirrus 论文同款常数)"""
    hx = wp.uint32(bx * 1)
    hy = wp.uint32(by * 2654435761)
    hz = wp.uint32(bz * 805459861)
    h = wp.bit_xor(wp.bit_xor(hx, hy), hz)
    return int(h) & HASH_MASK

#哈希查表 算哈希表id
@wp.func
def hash_lookup(
        bx: int, by: int, bz: int,
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int)
) -> int:
    """查表：根据 Block 坐标寻找物理 ID (仅在拓扑重建时调用)"""
    packed = pack_key(bx, by, bz)
    h = hash_func(bx, by, bz)

    # 线性探查
    for i in range(100):
        idx = (h + i) & HASH_MASK
        k = hash_keys[idx]
        if k == packed:
            return hash_vals[idx]  # 找到真实的 Block_ID
        if k == EMPTY_KEY:
            return -1  # 槽位为空，该块不存在
    return -1


# =========================================================================== #
# 2. 拓扑重建 Kernels (每帧调用一次)
# =========================================================================== #

# 纯网格的 Pass 1：网格自膨胀 (Dilation)
# 依据上一帧流体位置，预测这一帧流体动向
@wp.kernel
def pass1_request_halo_blocks(
        old_block_coords: wp.array(dtype=wp.vec3i),
        old_active_count: int,
        max_bx: int,
        max_by: int,
        max_bz: int,
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int)
):
    #launch old_active_count个线程，每个线程负责请求一个上一帧的活跃Block以及它周围的邻居Block
    block_idx = wp.tid()
    if block_idx >= old_active_count:
        return

    # 获取上一帧的活跃 Block 坐标
    coord = old_block_coords[block_idx]
    bx, by, bz = coord[0], coord[1], coord[2]

    # 【核心逻辑】：不仅请求自己，还要请求周围的 6 个邻居 Block！
    # (如果你的 CFL 条件允许水在一帧内流过 1 个 Block，这就足够了。
    #  如果流速极快，可能需要请求 26 个邻居)

    # 辅助函数：尝试请求一个 Block (把之前原代码的逻辑提取出来)
    # request_single_block(bx, by, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    # request_single_block(bx + 1, by, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    # request_single_block(bx - 1, by, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    # request_single_block(bx, by + 1, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    # request_single_block(bx, by - 1, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    # request_single_block(bx, by, bz + 1, max_bx, max_by, max_bz, hash_keys, hash_vals)
    # request_single_block(bx, by, bz - 1, max_bx, max_by, max_bz, hash_keys, hash_vals)
    for ox in range(-1, 2):
        for oy in range(-1, 2):
            for oz in range(-1, 2):
                request_single_block(
                    bx + ox, by + oy, bz + oz,
                    max_bx, max_by, max_bz,
                    hash_keys, hash_vals
                )


@wp.func
def request_single_block(
        bx: int, by: int, bz: int,
        max_bx: int, max_by: int, max_bz: int,
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int)
):
    if bx < 0 or by < 0 or bz < 0 or bx >= max_bx or by >= max_by or bz >= max_bz:
        return

    packed = pack_key(bx, by, bz)
    h = hash_func(bx, by, bz)

    for i in range(100):
        idx = (h + i) & HASH_MASK
        # 通过原子比较交换 (Atomic Compare-And-Swap) 来安全地在哈希表中插入请求
        # 返回值 = wp.atomic_cas(数组, 索引, 期待值, 新值)
        # 如果是EMPTY 赋值packed，如果不是，不赋值，都返回原来已有值
        prev_key = wp.atomic_cas(hash_keys, idx, EMPTY_KEY, packed)
        # 如果值恰好是packed
        if prev_key == EMPTY_KEY or prev_key == packed:
            wp.atomic_cas(hash_vals, idx, UNALLOCATED, REQUESTED)
            return

# add new request rule
@wp.func
def block_pool_index(block_id: int, lx: int, ly: int, lz: int) -> int:
    return block_id * BLOCK_VOL + lx * BLOCK_SIZE * BLOCK_SIZE + ly * BLOCK_SIZE + lz


# MODIFIED: boundary-driven topology now resolves the old pool slot from the
# old topology hash using the block's coordinates instead of assuming the
# pruned block list index still matches the pool layout.
@wp.func
def voxel_is_significant(
        bx: int,
        by: int,
        bz: int,
        lx: int,
        ly: int,
        lz: int,
        old_hash_keys: wp.array(dtype=int),
        old_hash_vals: wp.array(dtype=int),
        density: wp.array(dtype=float),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        density_threshold: float,
        velocity_threshold: float,
) -> int:
    block_id = hash_lookup(bx, by, bz, old_hash_keys, old_hash_vals)
    if block_id < 0:
        return 0

    idx = block_pool_index(block_id, lx, ly, lz)
    if density[idx] > density_threshold:
        return 1
    if wp.abs(vel_u[idx]) > velocity_threshold:
        return 1
    if wp.abs(vel_v[idx]) > velocity_threshold:
        return 1
    if wp.abs(vel_w[idx]) > velocity_threshold:
        return 1
    return 0


# MODIFIED: only request a neighbor when the source face has both enough
# boundary density and a normal velocity that actually points outward.
@wp.func
def face_needs_neighbor(
        bx: int,
        by: int,
        bz: int,
        face_dir: int,
        old_hash_keys: wp.array(dtype=int),
        old_hash_vals: wp.array(dtype=int),
        density: wp.array(dtype=float),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        density_threshold: float,
        velocity_threshold: float,
) -> int:
    block_id = hash_lookup(bx, by, bz, old_hash_keys, old_hash_vals)
    if block_id < 0:
        return 0

    for a in range(BLOCK_SIZE):
        for b in range(BLOCK_SIZE):
            lx = int(0)
            ly = int(0)
            lz = int(0)

            if face_dir == DIR_NEG_X:
                lx = 0
                ly = a
                lz = b
            elif face_dir == DIR_POS_X:
                lx = BLOCK_SIZE - 1
                ly = a
                lz = b
            elif face_dir == DIR_NEG_Y:
                lx = a
                ly = 0
                lz = b
            elif face_dir == DIR_POS_Y:
                lx = a
                ly = BLOCK_SIZE - 1
                lz = b
            elif face_dir == DIR_NEG_Z:
                lx = a
                ly = b
                lz = 0
            else:
                lx = a
                ly = b
                lz = BLOCK_SIZE - 1

            idx = block_pool_index(block_id, lx, ly, lz)
            if density[idx] <= density_threshold:
                continue

            normal_vel = float(0.0)

            if face_dir == DIR_NEG_X:
                normal_vel = vel_u[idx]
                if normal_vel >= -velocity_threshold:
                    continue
            elif face_dir == DIR_POS_X:
                normal_vel = vel_u[idx]
                if normal_vel <= velocity_threshold:
                    continue
            elif face_dir == DIR_NEG_Y:
                normal_vel = vel_v[idx]
                if normal_vel >= -velocity_threshold:
                    continue
            elif face_dir == DIR_POS_Y:
                normal_vel = vel_v[idx]
                if normal_vel <= velocity_threshold:
                    continue
            elif face_dir == DIR_NEG_Z:
                normal_vel = vel_w[idx]
                if normal_vel >= -velocity_threshold:
                    continue
            else:
                normal_vel = vel_w[idx]
                if normal_vel <= velocity_threshold:
                    continue

            if voxel_is_significant(
                    bx,
                    by,
                    bz,
                    lx,
                    ly,
                    lz,
                    old_hash_keys,
                    old_hash_vals,
                    density,
                    vel_u,
                    vel_v,
                    vel_w,
                    density_threshold,
                    velocity_threshold,
            ) == 1:
                return 1

    return 0


# MODIFIED: boundary-driven pass now inspects old block contents through the
# old hash table so pruning the coordinate list no longer invalidates the
# topology-growth decision.
@wp.kernel
def pass1_request_boundary_halo_blocks(
        old_block_coords: wp.array(dtype=wp.vec3i),
        old_active_count: int,
        max_bx: int,
        max_by: int,
        max_bz: int,
        old_hash_keys: wp.array(dtype=int),
        old_hash_vals: wp.array(dtype=int),
        density: wp.array(dtype=float),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        density_threshold: float,
        velocity_threshold: float,
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int)
):
    block_idx = wp.tid()
    if block_idx >= old_active_count:
        return

    coord = old_block_coords[block_idx]
    bx, by, bz = coord[0], coord[1], coord[2]

    # Always keep the current active block itself.
    request_single_block(bx, by, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)

    # Only request a face-neighbor block when the corresponding boundary voxels
    # already carry noticeable density or velocity.
    if face_needs_neighbor(
            bx, by, bz,
            DIR_NEG_X,
            old_hash_keys, old_hash_vals,
            density, vel_u, vel_v, vel_w,
            density_threshold, velocity_threshold,
    ) == 1:
        request_single_block(bx - 1, by, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    if face_needs_neighbor(
            bx, by, bz,
            DIR_POS_X,
            old_hash_keys, old_hash_vals,
            density, vel_u, vel_v, vel_w,
            density_threshold, velocity_threshold,
    ) == 1:
        request_single_block(bx + 1, by, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    if face_needs_neighbor(
            bx, by, bz,
            DIR_NEG_Y,
            old_hash_keys, old_hash_vals,
            density, vel_u, vel_v, vel_w,
            density_threshold, velocity_threshold,
    ) == 1:
        request_single_block(bx, by - 1, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    if face_needs_neighbor(
            bx, by, bz,
            DIR_POS_Y,
            old_hash_keys, old_hash_vals,
            density, vel_u, vel_v, vel_w,
            density_threshold, velocity_threshold,
    ) == 1:
        request_single_block(bx, by + 1, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)
    if face_needs_neighbor(
            bx, by, bz,
            DIR_NEG_Z,
            old_hash_keys, old_hash_vals,
            density, vel_u, vel_v, vel_w,
            density_threshold, velocity_threshold,
    ) == 1:
        request_single_block(bx, by, bz - 1, max_bx, max_by, max_bz, hash_keys, hash_vals)
    if face_needs_neighbor(
            bx, by, bz,
            DIR_POS_Z,
            old_hash_keys, old_hash_vals,
            density, vel_u, vel_v, vel_w,
            density_threshold, velocity_threshold,
    ) == 1:
        request_single_block(bx, by, bz + 1, max_bx, max_by, max_bz, hash_keys, hash_vals)


@wp.kernel
def pass2_allocate_blocks(
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        block_counter: wp.array(dtype=int),
        block_coords: wp.array(dtype=wp.vec3i),
        max_blocks: int
):
    tid = wp.tid()
    key = hash_keys[tid]

    if key != -1:
        block_id = wp.atomic_add(block_counter, 0, 1)

        if block_id < max_blocks:
            hash_vals[tid] = block_id
            # 解码坐标
            bx = (key >> 20) & 0x3FF
            by = (key >> 10) & 0x3FF
            bz = key & 0x3FF
            # 还原符号
            if bx >= 512: bx -= 1024
            if by >= 512: by -= 1024
            if bz >= 512: bz -= 1024

            block_coords[block_id] = wp.vec3i(bx, by, bz)
        else:
            hash_vals[tid] = -1

@wp.kernel
def pass3_build_neighbor_cache(
        block_coords: wp.array(dtype=wp.vec3i),
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int),
        neighbor_cache: wp.array2d(dtype=int),
        active_blocks: wp.array(dtype=int)
):
    """阶段 3 (终极优化)：为每一个激活的物理块，预先查好它 6 个邻居的物理 ID"""
    block_id = wp.tid()
    if block_id >= active_blocks[0]:
        return

    coord = block_coords[block_id]
    bx, by, bz = coord[0], coord[1], coord[2]

    # 将哈希查询的代价提前支付，并缓存起来
    neighbor_cache[block_id, DIR_NEG_X] = hash_lookup(bx - 1, by, bz, hash_keys, hash_vals)
    neighbor_cache[block_id, DIR_POS_X] = hash_lookup(bx + 1, by, bz, hash_keys, hash_vals)
    neighbor_cache[block_id, DIR_NEG_Y] = hash_lookup(bx, by - 1, bz, hash_keys, hash_vals)
    neighbor_cache[block_id, DIR_POS_Y] = hash_lookup(bx, by + 1, bz, hash_keys, hash_vals)
    neighbor_cache[block_id, DIR_NEG_Z] = hash_lookup(bx, by, bz - 1, hash_keys, hash_vals)
    neighbor_cache[block_id, DIR_POS_Z] = hash_lookup(bx, by, bz + 1, hash_keys, hash_vals)


# =========================================================================== #
# 3. Python 封装类
# =========================================================================== #

class SparseHashGrid:
    def __init__(self, max_blocks: int, device, domain_blocks: tuple[int, int, int]):
        self.device = device
        self.max_blocks = max_blocks
        self.domain_blocks = domain_blocks

        # --- 拓扑目录 (只需要一套当前帧的即可，因为用完就清空重盖) ---
        self.hash_keys = wp.full(HASH_SIZE, EMPTY_KEY, dtype=int, device=device)
        self.hash_vals = wp.full(HASH_SIZE, UNALLOCATED, dtype=int, device=device)
        self.block_coords = wp.empty(max_blocks, dtype=wp.vec3i, device=device)
        self.neighbor_cache = wp.full((max_blocks, 6), -1, dtype=int, device=device)
        self.block_counter = wp.zeros(1, dtype=int, device=device)

        # 为了平流，我们需要额外保存一份上一帧的哈希目录！
        self.old_hash_keys = wp.empty(HASH_SIZE, dtype=int, device=device)
        self.old_hash_vals = wp.empty(HASH_SIZE, dtype=int, device=device)

        # --- 物理数据池 (必须准备两套：Ping-Pong 双缓冲) ---
        pool_len = max_blocks * BLOCK_VOL

        # 密度场 (Level Set 或 烟雾浓度)
        self.pool_density_old = wp.zeros(pool_len, dtype=float, device=device)
        self.pool_density_new = wp.zeros(pool_len, dtype=float, device=device)

        # 速度场
        self.pool_velocity_old = wp.zeros(pool_len, dtype=wp.vec3, device=device)
        self.pool_velocity_new = wp.zeros(pool_len, dtype=wp.vec3, device=device)

    def update_topology(self, old_block_coords: wp.array, old_active_count: int):
        """纯网格架构下的每帧拓扑更新 (自膨胀模式)"""

        # 0. 保护性检查
        if old_active_count == 0:
            return # 如果世界里一滴水都没有，就不需要更新了

        #先备份
        wp.copy(self.old_hash_keys, self.hash_keys)
        wp.copy(self.old_hash_vals, self.hash_vals)
        # 1. 大扫除 (清空目录和计数器，为新一帧腾位置)
        self.hash_keys.fill_(EMPTY_KEY)
        self.hash_vals.fill_(UNALLOCATED)
        self.block_counter.fill_(0)

        # 2. Pass 1: 网格自膨胀 (请求自己及周围 6 个邻居)
        wp.launch(
            kernel=pass1_request_halo_blocks,
            dim=old_active_count, # 核心修改：线程数等于上一帧存活的 Block 数量！
            inputs=[
                old_block_coords,
                old_active_count,
                self.domain_blocks[0],
                self.domain_blocks[1],
                self.domain_blocks[2],
                self.hash_keys,
                self.hash_vals,
            ],
            device=self.device
        )

        # 3. Pass 2: 正式分配物理内存 ID，并记录新坐标
        wp.launch(
            kernel=pass2_allocate_blocks,
            dim=HASH_SIZE, # 遍历整个哈希表
            inputs=[self.hash_keys, self.hash_vals, self.block_counter, self.block_coords, self.max_blocks],
            device=self.device
        )

        # 4. Pass 3: 为所有新分配的块建立邻居通讯录
        # 注意：这里必须先获取刚刚分配了多少个新块
        new_active_count = self.get_active_blocks_count()

        wp.launch(
            kernel=pass3_build_neighbor_cache,
            dim=new_active_count, # 线程数等于当前帧新激活的 Block 数量
            inputs=[self.block_coords, self.hash_keys, self.hash_vals, self.neighbor_cache, self.block_counter],
            device=self.device
        )
    
    # MODIFIED: boundary-driven topology now reads old block activity through
    # the old hash tables so topology requests stay aligned with the old pool.
    def update_topology_boundary_driven(
            self,
            old_block_coords: wp.array,
            old_active_count: int,
            density: wp.array,
            vel_u: wp.array,
            vel_v: wp.array,
            vel_w: wp.array,
            density_threshold: float = 1.0e-3,
            velocity_threshold: float = 5.0e-2,
    ):
        """Selective topology update driven by boundary-face activity.

        This keeps the original active block and only allocates a face-neighbor
        block when the corresponding boundary voxels already contain noticeable
        density or velocity.
        """

        if old_active_count == 0:
            return

        wp.copy(self.old_hash_keys, self.hash_keys)
        wp.copy(self.old_hash_vals, self.hash_vals)
        self.hash_keys.fill_(EMPTY_KEY)
        self.hash_vals.fill_(UNALLOCATED)
        self.block_counter.fill_(0)

        # wp.launch(
        #     kernel=pass1_request_smart_halo_blocks,
        #     dim=old_active_count,
        #     inputs=[
        #         old_block_coords,
        #         old_active_count,
        #         self.domain_blocks[0],
        #         self.domain_blocks[1],
        #         self.domain_blocks[2],
        #         self.old_hash_keys,
        #         self.old_hash_vals,
        #         density,
        #         vel_u,
        #         vel_v,
        #         vel_w,
        #         density_threshold,
        #         velocity_threshold,
        #         self.hash_keys,
        #         self.hash_vals,
        #     ],
        #     device=self.device
        # )
        wp.launch(
            kernel=pass1_request_boundary_halo_blocks,
            dim=old_active_count,
            inputs=[
                old_block_coords,
                old_active_count,
                self.domain_blocks[0],
                self.domain_blocks[1],
                self.domain_blocks[2],
                self.old_hash_keys,
                self.old_hash_vals,
                density,
                vel_u,
                vel_v,
                vel_w,
                density_threshold,
                velocity_threshold,
                self.hash_keys,
                self.hash_vals,
            ],
            device=self.device
        )

        wp.launch(
            kernel=pass2_allocate_blocks,
            dim=HASH_SIZE,
            inputs=[self.hash_keys, self.hash_vals, self.block_counter, self.block_coords, self.max_blocks],
            device=self.device
        )

        new_active_count = self.get_active_blocks_count()

        wp.launch(
            kernel=pass3_build_neighbor_cache,
            dim=new_active_count,
            inputs=[self.block_coords, self.hash_keys, self.hash_vals, self.neighbor_cache, self.block_counter],
            device=self.device
        )

    def get_active_blocks_count(self):
        return int(self.block_counter.numpy()[0])


@wp.kernel
def pass1_request_smart_halo_blocks(
        old_block_coords: wp.array(dtype=wp.vec3i),
        old_active_count: int,
        max_bx: int,
        max_by: int,
        max_bz: int,
        old_hash_keys: wp.array(dtype=int),
        old_hash_vals: wp.array(dtype=int),
        density: wp.array(dtype=float),
        vel_u: wp.array(dtype=float),
        vel_v: wp.array(dtype=float),
        vel_w: wp.array(dtype=float),
        density_threshold: float,
        velocity_threshold: float,
        hash_keys: wp.array(dtype=int),
        hash_vals: wp.array(dtype=int)
):
    block_idx = wp.tid()
    if block_idx >= old_active_count:
        return

    coord = old_block_coords[block_idx]
    bx, by, bz = coord[0], coord[1], coord[2]

    # 1. 始终保留自己
    request_single_block(bx, by, bz, max_bx, max_by, max_bz, hash_keys, hash_vals)

    block_id = hash_lookup(bx, by, bz, old_hash_keys, old_hash_vals)
    if block_id < 0:
        return

    # 2. 扫描该 Block 内部的所有体素，建立“厚度为 2”的探测预警区
    for voxel_idx in range(BLOCK_VOL):
        lx = voxel_idx // (BLOCK_SIZE * BLOCK_SIZE)
        rem = voxel_idx % (BLOCK_SIZE * BLOCK_SIZE)
        ly = rem // BLOCK_SIZE
        lz = rem % BLOCK_SIZE

        margin = 1  # 只要流体逼近到 Block 边界 2 格以内，就触发膨胀

        dx_min = 0;
        dx_max = 0
        dy_min = 0;
        dy_max = 0
        dz_min = 0;
        dz_max = 0

        is_boundary = False
        if lx <= margin:
            dx_min = -1;
            is_boundary = True
        elif lx >= BLOCK_SIZE - 1 - margin:
            dx_max = 1;
            is_boundary = True

        if ly <= margin:
            dy_min = -1;
            is_boundary = True
        elif ly >= BLOCK_SIZE - 1 - margin:
            dy_max = 1;
            is_boundary = True

        if lz <= margin:
            dz_min = -1;
            is_boundary = True
        elif lz >= BLOCK_SIZE - 1 - margin:
            dz_max = 1;
            is_boundary = True

        if is_boundary:
            idx = block_pool_index(block_id, lx, ly, lz)

            # 检查该预警体素是否真的有流体（无视极其微弱的压强噪声）
            is_sig = False
            if density[idx] > density_threshold:
                is_sig = True
            elif wp.abs(vel_u[idx]) > velocity_threshold:
                is_sig = True
            elif wp.abs(vel_v[idx]) > velocity_threshold:
                is_sig = True
            elif wp.abs(vel_w[idx]) > velocity_threshold:
                is_sig = True

            if is_sig:
                # 核心修复：将该体素触碰到的所有方向（自动包含对面、边对角线、角对角线）一并请求！
                # 这完美覆盖了 26 个可能的空间邻居方向，且绝不盲目超发！
                for ox in range(dx_min, dx_max + 1):
                    for oy in range(dy_min, dy_max + 1):
                        for oz in range(dz_min, dz_max + 1):
                            if ox != 0 or oy != 0 or oz != 0:
                                request_single_block(bx + ox, by + oy, bz + oz, max_bx, max_by, max_bz, hash_keys,
                                                     hash_vals)