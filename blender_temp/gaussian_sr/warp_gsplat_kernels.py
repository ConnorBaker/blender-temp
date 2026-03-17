from .warp_runtime import wp


@wp.func
def _safe_rsqrt(x: float) -> float:
    return 1.0 / wp.sqrt(wp.max(x, 1.0e-20))


@wp.func
def _quat_normalize(w: float, x: float, y: float, z: float) -> wp.vec4:
    n2 = w * w + x * x + y * y + z * z
    inv = _safe_rsqrt(n2)
    return wp.vec4(w * inv, x * inv, y * inv, z * inv)


@wp.func
def _quat_to_rot_cols(wxyz: wp.vec4) -> tuple[wp.vec3, wp.vec3, wp.vec3]:
    w = wxyz[0]
    x = wxyz[1]
    y = wxyz[2]
    z = wxyz[3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)

    c0 = wp.vec3(r00, r10, r20)
    c1 = wp.vec3(r01, r11, r21)
    c2 = wp.vec3(r02, r12, r22)
    return c0, c1, c2


@wp.func
def _mat3_mul_vec3(
    m00: float,
    m01: float,
    m02: float,
    m10: float,
    m11: float,
    m12: float,
    m20: float,
    m21: float,
    m22: float,
    v: wp.vec3,
) -> wp.vec3:
    return wp.vec3(
        m00 * v[0] + m01 * v[1] + m02 * v[2],
        m10 * v[0] + m11 * v[1] + m12 * v[2],
        m20 * v[0] + m21 * v[1] + m22 * v[2],
    )


@wp.func
def _outer_sym_entries(v: wp.vec3) -> tuple[float, float, float, float, float, float]:
    x = v[0]
    y = v[1]
    z = v[2]
    return (x * x, x * y, x * z, y * y, y * z, z * z)


@wp.func
def _cov2d_from_cov3d_jacobian(
    c00: float,
    c01: float,
    c02: float,
    c11: float,
    c12: float,
    c22: float,
    fx: float,
    fy: float,
    X: float,
    Y: float,
    Z: float,
) -> tuple[float, float, float]:
    invZ = 1.0 / Z
    invZ2 = invZ * invZ
    j00 = fx * invZ
    j02 = -fx * X * invZ2
    j11 = fy * invZ
    j12 = -fy * Y * invZ2
    s00 = j00 * j00 * c00 + 2.0 * j00 * j02 * c02 + j02 * j02 * c22
    s11 = j11 * j11 * c11 + 2.0 * j11 * j12 * c12 + j12 * j12 * c22
    s01 = j00 * (c01 * j11 + c02 * j12) + j02 * (c12 * j11 + c22 * j12)
    return s00, s01, s11


@wp.func
def _inv_sym2(a: float, b: float, c: float) -> tuple[float, float, float, float]:
    det = a * c - b * b
    det_safe = wp.max(det, 1.0e-20)
    inv_det = 1.0 / det_safe
    A = c * inv_det
    B = -b * inv_det
    C = a * inv_det
    return A, B, C, det_safe


@wp.func
def _lambda_max_sym2(a: float, b: float, c: float) -> float:
    tr = a + c
    disc = wp.sqrt(wp.max((a - c) * (a - c) + 4.0 * b * b, 0.0))
    return 0.5 * (tr + disc)


@wp.func
def _write_culled_projection(
    i: int,
    u: float,
    v: float,
    A: float,
    B: float,
    C: float,
    rho_i: float,
    radius_i: int,
    depth_i: int,
    xys: wp.array(dtype=wp.vec2),
    conic: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=wp.float32),
    radius: wp.array(dtype=wp.int32),
    num_tiles_hit: wp.array(dtype=wp.int32),
    tile_min: wp.array(dtype=wp.vec2i),
    tile_max: wp.array(dtype=wp.vec2i),
    depth_key: wp.array(dtype=wp.int32),
):
    xys[i] = wp.vec2(u, v)
    conic[i] = wp.vec3(A, B, C)
    rho[i] = rho_i
    radius[i] = radius_i
    num_tiles_hit[i] = 0
    tile_min[i] = wp.vec2i(0, 0)
    tile_max[i] = wp.vec2i(-1, -1)
    depth_key[i] = depth_i


@wp.kernel
def project_gaussians_kernel(
    means: wp.array(dtype=wp.vec3),
    quat: wp.array(dtype=wp.vec4),
    scale: wp.array(dtype=wp.vec3),
    viewmat: wp.array2d(dtype=wp.float32),
    K: wp.array2d(dtype=wp.float32),
    width: int,
    height: int,
    tile_size: int,
    tiles_x: int,
    tiles_y: int,
    near_plane: float,
    far_plane: float,
    eps2d: float,
    radius_clip: float,
    depth_scale: float,
    xys: wp.array(dtype=wp.vec2),
    conic: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=wp.float32),
    radius: wp.array(dtype=wp.int32),
    num_tiles_hit: wp.array(dtype=wp.int32),
    tile_min: wp.array(dtype=wp.vec2i),
    tile_max: wp.array(dtype=wp.vec2i),
    depth_key: wp.array(dtype=wp.int32),
):
    i = wp.tid()

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    r00 = float(viewmat[0, 0])
    r01 = float(viewmat[0, 1])
    r02 = float(viewmat[0, 2])
    t0 = float(viewmat[0, 3])
    r10 = float(viewmat[1, 0])
    r11 = float(viewmat[1, 1])
    r12 = float(viewmat[1, 2])
    t1 = float(viewmat[1, 3])
    r20 = float(viewmat[2, 0])
    r21 = float(viewmat[2, 1])
    r22 = float(viewmat[2, 2])
    t2 = float(viewmat[2, 3])

    mean = means[i]
    Xw = mean[0]
    Yw = mean[1]
    Zw = mean[2]
    X = r00 * Xw + r01 * Yw + r02 * Zw + t0
    Y = r10 * Xw + r11 * Yw + r12 * Zw + t1
    Z = r20 * Xw + r21 * Yw + r22 * Zw + t2

    if (Z <= near_plane) or (Z >= far_plane):
        _write_culled_projection(
            i, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0, xys, conic, rho, radius, num_tiles_hit, tile_min, tile_max, depth_key
        )
        return

    invZ = 1.0 / Z
    u = fx * X * invZ + cx
    v = fy * Y * invZ + cy

    q = quat[i]
    qn = _quat_normalize(q[0], q[1], q[2], q[3])
    c0, c1, c2 = _quat_to_rot_cols(qn)
    t0v = _mat3_mul_vec3(r00, r01, r02, r10, r11, r12, r20, r21, r22, c0)
    t1v = _mat3_mul_vec3(r00, r01, r02, r10, r11, r12, r20, r21, r22, c1)
    t2v = _mat3_mul_vec3(r00, r01, r02, r10, r11, r12, r20, r21, r22, c2)

    s = scale[i]
    sx = wp.max(s[0], 1.0e-8)
    sy = wp.max(s[1], 1.0e-8)
    sz = wp.max(s[2], 1.0e-8)
    s0 = sx * sx
    s1 = sy * sy
    s2 = sz * sz

    o00, o01, o02, o11, o12, o22 = _outer_sym_entries(t0v)
    p00, p01, p02, p11, p12, p22 = _outer_sym_entries(t1v)
    q00, q01, q02, q11, q12, q22 = _outer_sym_entries(t2v)

    c00 = s0 * o00 + s1 * p00 + s2 * q00
    c01 = s0 * o01 + s1 * p01 + s2 * q01
    c02 = s0 * o02 + s1 * p02 + s2 * q02
    c11 = s0 * o11 + s1 * p11 + s2 * q11
    c12 = s0 * o12 + s1 * p12 + s2 * q12
    c22 = s0 * o22 + s1 * p22 + s2 * q22

    s00, s01, s11 = _cov2d_from_cov3d_jacobian(c00, c01, c02, c11, c12, c22, fx, fy, X, Y, Z)
    det_noeps = wp.max(s00 * s11 - s01 * s01, 1.0e-20)
    s00e = s00 + eps2d
    s11e = s11 + eps2d
    s01e = s01
    A, B, C, det_eps = _inv_sym2(s00e, s01e, s11e)
    rho_i = wp.sqrt(det_noeps / det_eps)

    lam = _lambda_max_sym2(s00e, s01e, s11e)
    rad_f = 3.0 * wp.sqrt(wp.max(lam, 0.0))
    rad_i = int(wp.ceil(rad_f))

    if radius_clip > 0.0 and float(rad_i) >= radius_clip:
        _write_culled_projection(
            i,
            u,
            v,
            A,
            B,
            C,
            rho_i,
            0,
            int(wp.clamp(Z * depth_scale, 0.0, 2147483647.0)),
            xys,
            conic,
            rho,
            radius,
            num_tiles_hit,
            tile_min,
            tile_max,
            depth_key,
        )
        return

    if (
        (u + float(rad_i) < 0.0)
        or (u - float(rad_i) >= float(width))
        or (v + float(rad_i) < 0.0)
        or (v - float(rad_i) >= float(height))
    ):
        _write_culled_projection(
            i,
            u,
            v,
            A,
            B,
            C,
            rho_i,
            0,
            int(wp.clamp(Z * depth_scale, 0.0, 2147483647.0)),
            xys,
            conic,
            rho,
            radius,
            num_tiles_hit,
            tile_min,
            tile_max,
            depth_key,
        )
        return

    x0 = int(wp.floor((u - float(rad_i)) / float(tile_size)))
    x1 = int(wp.floor((u + float(rad_i)) / float(tile_size)))
    y0 = int(wp.floor((v - float(rad_i)) / float(tile_size)))
    y1 = int(wp.floor((v + float(rad_i)) / float(tile_size)))
    x0 = wp.clamp(x0, 0, tiles_x - 1)
    x1 = wp.clamp(x1, 0, tiles_x - 1)
    y0 = wp.clamp(y0, 0, tiles_y - 1)
    y1 = wp.clamp(y1, 0, tiles_y - 1)

    xys[i] = wp.vec2(u, v)
    conic[i] = wp.vec3(A, B, C)
    rho[i] = rho_i
    radius[i] = rad_i
    tile_min[i] = wp.vec2i(x0, y0)
    tile_max[i] = wp.vec2i(x1, y1)
    num_tiles_hit[i] = (x1 - x0 + 1) * (y1 - y0 + 1)
    depth_key[i] = int(wp.clamp(Z * depth_scale, 0.0, 2147483647.0))


def specialize_project_kernel(
    *,
    tile_size: int,
    near_plane: float,
    far_plane: float,
    eps2d: float,
    radius_clip: float,
    depth_scale: float,
):
    TILE_SIZE = wp.constant(int(tile_size))
    NEAR_PLANE = wp.constant(float(near_plane))
    FAR_PLANE = wp.constant(float(far_plane))
    EPS2D = wp.constant(float(eps2d))
    RADIUS_CLIP = wp.constant(float(radius_clip))
    DEPTH_SCALE = wp.constant(float(depth_scale))

    @wp.kernel(module="unique")
    def project_gaussians_kernel_specialized(
        means: wp.array(dtype=wp.vec3),
        quat: wp.array(dtype=wp.vec4),
        scale: wp.array(dtype=wp.vec3),
        viewmat: wp.array2d(dtype=wp.float32),
        K: wp.array2d(dtype=wp.float32),
        width: int,
        height: int,
        tiles_x: int,
        tiles_y: int,
        xys: wp.array(dtype=wp.vec2),
        conic: wp.array(dtype=wp.vec3),
        rho: wp.array(dtype=wp.float32),
        radius: wp.array(dtype=wp.int32),
        num_tiles_hit: wp.array(dtype=wp.int32),
        tile_min: wp.array(dtype=wp.vec2i),
        tile_max: wp.array(dtype=wp.vec2i),
        depth_key: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        r00 = float(viewmat[0, 0])
        r01 = float(viewmat[0, 1])
        r02 = float(viewmat[0, 2])
        t0 = float(viewmat[0, 3])
        r10 = float(viewmat[1, 0])
        r11 = float(viewmat[1, 1])
        r12 = float(viewmat[1, 2])
        t1 = float(viewmat[1, 3])
        r20 = float(viewmat[2, 0])
        r21 = float(viewmat[2, 1])
        r22 = float(viewmat[2, 2])
        t2 = float(viewmat[2, 3])

        mean = means[i]
        Xw = mean[0]
        Yw = mean[1]
        Zw = mean[2]
        X = r00 * Xw + r01 * Yw + r02 * Zw + t0
        Y = r10 * Xw + r11 * Yw + r12 * Zw + t1
        Z = r20 * Xw + r21 * Yw + r22 * Zw + t2

        if (Z <= NEAR_PLANE) or (Z >= FAR_PLANE):
            _write_culled_projection(
                i,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0,
                0,
                xys,
                conic,
                rho,
                radius,
                num_tiles_hit,
                tile_min,
                tile_max,
                depth_key,
            )
            return

        invZ = 1.0 / Z
        u = fx * X * invZ + cx
        v = fy * Y * invZ + cy

        q = quat[i]
        qn = _quat_normalize(q[0], q[1], q[2], q[3])
        c0, c1, c2 = _quat_to_rot_cols(qn)
        t0v = _mat3_mul_vec3(r00, r01, r02, r10, r11, r12, r20, r21, r22, c0)
        t1v = _mat3_mul_vec3(r00, r01, r02, r10, r11, r12, r20, r21, r22, c1)
        t2v = _mat3_mul_vec3(r00, r01, r02, r10, r11, r12, r20, r21, r22, c2)

        s = scale[i]
        sx = wp.max(s[0], 1.0e-8)
        sy = wp.max(s[1], 1.0e-8)
        sz = wp.max(s[2], 1.0e-8)
        s0 = sx * sx
        s1 = sy * sy
        s2 = sz * sz

        o00, o01, o02, o11, o12, o22 = _outer_sym_entries(t0v)
        p00, p01, p02, p11, p12, p22 = _outer_sym_entries(t1v)
        q00, q01, q02, q11, q12, q22 = _outer_sym_entries(t2v)

        c00 = s0 * o00 + s1 * p00 + s2 * q00
        c01 = s0 * o01 + s1 * p01 + s2 * q01
        c02 = s0 * o02 + s1 * p02 + s2 * q02
        c11 = s0 * o11 + s1 * p11 + s2 * q11
        c12 = s0 * o12 + s1 * p12 + s2 * q12
        c22 = s0 * o22 + s1 * p22 + s2 * q22

        s00, s01, s11 = _cov2d_from_cov3d_jacobian(c00, c01, c02, c11, c12, c22, fx, fy, X, Y, Z)
        det_noeps = wp.max(s00 * s11 - s01 * s01, 1.0e-20)
        s00e = s00 + EPS2D
        s11e = s11 + EPS2D
        s01e = s01
        A, B, C, det_eps = _inv_sym2(s00e, s01e, s11e)
        rho_i = wp.sqrt(det_noeps / det_eps)

        lam = _lambda_max_sym2(s00e, s01e, s11e)
        rad_f = 3.0 * wp.sqrt(wp.max(lam, 0.0))
        rad_i = int(wp.ceil(rad_f))
        depth_i = int(wp.clamp(Z * DEPTH_SCALE, 0.0, 2147483647.0))

        if RADIUS_CLIP > 0.0 and float(rad_i) >= RADIUS_CLIP:
            _write_culled_projection(
                i,
                u,
                v,
                A,
                B,
                C,
                rho_i,
                0,
                depth_i,
                xys,
                conic,
                rho,
                radius,
                num_tiles_hit,
                tile_min,
                tile_max,
                depth_key,
            )
            return

        if (
            (u + float(rad_i) < 0.0)
            or (u - float(rad_i) >= float(width))
            or (v + float(rad_i) < 0.0)
            or (v - float(rad_i) >= float(height))
        ):
            _write_culled_projection(
                i,
                u,
                v,
                A,
                B,
                C,
                rho_i,
                0,
                depth_i,
                xys,
                conic,
                rho,
                radius,
                num_tiles_hit,
                tile_min,
                tile_max,
                depth_key,
            )
            return

        x0 = int(wp.floor((u - float(rad_i)) / float(TILE_SIZE)))
        x1 = int(wp.floor((u + float(rad_i)) / float(TILE_SIZE)))
        y0 = int(wp.floor((v - float(rad_i)) / float(TILE_SIZE)))
        y1 = int(wp.floor((v + float(rad_i)) / float(TILE_SIZE)))
        x0 = wp.clamp(x0, 0, tiles_x - 1)
        x1 = wp.clamp(x1, 0, tiles_x - 1)
        y0 = wp.clamp(y0, 0, tiles_y - 1)
        y1 = wp.clamp(y1, 0, tiles_y - 1)

        xys[i] = wp.vec2(u, v)
        conic[i] = wp.vec3(A, B, C)
        rho[i] = rho_i
        radius[i] = rad_i
        tile_min[i] = wp.vec2i(x0, y0)
        tile_max[i] = wp.vec2i(x1, y1)
        num_tiles_hit[i] = (x1 - x0 + 1) * (y1 - y0 + 1)
        depth_key[i] = depth_i

    return project_gaussians_kernel_specialized


@wp.kernel
def map_to_intersects_kernel(
    num_tiles_hit: wp.array(dtype=wp.int32),
    cum_tiles_hit: wp.array(dtype=wp.int32),
    tile_min: wp.array(dtype=wp.vec2i),
    tile_max: wp.array(dtype=wp.vec2i),
    depth_key: wp.array(dtype=wp.int32),
    tiles_x: int,
    out_keys: wp.array(dtype=wp.int64),
    out_vals: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    cnt = num_tiles_hit[i]
    if cnt <= 0:
        return
    start = cum_tiles_hit[i] - cnt
    tile_min_i = tile_min[i]
    tile_max_i = tile_max[i]
    xmin = tile_min_i[0]
    xmax = tile_max_i[0]
    ymin = tile_min_i[1]
    ymax = tile_max_i[1]
    dk = depth_key[i]

    local = int(0)
    for ty in range(ymin, ymax + 1):
        base = ty * tiles_x
        for tx in range(xmin, xmax + 1):
            tile_id = base + tx
            idx = start + local
            key_hi = wp.int64(tile_id) << wp.int64(32)
            key_lo = wp.int64(dk) & wp.int64(0xFFFFFFFF)
            out_keys[idx] = key_hi | key_lo
            out_vals[idx] = wp.int32(i)
            local += 1


@wp.kernel
def init_int32_kernel(arr: wp.array(dtype=wp.int32), value: int):
    i = wp.tid()
    arr[i] = wp.int32(value)


def specialize_raster_kernels(
    channels: int,
    *,
    tile_size: int,
    antialiased: bool,
    background_is_zero: bool,
):
    VecC = wp.types.vector(length=channels, dtype=wp.float32)
    TILE_SIZE = wp.constant(int(tile_size))

    @wp.kernel(module="unique")
    def rasterize_values_kernel(
        tile_start: wp.array(dtype=wp.int32),
        tile_end: wp.array(dtype=wp.int32),
        sorted_vals: wp.array(dtype=wp.int32),
        xys: wp.array(dtype=wp.vec2),
        conic: wp.array(dtype=wp.vec3),
        rho: wp.array(dtype=wp.float32),
        values: wp.array(dtype=VecC),
        opacity: wp.array(dtype=wp.float32),
        width: int,
        height: int,
        tiles_x: int,
        alpha_min: float,
        trans_eps: float,
        clamp_alpha_max: float,
        background: wp.array(dtype=wp.float32),
        out_values: wp.array(dtype=VecC),
        out_final_T: wp.array(dtype=wp.float32),
        out_stop_idx: wp.array(dtype=wp.int32),
    ):
        p = wp.tid()
        px = p - (p // width) * width
        py = p // width
        tile_x = px // TILE_SIZE
        tile_y = py // TILE_SIZE
        tile_id = tile_y * tiles_x + tile_x
        start = tile_start[tile_id]
        end = tile_end[tile_id]
        fx = float(px) + 0.5
        fy = float(py) + 0.5

        bg = VecC(0.0)
        if not wp.static(background_is_zero):
            for c in range(wp.static(channels)):
                bg[c] = background[c]

        T = float(1.0)
        stop = start
        accum = VecC(0.0)
        for idx in range(start, end):
            gid = sorted_vals[idx]
            xy = xys[gid]
            dx = fx - xy[0]
            dy = fy - xy[1]
            abc = conic[gid]
            sigma = 0.5 * (abc[0] * dx * dx + abc[2] * dy * dy) + abc[1] * dx * dy
            w = wp.exp(-sigma)
            a = opacity[gid] * w
            if wp.static(antialiased):
                a *= rho[gid]
            if a > clamp_alpha_max:
                a = clamp_alpha_max
            if a < alpha_min:
                continue
            w_i = T * a
            accum = accum + w_i * values[gid]
            T *= 1.0 - a
            stop = idx + 1
            if T < trans_eps:
                break

        if not wp.static(background_is_zero):
            accum = accum + T * bg

        out_values[p] = accum
        out_final_T[p] = T
        out_stop_idx[p] = stop

    @wp.kernel(module="unique")
    def rasterize_values_backward_kernel(
        tile_start: wp.array(dtype=wp.int32),
        tile_end: wp.array(dtype=wp.int32),
        sorted_vals: wp.array(dtype=wp.int32),
        xys: wp.array(dtype=wp.vec2),
        conic: wp.array(dtype=wp.vec3),
        rho: wp.array(dtype=wp.float32),
        values: wp.array(dtype=VecC),
        opacity: wp.array(dtype=wp.float32),
        width: int,
        height: int,
        tiles_x: int,
        alpha_min: float,
        trans_eps: float,
        clamp_alpha_max: float,
        background: wp.array(dtype=wp.float32),
        final_T: wp.array(dtype=wp.float32),
        stop_idx: wp.array(dtype=wp.int32),
        grad_out: wp.array(dtype=VecC),
        grad_xys: wp.array(dtype=wp.vec2),
        grad_conic: wp.array(dtype=wp.vec3),
        grad_rho: wp.array(dtype=wp.float32),
        grad_values: wp.array(dtype=VecC),
        grad_opacity: wp.array(dtype=wp.float32),
    ):
        p = wp.tid()
        px = p - (p // width) * width
        py = p // width
        tile_x = px // TILE_SIZE
        tile_y = py // TILE_SIZE
        p = py * width + px
        grad_out_p = grad_out[p]
        tile_id = tile_y * tiles_x + tile_x
        start = tile_start[tile_id]
        fx = float(px) + 0.5
        fy = float(py) + 0.5

        bg = VecC(0.0)
        if not wp.static(background_is_zero):
            for c in range(wp.static(channels)):
                bg[c] = background[c]

        stop = stop_idx[p]
        final_T_p = final_T[p]
        suffix_dot = float(0.0)
        if not wp.static(background_is_zero):
            suffix_dot = wp.dot(grad_out_p, bg)

        suffix_trans = float(1.0)
        for rev in range(stop - start):
            idx = stop - 1 - rev
            gid = sorted_vals[idx]
            xy = xys[gid]
            dx = fx - xy[0]
            dy = fy - xy[1]
            abc = conic[gid]
            weight = wp.exp(-(0.5 * (abc[0] * dx * dx + abc[2] * dy * dy) + abc[1] * dx * dy))
            alpha_base = opacity[gid] * weight
            alpha_unclamped = alpha_base
            if wp.static(antialiased):
                alpha_unclamped *= rho[gid]
            alpha = alpha_unclamped
            clamped = 0
            if alpha > clamp_alpha_max:
                alpha = clamp_alpha_max
                clamped = 1
            if alpha < alpha_min:
                continue

            denom = (1.0 - alpha) * suffix_trans
            if denom < 1.0e-12:
                denom = 1.0e-12
            T_i = final_T_p / denom
            weight_i = T_i * alpha
            value_vec = values[gid]
            wp.atomic_add(grad_values, gid, grad_out_p * weight_i)

            dot_grad_value = wp.dot(grad_out_p, value_vec)
            dL_dalpha = T_i * (dot_grad_value - suffix_dot)
            if clamped == 0:
                dL_dalpha_base = dL_dalpha
                if wp.static(antialiased):
                    wp.atomic_add(grad_rho, gid, dL_dalpha * alpha_base)
                    dL_dalpha_base *= rho[gid]

                wp.atomic_add(grad_opacity, gid, dL_dalpha_base * weight)

                dL_dsigma = -dL_dalpha_base * alpha_base
                dL_dx = -dL_dsigma * (abc[0] * dx + abc[1] * dy)
                dL_dy = -dL_dsigma * (abc[2] * dy + abc[1] * dx)
                dL_dA = dL_dsigma * 0.5 * dx * dx
                dL_dB = dL_dsigma * dx * dy
                dL_dC = dL_dsigma * 0.5 * dy * dy

                wp.atomic_add(grad_xys, gid, wp.vec2(dL_dx, dL_dy))
                wp.atomic_add(grad_conic, gid, wp.vec3(dL_dA, dL_dB, dL_dC))

            suffix_dot = alpha * dot_grad_value + (1.0 - alpha) * suffix_dot
            suffix_trans *= 1.0 - alpha

    return rasterize_values_kernel, rasterize_values_backward_kernel, VecC


@wp.kernel
def rasterize_visibility_stats_kernel(
    tile_start: wp.array(dtype=wp.int32),
    tile_end: wp.array(dtype=wp.int32),
    sorted_vals: wp.array(dtype=wp.int32),
    xys: wp.array(dtype=wp.vec2),
    conic: wp.array(dtype=wp.vec3),
    rho: wp.array(dtype=wp.float32),
    opacity: wp.array(dtype=wp.float32),
    width: int,
    height: int,
    tiles_x: int,
    tile_size: int,
    residual_map: wp.array(dtype=wp.float32),
    error_bins_x: int,
    error_bins_y: int,
    alpha_min: float,
    trans_eps: float,
    clamp_alpha_max: float,
    antialiased: int,
    out_contrib: wp.array(dtype=wp.float32),
    out_trans: wp.array(dtype=wp.float32),
    out_hits: wp.array(dtype=wp.float32),
    out_residual: wp.array(dtype=wp.float32),
    out_error_map: wp.array(dtype=wp.float32),
):
    p = wp.tid()
    px = p - (p // width) * width
    py = p // width
    tile_x = px // tile_size
    tile_y = py // tile_size
    tile_id = tile_y * tiles_x + tile_x
    start = tile_start[tile_id]
    end = tile_end[tile_id]
    fx = float(px) + 0.5
    fy = float(py) + 0.5

    T = float(1.0)
    if start < end:
        for idx in range(start, end):
            gid = sorted_vals[idx]
            xy = xys[gid]
            conic_abc = conic[gid]
            dx = fx - xy[0]
            dy = fy - xy[1]
            A = conic_abc[0]
            B = conic_abc[1]
            C = conic_abc[2]
            sigma = 0.5 * (A * dx * dx + C * dy * dy) + B * dx * dy
            w = wp.exp(-sigma)
            a = opacity[gid] * w
            if antialiased != 0:
                a *= rho[gid]
            if a > clamp_alpha_max:
                a = clamp_alpha_max
            if a < alpha_min:
                continue
            w_i = T * a
            residual = residual_map[p]
            wp.atomic_add(out_contrib, gid, w_i)
            wp.atomic_add(out_trans, gid, T)
            wp.atomic_add(out_hits, gid, 1.0)
            wp.atomic_add(out_residual, gid, w_i * residual)
            bx = int(wp.clamp((float(px) / float(width)) * float(error_bins_x), 0.0, float(error_bins_x - 1)))
            by = int(wp.clamp((float(py) / float(height)) * float(error_bins_y), 0.0, float(error_bins_y - 1)))
            bin_idx = by * error_bins_x + bx
            wp.atomic_add(out_error_map, gid * (error_bins_x * error_bins_y) + bin_idx, w_i * residual)
            T *= 1.0 - a
            if T < trans_eps:
                break


def specialize_visibility_stats_kernel(
    *,
    antialiased: bool,
    error_bins_x: int,
    error_bins_y: int,
):
    ERROR_BINS_X = wp.constant(int(error_bins_x))
    ERROR_BINS_Y = wp.constant(int(error_bins_y))
    ERROR_BIN_COUNT = wp.constant(int(error_bins_x) * int(error_bins_y))

    @wp.kernel(module="unique")
    def rasterize_visibility_stats_kernel_specialized(
        tile_start: wp.array(dtype=wp.int32),
        tile_end: wp.array(dtype=wp.int32),
        sorted_vals: wp.array(dtype=wp.int32),
        xys: wp.array(dtype=wp.vec2),
        conic: wp.array(dtype=wp.vec3),
        rho: wp.array(dtype=wp.float32),
        opacity: wp.array(dtype=wp.float32),
        width: int,
        height: int,
        tiles_x: int,
        tile_size: int,
        residual_map: wp.array(dtype=wp.float32),
        alpha_min: float,
        trans_eps: float,
        clamp_alpha_max: float,
        out_contrib: wp.array(dtype=wp.float32),
        out_trans: wp.array(dtype=wp.float32),
        out_hits: wp.array(dtype=wp.float32),
        out_residual: wp.array(dtype=wp.float32),
        out_error_map: wp.array(dtype=wp.float32),
    ):
        p = wp.tid()
        px = p - (p // width) * width
        py = p // width
        tile_x = px // tile_size
        tile_y = py // tile_size
        tile_id = tile_y * tiles_x + tile_x
        start = tile_start[tile_id]
        end = tile_end[tile_id]
        fx = float(px) + 0.5
        fy = float(py) + 0.5

        T = float(1.0)
        if start < end:
            for idx in range(start, end):
                gid = sorted_vals[idx]
                xy = xys[gid]
                conic_abc = conic[gid]
                dx = fx - xy[0]
                dy = fy - xy[1]
                A = conic_abc[0]
                B = conic_abc[1]
                C = conic_abc[2]
                sigma = 0.5 * (A * dx * dx + C * dy * dy) + B * dx * dy
                w = wp.exp(-sigma)
                a = opacity[gid] * w
                if wp.static(antialiased):
                    a *= rho[gid]
                if a > clamp_alpha_max:
                    a = clamp_alpha_max
                if a < alpha_min:
                    continue
                w_i = T * a
                residual = residual_map[p]
                wp.atomic_add(out_contrib, gid, w_i)
                wp.atomic_add(out_trans, gid, T)
                wp.atomic_add(out_hits, gid, 1.0)
                wp.atomic_add(out_residual, gid, w_i * residual)
                bx = int(wp.clamp((float(px) / float(width)) * float(ERROR_BINS_X), 0.0, float(ERROR_BINS_X - 1)))
                by = int(wp.clamp((float(py) / float(height)) * float(ERROR_BINS_Y), 0.0, float(ERROR_BINS_Y - 1)))
                bin_idx = by * ERROR_BINS_X + bx
                wp.atomic_add(out_error_map, gid * ERROR_BIN_COUNT + bin_idx, w_i * residual)
                T *= 1.0 - a
                if T < trans_eps:
                    break

    return rasterize_visibility_stats_kernel_specialized


@wp.kernel
def get_tile_bin_edges_kernel(
    sorted_keys: wp.array(dtype=wp.int64),
    M: int,
    tile_start: wp.array(dtype=wp.int32),
    tile_end: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    if i >= M:
        return
    tile = wp.int32(sorted_keys[i] >> wp.int64(32))
    if i == 0:
        tile_start[tile] = 0
    else:
        prev_tile = wp.int32(sorted_keys[i - 1] >> wp.int64(32))
        if prev_tile != tile:
            tile_end[prev_tile] = i
            tile_start[tile] = i
    if i == M - 1:
        tile_end[tile] = M


__all__ = [
    "project_gaussians_kernel",
    "specialize_project_kernel",
    "map_to_intersects_kernel",
    "init_int32_kernel",
    "get_tile_bin_edges_kernel",
    "specialize_raster_kernels",
    "rasterize_visibility_stats_kernel",
    "specialize_visibility_stats_kernel",
]
