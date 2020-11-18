import math, noisy/simd/sse2, random

const
  F2 = (0.5 * (sqrt(3.0) - 1)).float32
  G2 = ((3 - sqrt(3.0)) / 6).float32
  F3 = (1.0 / 3.0).float32
  G3 = (1.0 / 6.0).float32

  grad3 = [
    [1.float32, 1, 0], [-1.float32, 1, 0], [1.float32, -1, 0],
    [-1.float32, -1, 0], [1.float32, 0, 1], [-1.float32, 0, 1],
    [1.float32, 0, -1], [-1.float32, 0, -1], [0.float32, 1, 1],
    [0.float32, -1, 1], [0.float32, 1, -1], [0.float32, -1, -1]
  ]

type
  Simplex* = object
    octaves*: int
    amplitude*, frequency*, lacunarity*, gain*: float32
    perm: array[512, int32]
    permMod12: array[512, int32]

  Grid* = ref object
    width*, height*, depth*: int
    values*: seq[float32]

  NoisyError* = object of ValueError

when defined(release):
  {.push checks: off.}

template failOctaves() =
  raise newException(NoisyError, "Octaves must be > 0")

func initSimplex*(seed: int): Simplex =
  result.octaves = 1
  result.amplitude = 1
  result.frequency = 1
  result.lacunarity = 2
  result.gain = 0.5

  for i in 0 ..< result.perm.len:
    result.perm[i] = i.int32 and 255
  var r = initRand(seed)
  shuffle(r, result.perm)
  for i in 0 ..< result.perm.len:
    result.permMod12[i] = result.perm[i] mod 12

template valueIndex(g: Grid, x, y, z: int): int =
  z + y * g.depth + x * g.height * g.depth

func `[]`*(g: Grid, x, y: int, z = 0): float32 =
  if x < 0 or x >= g.width:
    raise newException(IndexDefect, "Index x out of bounds")
  if y < 0 or y >= g.height:
    raise newException(IndexDefect, "Index y out of bounds")
  if z < 0 or z >= g.depth:
    raise newException(IndexDefect, "Index z out of bounds")
  g.values[g.valueIndex(x, y, z)]

func `[]=`(g: Grid, x, y, z: int, value: float32) =
  g.values[g.valueIndex(x, y, z)] = value

func dot(g: array[3, float32], x, y: float32): float32 {.inline.} =
  g[0] * x + g[1] * y

func dot(g: array[3, float32], x, y, z: float32): float32 {.inline.} =
  g[0] * x + g[1] * y + g[2] * z

func noise(simplex: Simplex, x, y: float32): float32 =
  let
    s = (x + y) * F2
    i = floor(x + s)
    j = floor(y + s)
    t = (i + j) * G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    gt = (x0 > y0).int32
    i1 = gt
    j1 = (not gt) and 1
    x1 = x0 - i1.float32 + G2
    y1 = y0 - j1.float32 + G2
    x2 = x0 - 1.float32 + 2.float32 * G2
    y2 = y0 - 1.float32 + 2.float32 * G2
    ii = i.int32 and 255
    jj = j.int32 and 255
    t0 = 0.5.float32 - x0 * x0 - y0 * y0
    t1 = 0.5.float32 - x1 * x1 - y1 * y1
    t2 = 0.5.float32 - x2 * x2 - y2 * y2

  var n0, n1, n2: float32
  if t0 > 0:
    n0 = t0 * t0 * t0 * t0 * dot(
      grad3[simplex.permMod12[ii + simplex.perm[jj]]], x0, y0
    )

  if t1 > 0:
    n1 = t1 * t1 * t1 * t1 * dot(
      grad3[simplex.permMod12[ii + i1 + simplex.perm[jj + j1]]], x1, y1
    )

  if t2 > 0:
    n2 = t2 * t2 * t2 * t2 * dot(
      grad3[simplex.permMod12[ii + 1 + simplex.perm[jj + 1]]], x2, y2
    )

  70.float32 * (n0 + n1 + n2)

func noise(simplex: Simplex, x, y, z: float32): float32 =
  let
    s = (x + y + z) * F3
    i = floor(x + s)
    j = floor(y + s)
    k = floor(z + s)
    t = (i + j + k) * G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)
    x0gey0 = (x0 >= y0).int32
    y0gez0 = (y0 >= z0).int32
    x0gez0 = (x0 >= z0).int32
    i1 = x0gey0 and x0gez0
    j1 = (not x0gey0) and y0gez0
    k1 = (not y0gez0) and 1
    i2 = x0gey0 or (x0gez0 and y0gez0)
    j2 = (x0gey0 and y0gez0) or ((not x0gey0) and 1)
    k2  = (x0gey0 and ((not y0gez0) and 1)) or (((not x0gey0) and 1) and ((not x0gez0) and 1))
    x1 = x0 - i1.float32 + G3
    y1 = y0 - j1.float32 + G3
    z1 = z0 - k1.float32 + G3
    x2 = x0 - i2.float32 + 2.float32 * G3
    y2 = y0 - j2.float32 + 2.float32 * G3
    z2 = z0 - k2.float32 + 2.float32 * G3
    x3 = x0 - 1.float32 + 3.float32 * G3
    y3 = y0 - 1.float32 + 3.float32 * G3
    z3 = z0 - 1.float32 + 3.float32 * G3
    ii = i.int32 and 255
    jj = j.int32 and 255
    kk = k.int32 and 255
    t0 = 0.6.float32 - x0 * x0 - y0 * y0 - z0 * z0
    t1 = 0.6.float32 - x1 * x1 - y1 * y1 - z1 * z1
    t2 = 0.6.float32 - x2 * x2 - y2 * y2 - z2 * z2
    t3 = 0.6.float32 - x3 * x3 - y3 * y3 - z3 * z3

  var n0, n1, n2, n3: float32
  if t0 >= 0:
    let gi0 = simplex.permMod12[
      ii + simplex.perm[jj + simplex.perm[kk]]
    ]
    n0 = t0 * t0 * t0 * t0 * dot(grad3[gi0], x0, y0, z0)

  if t1 >= 0:
    let gi1 = simplex.permMod12[
      ii + i1 + simplex.perm[jj + j1 + simplex.perm[kk + k1]]
    ]
    n1 = t1 * t1 * t1 * t1 * dot(grad3[gi1], x1, y1, z1)

  if t2 >= 0:
    let gi2 = simplex.permMod12[
      ii + i2 + simplex.perm[jj + j2 + simplex.perm[kk + k2]]
    ]
    n2 = t2 * t2 * t2 * t2 * dot(grad3[gi2], x2, y2, z2)

  if t3 >= 0:
    let gi3 = simplex.permMod12[
      ii + 1 + simplex.perm[jj + 1 + simplex.perm[kk + 1]]
    ]
    n3 = t3 * t3 * t3 * t3 * dot(grad3[gi3], x3, y3, z3)

  32.float32 * (n0 + n1 + n2 + n3)

func value*(simplex: Simplex, x, y: float32): float32 =
  ## Generates the 2D noise value at (x, y) based on the Simplex parameters.

  if simplex.octaves == 0:
    failOctaves()

  var
    total: float32
    amplitude = simplex.amplitude
    frequency = simplex.frequency

  for _ in 0 ..< simplex.octaves:
    total += simplex.noise(x * frequency, y * frequency) * amplitude
    amplitude *= simplex.gain
    frequency *= simplex.lacunarity

  total / simplex.octaves.float32

func value*(simplex: Simplex, x, y, z: float32): float32 =
  ## Generates the 3D noise value at (x, y, z) based on the Simplex parameters.

  if simplex.octaves == 0:
    raise newException(NoisyError, "Octaves must be > 0")

  var
    total: float32
    amplitude = simplex.amplitude
    frequency = simplex.frequency

  for _ in 0 ..< simplex.octaves:
    total += simplex.noise(
      x * frequency, y * frequency, z * frequency
    ) * amplitude
    amplitude *= simplex.gain
    frequency *= simplex.lacunarity

  total / simplex.octaves.float32

template value*(simplex: Simplex, x, y: int): float32 =
  simplex.value(x.float32, y.float32)

template value*(simplex: Simplex, x, y, z: int): float32 =
  simplex.value(x.float32, y.float32, z.float32)

func column4(simplex: Simplex, x, y, step: float32): m128 =
  let
    F2 = mm_set1_ps(F2)
    G2 = mm_set1_ps(G2)
    steps = cast[m128]([0.float32, step, step * 2, step * 3])
    x = mm_set1_ps(x)
    y = mm_set1_ps(y) + steps
    vec0 = mm_set1_ps(0)
    vec1 = mm_set1_ps(1)
    vec2 = mm_set1_ps(2)
    vec0dot5 = mm_set1_ps(0.5)
    vec255 = mm_set1_epi32(255)

  let
    s = (x + y) * F2
    i = floor(x + s)
    j = floor(y + s)
    t = (i + j) * G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    gt = x0 > y0
    i1 = blend(vec0, vec1, gt)
    j1 = blend(vec1, vec0, gt)
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - vec1 + vec2 * G2
    y2 = y0 - vec1 + vec2 * G2
    ii = mm_and_si128(mm_cvtps_epi32(i), vec255)
    jj = mm_and_si128(mm_cvtps_epi32(j), vec255)
    t0 = vec0dot5 - x0 * x0 - y0 * y0
    t1 = vec0dot5 - x1 * x1 - y1 * y1
    t2 = vec0dot5 - x2 * x2 - y2 * y2
    t0gt = t0 > vec0
    t1gt = t1 > vec0
    t2gt = t2 > vec0

  # Set up the gradient vectors
  let
    i1i = cast[array[4, int32]](mm_cvtps_epi32(i1))
    iii = cast[array[4, int32]](ii)
    j1i = cast[array[4, int32]](mm_cvtps_epi32(j1))
    jji = cast[array[4, int32]](jj)

    gx0 = cast[m128]([
      grad3[simplex.permMod12[iii[0] + simplex.perm[jji[0]]]][0],
      grad3[simplex.permMod12[iii[1] + simplex.perm[jji[1]]]][0],
      grad3[simplex.permMod12[iii[2] + simplex.perm[jji[2]]]][0],
      grad3[simplex.permMod12[iii[3] + simplex.perm[jji[3]]]][0],
    ])
    gy0 = cast[m128]([
      grad3[simplex.permMod12[iii[0] + simplex.perm[jji[0]]]][1],
      grad3[simplex.permMod12[iii[1] + simplex.perm[jji[1]]]][1],
      grad3[simplex.permMod12[iii[2] + simplex.perm[jji[2]]]][1],
      grad3[simplex.permMod12[iii[3] + simplex.perm[jji[3]]]][1],
    ])
    gx1  = cast[m128]([
      grad3[
        simplex.permMod12[iii[0] + i1i[0] + simplex.perm[jji[0] + j1i[0]]]
      ][0],
      grad3[
        simplex.permMod12[iii[1] + i1i[1] + simplex.perm[jji[1] + j1i[1]]]
      ][0],
      grad3[
        simplex.permMod12[iii[2] + i1i[2] + simplex.perm[jji[2] + j1i[2]]]
      ][0],
      grad3[
        simplex.permMod12[iii[3] + i1i[3] + simplex.perm[jji[3] + j1i[3]]]
      ][0]
    ])
    gy1  = cast[m128]([
      grad3[
        simplex.permMod12[iii[0] + i1i[0] + simplex.perm[jji[0] + j1i[0]]]
      ][1],
      grad3[
        simplex.permMod12[iii[1] + i1i[1] + simplex.perm[jji[1] + j1i[1]]]
      ][1],
      grad3[
        simplex.permMod12[iii[2] + i1i[2] + simplex.perm[jji[2] + j1i[2]]]
      ][1],
      grad3[
        simplex.permMod12[iii[3] + i1i[3] + simplex.perm[jji[3] + j1i[3]]]
      ][1]
    ])
    gx2  = cast[m128]([
      grad3[simplex.permMod12[iii[0] + 1 + simplex.perm[jji[0] + 1]]][0],
      grad3[simplex.permMod12[iii[1] + 1 + simplex.perm[jji[1] + 1]]][0],
      grad3[simplex.permMod12[iii[2] + 1 + simplex.perm[jji[2] + 1]]][0],
      grad3[simplex.permMod12[iii[3] + 1 + simplex.perm[jji[3] + 1]]][0]
    ])
    gy2  = cast[m128]([
      grad3[simplex.permMod12[iii[0] + 1 + simplex.perm[jji[0] + 1]]][1],
      grad3[simplex.permMod12[iii[1] + 1 + simplex.perm[jji[1] + 1]]][1],
      grad3[simplex.permMod12[iii[2] + 1 + simplex.perm[jji[2] + 1]]][1],
      grad3[simplex.permMod12[iii[3] + 1 + simplex.perm[jji[3] + 1]]][1]
    ])

  let
    n0 = blend(vec0, vec1, t0gt) * t0 * t0 * t0 * t0 * (gx0 * x0 + gy0 * y0)
    n1 = blend(vec0, vec1, t1gt) * t1 * t1 * t1 * t1 * (gx1 * x1 + gy1 * y1)
    n2 = blend(vec0, vec1, t2gt) * t2 * t2 * t2 * t2 * (gx2 * x2 + gy2 * y2)

  mm_set1_ps(70) * (n0 + n1 + n2)

func row4(simplex: Simplex, x, y, step: float32): array[4, m128] =
  for i in 0 ..< 4:
    result[i] = simplex.column4(x + i.float32 * step, y, step)

func grid4(simplex: Simplex, x, y: float32): array[4, m128] =
  ## Generates a 4x4 2D noise grid based on the Simplex parameters.
  ## Starts at (x, y) and moves by +1 in the x and y directions.
  ## Uses SSE2 SIMD insructions.

  if simplex.octaves == 0:
    failOctaves()

  var
    totals: array[4, m128]
    amplitude = mm_set1_ps(simplex.amplitude)
    gain = mm_set1_ps(simplex.gain)
    frequency = simplex.frequency

  for _ in 0 ..< simplex.octaves:
    let rows = simplex.row4(x * frequency, y * frequency, frequency)
    for i in 0 ..< 4:
      totals[i] += rows[i] * amplitude

    amplitude = amplitude * gain
    frequency *= simplex.lacunarity

  if simplex.octaves > 1:
    let octaves = mm_set1_ps(simplex.octaves.float32)
    for i in 0 ..< 4:
      totals[i] /= octaves

  totals

func layer4(simplex: Simplex, x, y, z, step: float32): m128 =
  let
    F3 = mm_set1_ps(F3)
    G3 = mm_set1_ps(G3)
    steps = cast[m128]([0.float32, step, step * 2, step * 3])
    x = mm_set1_ps(x)
    y = mm_set1_ps(y)
    z = mm_set1_ps(z) + steps
    vec0 = mm_set1_ps(0)
    vec1 = mm_set1_ps(1)
    vec2 = mm_set1_ps(2)
    vec3 = mm_set1_ps(3)
    vec0dot6 = mm_set1_ps(0.6)
    vec255 = mm_set1_epi32(255)

  let
    s = (x + y + z) * F3
    i = floor(x + s)
    j = floor(y + s)
    k = floor(z + s)
    t = (i + j + k) * G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)
    x0gey0 = x0 >= y0
    y0gez0 = y0 >= z0
    x0gez0 = x0 >= z0
    i1 = blend(vec0, vec1, x0gey0 and x0gez0)
    j1 = blend(vec0, vec1, mm_andnot_ps(x0gey0, y0gez0))
    k1 = blend(vec0, vec1, mm_andnot_ps(y0gez0, vec1))
    i2 = blend(vec0, vec1, x0gey0 or (x0gez0 and y0gez0))
    j2 = blend(vec0, vec1, (x0gey0 and y0gez0) or mm_andnot_ps(x0gey0, vec1))
    k2 = blend(
      vec0,
      vec1,
      (x0gey0 and mm_andnot_ps(y0gez0, vec1)) or
        (mm_andnot_ps(x0gey0, vec1) and mm_andnot_ps(x0gez0, vec1))
    )
    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + vec2 * G3
    y2 = y0 - j2 + vec2 * G3
    z2 = z0 - k2 + vec2 * G3
    x3 = x0 - vec1 + vec3 * G3
    y3 = y0 - vec1 + vec3 * G3
    z3 = z0 - vec1 + vec3 * G3
    ii = mm_and_si128(mm_cvtps_epi32(i), vec255)
    jj = mm_and_si128(mm_cvtps_epi32(j), vec255)
    kk = mm_and_si128(mm_cvtps_epi32(k), vec255)
    t0 = vec0dot6 - x0 * x0 - y0 * y0 - z0 * z0
    t1 = vec0dot6 - x1 * x1 - y1 * y1 - z1 * z1
    t2 = vec0dot6 - x2 * x2 - y2 * y2 - z2 * z2
    t3 = vec0dot6 - x3 * x3 - y3 * y3 - z3 * z3
    t0gt = t0 > vec0
    t1gt = t1 > vec0
    t2gt = t2 > vec0
    t3gt = t3 > vec0

  # Set up the gradient vectors
  let
    i1i = cast[array[4, int32]](mm_cvtps_epi32(i1))
    i2i = cast[array[4, int32]](mm_cvtps_epi32(i2))
    iii = cast[array[4, int32]](ii)
    j1i = cast[array[4, int32]](mm_cvtps_epi32(j1))
    j2i = cast[array[4, int32]](mm_cvtps_epi32(j2))
    jji = cast[array[4, int32]](jj)
    k1i = cast[array[4, int32]](mm_cvtps_epi32(k1))
    k2i = cast[array[4, int32]](mm_cvtps_epi32(k2))
    kki = cast[array[4, int32]](kk)

    gx0 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + simplex.perm[jji[0] + simplex.perm[kki[0]]]
      ]][0],
      grad3[simplex.permMod12[
        iii[1] + simplex.perm[jji[1] + simplex.perm[kki[1]]]
      ]][0],
      grad3[simplex.permMod12[
        iii[2] + simplex.perm[jji[2] + simplex.perm[kki[2]]]
      ]][0],
      grad3[simplex.permMod12[
        iii[3] + simplex.perm[jji[3] + simplex.perm[kki[3]]]
      ]][0]
    ])
    gy0 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + simplex.perm[jji[0] + simplex.perm[kki[0]]]
      ]][1],
      grad3[simplex.permMod12[
        iii[1] + simplex.perm[jji[1] + simplex.perm[kki[1]]]
      ]][1],
      grad3[simplex.permMod12[
        iii[2] + simplex.perm[jji[2] + simplex.perm[kki[2]]]
      ]][1],
      grad3[simplex.permMod12[
        iii[3] + simplex.perm[jji[3] + simplex.perm[kki[3]]]
      ]][1]
    ])
    gz0 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + simplex.perm[jji[0] + simplex.perm[kki[0]]]
      ]][2],
      grad3[simplex.permMod12[
        iii[1] + simplex.perm[jji[1] + simplex.perm[kki[1]]]
      ]][2],
      grad3[simplex.permMod12[
        iii[2] + simplex.perm[jji[2] + simplex.perm[kki[2]]]
      ]][2],
      grad3[simplex.permMod12[
        iii[3] + simplex.perm[jji[3] + simplex.perm[kki[3]]]
      ]][2]
    ])
    gx1 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + i1i[0] + simplex.perm[
          jji[0] + j1i[0] + simplex.perm[kki[0] + k1i[0]]
        ]
      ]][0],
      grad3[simplex.permMod12[
        iii[1] + i1i[1] + simplex.perm[
          jji[1] + j1i[1] + simplex.perm[kki[1] + k1i[1]]
        ]
      ]][0],
      grad3[simplex.permMod12[
        iii[2] + i1i[2] + simplex.perm[
          jji[2] + j1i[2] + simplex.perm[kki[2] + k1i[2]]
        ]
      ]][0],
      grad3[simplex.permMod12[
        iii[3] + i1i[3] + simplex.perm[
          jji[3] + j1i[3] + simplex.perm[kki[3] + k1i[3]]
        ]
      ]][0]
    ])
    gy1 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + i1i[0] + simplex.perm[
          jji[0] + j1i[0] + simplex.perm[kki[0] + k1i[0]]
        ]
      ]][1],
      grad3[simplex.permMod12[
        iii[1] + i1i[1] + simplex.perm[
          jji[1] + j1i[1] + simplex.perm[kki[1] + k1i[1]]
        ]
      ]][1],
      grad3[simplex.permMod12[
        iii[2] + i1i[2] + simplex.perm[
          jji[2] + j1i[2] + simplex.perm[kki[2] + k1i[2]]
        ]
      ]][1],
      grad3[simplex.permMod12[
        iii[3] + i1i[3] + simplex.perm[
          jji[3] + j1i[3] + simplex.perm[kki[3] + k1i[3]]
        ]
      ]][1]
    ])
    gz1 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + i1i[0] + simplex.perm[
          jji[0] + j1i[0] + simplex.perm[kki[0] + k1i[0]]
        ]
      ]][2],
      grad3[simplex.permMod12[
        iii[1] + i1i[1] + simplex.perm[
          jji[1] + j1i[1] + simplex.perm[kki[1] + k1i[1]]
        ]
      ]][2],
      grad3[simplex.permMod12[
        iii[2] + i1i[2] + simplex.perm[
          jji[2] + j1i[2] + simplex.perm[kki[2] + k1i[2]]
        ]
      ]][2],
      grad3[simplex.permMod12[
        iii[3] + i1i[3] + simplex.perm[
          jji[3] + j1i[3] + simplex.perm[kki[3] + k1i[3]]
        ]
      ]][2]
    ])
    gx2 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + i2i[0] + simplex.perm[
          jji[0] + j2i[0] + simplex.perm[kki[0] + k2i[0]]
        ]
      ]][0],
      grad3[simplex.permMod12[
        iii[1] + i2i[1] + simplex.perm[
          jji[1] + j2i[1] + simplex.perm[kki[1] + k2i[1]]
        ]
      ]][0],
      grad3[simplex.permMod12[
        iii[2] + i2i[2] + simplex.perm[
          jji[2] + j2i[2] + simplex.perm[kki[2] + k2i[2]]
        ]
      ]][0],
      grad3[simplex.permMod12[
        iii[3] + i2i[3] + simplex.perm[
          jji[3] + j2i[3] + simplex.perm[kki[3] + k2i[3]]
        ]
      ]][0]
    ])
    gy2 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + i2i[0] + simplex.perm[
          jji[0] + j2i[0] + simplex.perm[kki[0] + k2i[0]]
        ]
      ]][1],
      grad3[simplex.permMod12[
        iii[1] + i2i[1] + simplex.perm[
          jji[1] + j2i[1] + simplex.perm[kki[1] + k2i[1]]
        ]
      ]][1],
      grad3[simplex.permMod12[
        iii[2] + i2i[2] + simplex.perm[
          jji[2] + j2i[2] + simplex.perm[kki[2] + k2i[2]]
        ]
      ]][1],
      grad3[simplex.permMod12[
        iii[3] + i2i[3] + simplex.perm[
          jji[3] + j2i[3] + simplex.perm[kki[3] + k2i[3]]
        ]
      ]][1]
    ])
    gz2 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + i2i[0] + simplex.perm[
          jji[0] + j2i[0] + simplex.perm[kki[0] + k2i[0]]
        ]
      ]][2],
      grad3[simplex.permMod12[
        iii[1] + i2i[1] + simplex.perm[
          jji[1] + j2i[1] + simplex.perm[kki[1] + k2i[1]]
        ]
      ]][2],
      grad3[simplex.permMod12[
        iii[2] + i2i[2] + simplex.perm[
          jji[2] + j2i[2] + simplex.perm[kki[2] + k2i[2]]
        ]
      ]][2],
      grad3[simplex.permMod12[
        iii[3] + i2i[3] + simplex.perm[
          jji[3] + j2i[3] + simplex.perm[kki[3] + k2i[3]]
        ]
      ]][2]
    ])
    gx3 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + 1 + simplex.perm[jji[0] + 1 + simplex.perm[kki[0] + 1]
      ]]][0],
      grad3[simplex.permMod12[
        iii[1] + 1 + simplex.perm[jji[1] + 1 + simplex.perm[kki[1] + 1]
      ]]][0],
      grad3[simplex.permMod12[
        iii[2] + 1 + simplex.perm[jji[2] + 1 + simplex.perm[kki[2] + 1]
      ]]][0],
      grad3[simplex.permMod12[
        iii[3] + 1 + simplex.perm[jji[3] + 1 + simplex.perm[kki[3] + 1]
      ]]][0]
    ])
    gy3 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + 1 + simplex.perm[jji[0] + 1 + simplex.perm[kki[0] + 1]
      ]]][1],
      grad3[simplex.permMod12[
        iii[1] + 1 + simplex.perm[jji[1] + 1 + simplex.perm[kki[1] + 1]
      ]]][1],
      grad3[simplex.permMod12[
        iii[2] + 1 + simplex.perm[jji[2] + 1 + simplex.perm[kki[2] + 1]
      ]]][1],
      grad3[simplex.permMod12[
        iii[3] + 1 + simplex.perm[jji[3] + 1 + simplex.perm[kki[3] + 1]
      ]]][1]
    ])
    gz3 = cast[m128]([
      grad3[simplex.permMod12[
        iii[0] + 1 + simplex.perm[jji[0] + 1 + simplex.perm[kki[0] + 1]
      ]]][2],
      grad3[simplex.permMod12[
        iii[1] + 1 + simplex.perm[jji[1] + 1 + simplex.perm[kki[1] + 1]
      ]]][2],
      grad3[simplex.permMod12[
        iii[2] + 1 + simplex.perm[jji[2] + 1 + simplex.perm[kki[2] + 1]
      ]]][2],
      grad3[simplex.permMod12[
        iii[3] + 1 + simplex.perm[jji[3] + 1 + simplex.perm[kki[3] + 1]
      ]]][2]
    ])

  let
    n0 = blend(vec0, vec1, t0gt) *
      t0 * t0 * t0 * t0 * (gx0 * x0 + gy0 * y0 + gz0 * z0)
    n1 = blend(vec0, vec1, t1gt) *
      t1 * t1 * t1 * t1 * (gx1 * x1 + gy1 * y1 + gz1 * z1)
    n2 = blend(vec0, vec1, t2gt) *
      t2 * t2 * t2 * t2 * (gx2 * x2 + gy2 * y2 + gz2 * z2)
    n3 = blend(vec0, vec1, t3gt) *
      t3 * t3 * t3 * t3 * (gx3 * x3 + gy3 * y3 + gz3 * z3)

  mm_set1_ps(32) * (n0 + n1 + n2 + n3)

func column4(simplex: Simplex, x, y, z, step: float32): array[4, m128] =
  for i in 0 ..< 4:
    result[i] = simplex.layer4(x, y + i.float32 * step, z, step)

func row4(simplex: Simplex, x, y, z, step: float32): array[4, array[4, m128]] =
  for i in 0 ..< 4:
    result[i] = simplex.column4(x + i.float32 * step, y, z, step)

func grid4(
  simplex: Simplex, x, y, z: float32
): array[4, array[4, m128]] =
  ## Generates a 4x4 2D noise grid based on the Simplex parameters.
  ## Starts at (x, y) and moves by +1 in the x and y directions.
  ## Uses SSE2 SIMD insructions.

  if simplex.octaves == 0:
    failOctaves()

  var
    totals: array[4, array[4, m128]]
    amplitude = mm_set1_ps(simplex.amplitude)
    gain = mm_set1_ps(simplex.gain)
    frequency = simplex.frequency

  for _ in 0 ..< simplex.octaves:
    let rows = simplex.row4(
      x * frequency, y * frequency, z * frequency, frequency
    )
    for i in 0 ..< 4:
      for j in 0 ..< 4:
        totals[i][j] += rows[i][j] * amplitude

    amplitude = amplitude * gain
    frequency *= simplex.lacunarity

  if simplex.octaves > 1:
    let octaves = mm_set1_ps(simplex.octaves.float32)
    for i in 0 ..< 4:
      for j in 0 ..< 4:
        totals[i][j] /= octaves

  totals

func grid*(simplex: Simplex, x, y: float32, width, height: int): Grid =
  result = Grid()
  result.width = width
  result.height = height
  result.depth = 1
  result.values.setLen(width * height)

  var widthDone, heightDone: int
  for i in countup(0, width - 4, 4):
    for j in countup(0, height - 4, 4):
      let grid4 = simplex.grid4(x + i.float32, y + j.float32)
      for a in 0 ..< 4:
        mm_storeu_ps(
          result.values[result.valueIndex(a + i, j, 0)].unsafeAddr,
          grid4[a]
        )
      widthDone = i + 4
      heightDone = j + 4

  # Fill any width leftover
  for i in widthDone ..< width:
    for j in 0 ..< heightDone:
      result[i, j, 0] = simplex.value(x + i.float32, y + j.float32)

  # Fill any height leftover
  for i in 0 ..< width:
    for j in heightDone ..< height:
      result[i, j, 0] = simplex.value(x + i.float32, y + j.float32)

template grid*(simplex: Simplex, x, y, width, height: int): Grid =
  simplex.grid(x.float32, y.float32, width, height)

func grid*(
  simplex: Simplex, x, y, z: float32, width, height, depth: int
): Grid =
  result = Grid()
  result.width = width
  result.height = height
  result.depth = depth
  result.values.setLen(width * height * depth)

  var widthDone, heightDone, depthDone: int
  for i in countup(0, width - 4, 4):
    for j in countup(0, height - 4, 4):
      for k in countup(0, depth - 4, 4):
        let grid4 = simplex.grid4(
          x + i.float32, y + j.float32, z + k.float32
        )
        for a in 0 ..< 4:
          for b in 0 ..< 4:
            mm_storeu_ps(
              result.values[result.valueIndex(a + i, b + j, k)].unsafeAddr,
              grid4[a][b]
            )
        widthDone = i + 4
        heightDone = j + 4
        depthDone = k + 4

  # Fill any width leftover
  for i in widthDone ..< width:
    for j in 0 ..< heightDone:
      for k in 0 ..< depthDone:
        result[i, j, k] = simplex.value(
          x + i.float32, y + j.float32, z + k.float32
        )

  # Fill any height leftover
  for i in 0 ..< width:
    for j in heightDone ..< height:
      for k in 0 ..< depthDone:
        result[i, j, k] = simplex.value(
          x + i.float32, y + j.float32, z + k.float32
        )

  # Fill any depth leftover
  for i in 0 ..< width:
    for j in 0 ..< height:
      for k in depthDone ..< depth:
        result[i, j, k] = simplex.value(
          x + i.float32, y + j.float32, z + k.float32
        )

template grid*(simplex: Simplex, x, y, z, width, depth, height: int): Grid =
  simplex.grid(x.float32, y.float32, z.float32, width, depth, height)

when defined(release):
  {.pop.}

when isMainModule:
  import fidget/opengl/perf

  var s = initSimplex(1988)
  s.octaves = 3
  s.frequency = 4
  s.amplitude = 0.2
  s.lacunarity = 1.5
  s.gain = 4.3

  # let g = s.grid(0, 0, 4, 4)
  # debugEcho g.values

  # timeIt "3d normal":
  #   var c: int
  #   var q: float
  #   for x in 0 ..< 4:
  #     for y in 0 ..< 4:
  #       for z in countup(-120000, 240000-1, 1):
  #         q += s.value(x, y, z)
  #         inc c
  #   debugecho "verify: ", c, " ", q

  # timeIt "3d simd":
  #   var c: int
  #   var q: float
  #   for z in countup(-120000, 240000-1, 4):
  #     let tmp = s.grid4(0.float32, 0.float32, z.float32)
  #     for i in 0 ..< 4:
  #       for j in 0 ..< 4:
  #         for k in 0 ..< 4:
  #           q = q + tmp[i][j][k]
  #     inc(c, 64)
  #   debugecho "verify: ", c, " ", q

  # import chroma, flippy

  # var s = initSimplex(1988)
  # let img = newImage(256, 256, 3)
  # for x in 0 ..< 256:
  #   for y in 0 ..< 256:
  #     let
  #       v0 = s.value(x.float32 * 0.1.float32, y.float32 * 0.1.float32, 0.float32)
  #       c0 = (((v0 + 1) / 2) * 255)
  #     img.putRgba(x, y, rgba(c0, c0, c0, 255))

  # img.save("noise.png")
