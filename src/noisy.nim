import math, nimsimd/sse2, random

func m128(a: float32): M128 {.inline.} = mm_set1_ps(a)

func m128i(a: int32): M128i {.inline.} = mm_set1_epi32(a)

func `and`(a, b: M128): M128 {.inline.} = mm_and_ps(a, b)

func `and`(a, b: M128i): M128i {.inline.} = mm_and_si128(a, b)

func `or`(a, b: M128): M128 {.inline.} = mm_or_ps(a, b)

func `xor`(a, b: M128): M128 {.inline.} = mm_xor_ps(a, b)

func `>`(a, b: M128): M128 {.inline.} = mm_cmpgt_ps(a, b)

func `>=`(a, b: M128): M128 {.inline.} = mm_cmpge_ps(a, b)

func `<`(a, b: M128): M128 {.inline.} = mm_cmplt_ps(a, b)

func `+`(a, b: M128): M128 {.inline.} = mm_add_ps(a, b)

func `+=`(a: var M128, b: M128) {.inline.} = a = a + b

func `-`(a, b: M128): M128 {.inline.} = mm_sub_ps(a, b)

func `*`(a, b: M128): M128 {.inline.} = mm_mul_ps(a, b)

func `/`(a, b: M128): M128 {.inline.} = mm_div_ps(a, b)

func `/=`(a: var M128, b: M128) {.inline.} = a = a / b

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
    perm*: array[512, int32]
    permMod12*: array[512, int32]

  Grid* = ref object
    width*, height*, depth*: int
    values*: seq[float32]

  NoisyError* = object of CatchableError

when defined(release):
  {.push checks: off.}

template failOctaves() =
  raise newException(NoisyError, "Octaves must be > 0")

func floor(a: M128): M128 {.inline.} =
  const one = [1.float32, 1, 1, 1]
  let tmp = mm_cvtepi32_ps(mm_cvttps_epi32(a))
  tmp - ((a < tmp) and cast[M128](one))

func blend(a, b, mask: M128): M128 {.inline.} =
  ((a xor b) and mask) xor a

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

func valueIndex(g: Grid, x, y, z: int): int {.inline.} =
  z + y * g.depth + x * g.height * g.depth

func `[]`*(g: Grid, x, y: int, z = 0): float32 {.inline.} =
  ## Returns the noise value at (x, y) or (x, y, z).
  if x < 0 or x >= g.width:
    raise newException(IndexDefect, "Index x out of bounds")
  if y < 0 or y >= g.height:
    raise newException(IndexDefect, "Index y out of bounds")
  if z < 0 or z >= g.depth:
    raise newException(IndexDefect, "Index z out of bounds")
  g.values[g.valueIndex(x, y, z)]

func `[]=`(g: var Grid, x, y, z: int, value: float32) {.inline.} =
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
    i1 = (x0 >= y0 and x0 >= z0).int32
    j1 = (x0 < y0 and y0 >= z0).int32
    k1 = (x0 < z0 and y0 < z0).int32
    i2 = (x0 >= y0 or x0 >= z0).int32
    j2 = (x0 < y0 or y0 >= z0).int32
    k2 = (x0 < z0 or y0 < z0).int32
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

func value*(simplex: Simplex, x, y: int): float32 {.inline.} =
  ## Helper for working with ints.
  simplex.value(x.float32, y.float32)

func value*(simplex: Simplex, x, y, z: int): float32 {.inline.} =
  ## Helper for working with ints
  simplex.value(x.float32, y.float32, z.float32)

func column4(simplex: Simplex, x, y, step: float32): M128 =
  let
    F2 = m128(F2)
    G2 = m128(G2)
    steps = cast[M128]([0.float32, step, step * 2, step * 3])
    x = m128(x)
    y = m128(y) + steps
    vec0 = m128(0)
    vec1 = m128(1)
    vec2 = m128(2)
    vec0dot5 = m128(0.5)
    vec255i = m128i(255)
    vec1i = m128i(1)

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
    ii = mm_cvtps_epi32(i) and vec255i
    jj = mm_cvtps_epi32(j) and vec255i
    t0 = vec0dot5 - x0 * x0 - y0 * y0
    t1 = vec0dot5 - x1 * x1 - y1 * y1
    t2 = vec0dot5 - x2 * x2 - y2 * y2
    t0gt = t0 > vec0
    t1gt = t1 > vec0
    t2gt = t2 > vec0

  # Set up the gradient vectors
  let
    jji = cast[array[4, int32]](jj)
    jjj1 = cast[array[4, int32]](mm_add_epi32(jj, mm_cvtps_epi32(j1)))
    jjplus1 = cast[array[4, int32]](mm_add_epi32(jj, vec1i))
    jjperm = cast[M128i]([
      simplex.perm[jji[0]],
      simplex.perm[jji[1]],
      simplex.perm[jji[2]],
      simplex.perm[jji[3]]
    ])
    jjj1perm = cast[M128i]([
      simplex.perm[jjj1[0]],
      simplex.perm[jjj1[1]],
      simplex.perm[jjj1[2]],
      simplex.perm[jjj1[3]]
    ])
    jjplus1perm = cast[M128i]([
      simplex.perm[jjplus1[0]],
      simplex.perm[jjplus1[1]],
      simplex.perm[jjplus1[2]],
      simplex.perm[jjplus1[3]]
    ])
    iiplusjjperm = cast[array[4, int32]](mm_add_epi32(ii, jjperm))
    iii1plusjjj1perm = cast[array[4, int32]](mm_add_epi32(
      mm_add_epi32(ii, mm_cvtps_epi32(i1)), jjj1perm)
    )
    iiplus1plusjjplus1perm = cast[array[4, int32]](
      mm_add_epi32(mm_add_epi32(ii, vec1i), jjplus1perm)
    )
    gx0 = cast[M128]([
      grad3[simplex.permMod12[iiplusjjperm[0]]][0],
      grad3[simplex.permMod12[iiplusjjperm[1]]][0],
      grad3[simplex.permMod12[iiplusjjperm[2]]][0],
      grad3[simplex.permMod12[iiplusjjperm[3]]][0],
    ])
    gy0 = cast[M128]([
      grad3[simplex.permMod12[iiplusjjperm[0]]][1],
      grad3[simplex.permMod12[iiplusjjperm[1]]][1],
      grad3[simplex.permMod12[iiplusjjperm[2]]][1],
      grad3[simplex.permMod12[iiplusjjperm[3]]][1],
    ])
    gx1 = cast[M128]([
      grad3[simplex.permMod12[iii1plusjjj1perm[0]]][0],
      grad3[simplex.permMod12[iii1plusjjj1perm[1]]][0],
      grad3[simplex.permMod12[iii1plusjjj1perm[2]]][0],
      grad3[simplex.permMod12[iii1plusjjj1perm[3]]][0]
    ])
    gy1 = cast[M128]([
      grad3[simplex.permMod12[iii1plusjjj1perm[0]]][1],
      grad3[simplex.permMod12[iii1plusjjj1perm[1]]][1],
      grad3[simplex.permMod12[iii1plusjjj1perm[2]]][1],
      grad3[simplex.permMod12[iii1plusjjj1perm[3]]][1]
    ])
    gx2 = cast[M128]([
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[0]]][0],
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[1]]][0],
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[2]]][0],
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[3]]][0]
    ])
    gy2 = cast[M128]([
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[0]]][1],
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[1]]][1],
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[2]]][1],
      grad3[simplex.permMod12[iiplus1plusjjplus1perm[3]]][1]
    ])

  let
    n0 = blend(vec0, vec1, t0gt) * t0 * t0 * t0 * t0 * (gx0 * x0 + gy0 * y0)
    n1 = blend(vec0, vec1, t1gt) * t1 * t1 * t1 * t1 * (gx1 * x1 + gy1 * y1)
    n2 = blend(vec0, vec1, t2gt) * t2 * t2 * t2 * t2 * (gx2 * x2 + gy2 * y2)

  m128(70) * (n0 + n1 + n2)

func row4(simplex: Simplex, x, y, step: float32): array[4, M128] =
  for i in 0 ..< 4:
    result[i] = simplex.column4(x + i.float32 * step, y, step)

func grid4(simplex: Simplex, x, y: float32): array[4, M128] =
  ## Generates a 4x4 2D noise grid based on the Simplex parameters.
  ## Starts at (x, y) and moves by +1 in the x and y directions.
  ## Uses SSE2 SIMD insructions.

  if simplex.octaves == 0:
    failOctaves()

  var
    totals: array[4, M128]
    amplitude = m128(simplex.amplitude)
    gain = m128(simplex.gain)
    frequency = simplex.frequency

  for _ in 0 ..< simplex.octaves:
    let rows = simplex.row4(x * frequency, y * frequency, frequency)
    for i in 0 ..< 4:
      totals[i] += rows[i] * amplitude

    amplitude = amplitude * gain
    frequency *= simplex.lacunarity

  if simplex.octaves > 1:
    let octaves = m128(simplex.octaves.float32)
    for i in 0 ..< 4:
      totals[i] /= octaves

  totals

func layer4(simplex: Simplex, x, y, z, step: float32): M128 =
  let
    F3 = m128(F3)
    G3 = m128(G3)
    steps = cast[M128]([0.float32, step, step * 2, step * 3])
    x = m128(x)
    y = m128(y)
    z = m128(z) + steps
    vec0 = m128(0)
    vec1 = m128(1)
    vec2 = m128(2)
    vec3 = m128(3)
    vec0dot6 = m128(0.6)
    vec255i = m128i(255)
    vec1i = m128i(1)

  let
    s = (x + y + z) * F3
    i = floor(x + s)
    j = floor(y + s)
    k = floor(z + s)
    t = (i + j + k) * G3
    x0 = x - (i - t)
    y0 = y - (j - t)
    z0 = z - (k - t)
    i1 = blend(vec0, vec1, (x0 >= y0 and x0 >= z0))
    j1 = blend(vec0, vec1, (x0 < y0 and y0 >= z0))
    k1 = blend(vec0, vec1, (x0 < z0 and y0 < z0))
    i2 = blend(vec0, vec1, (x0 >= y0 or x0 >= z0))
    j2 = blend(vec0, vec1, (x0 < y0 or y0 >= z0))
    k2 = blend(vec0, vec1, (x0 < z0 or y0 < z0))
    x1 = x0 - i1 + G3
    y1 = y0 - j1 + G3
    z1 = z0 - k1 + G3
    x2 = x0 - i2 + vec2 * G3
    y2 = y0 - j2 + vec2 * G3
    z2 = z0 - k2 + vec2 * G3
    x3 = x0 - vec1 + vec3 * G3
    y3 = y0 - vec1 + vec3 * G3
    z3 = z0 - vec1 + vec3 * G3
    ii = mm_cvtps_epi32(i) and vec255i
    jj = mm_cvtps_epi32(j) and vec255i
    kk = mm_cvtps_epi32(k) and vec255i
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
    kki = cast[array[4, int32]](kk)
    kkk1 = cast[array[4, int32]](mm_add_epi32(kk, mm_cvtps_epi32(k1)))
    kkk2 = cast[array[4, int32]](mm_add_epi32(kk, mm_cvtps_epi32(k2)))
    kkplus1 = cast[array[4, int32]](mm_add_epi32(kk, vec1i))
    kkperm = cast[M128i]([
      simplex.perm[kki[0]],
      simplex.perm[kki[1]],
      simplex.perm[kki[2]],
      simplex.perm[kki[3]]
    ])
    kkk1perm = cast[M128i]([
      simplex.perm[kkk1[0]],
      simplex.perm[kkk1[1]],
      simplex.perm[kkk1[2]],
      simplex.perm[kkk1[3]]
    ])
    kkk2perm = cast[M128i]([
      simplex.perm[kkk2[0]],
      simplex.perm[kkk2[1]],
      simplex.perm[kkk2[2]],
      simplex.perm[kkk2[3]]
    ])
    kkplus1perm = cast[M128i]([
      simplex.perm[kkplus1[0]],
      simplex.perm[kkplus1[1]],
      simplex.perm[kkplus1[2]],
      simplex.perm[kkplus1[3]]
    ])
    jjpluskkperm = cast[array[4, int32]](mm_add_epi32(jj, kkperm))
    jjj1pluskkk1perm = cast[array[4, int32]](
      mm_add_epi32(mm_add_epi32(jj, mm_cvtps_epi32(j1)), kkk1perm)
    )
    jjj2pluskkk2perm = cast[array[4, int32]](
      mm_add_epi32(mm_add_epi32(jj, mm_cvtps_epi32(j2)), kkk2perm)
    )
    jjplus1pluskkplus1perm = cast[array[4, int32]](
      mm_add_epi32(mm_add_epi32(jj, vec1i), kkplus1perm)
    )
    jjpluskpermperm = cast[M128i]([
      simplex.perm[jjpluskkperm[0]],
      simplex.perm[jjpluskkperm[1]],
      simplex.perm[jjpluskkperm[2]],
      simplex.perm[jjpluskkperm[3]]
    ])
    jjj1pluskkk1permperm = cast[M128i]([
      simplex.perm[jjj1pluskkk1perm[0]],
      simplex.perm[jjj1pluskkk1perm[1]],
      simplex.perm[jjj1pluskkk1perm[2]],
      simplex.perm[jjj1pluskkk1perm[3]]
    ])
    jjj2pluskkk2permperm = cast[M128i]([
      simplex.perm[jjj2pluskkk2perm[0]],
      simplex.perm[jjj2pluskkk2perm[1]],
      simplex.perm[jjj2pluskkk2perm[2]],
      simplex.perm[jjj2pluskkk2perm[3]],
    ])
    jjplus1pluskkplus1permperm = cast[M128i]([
      simplex.perm[jjplus1pluskkplus1perm[0]],
      simplex.perm[jjplus1pluskkplus1perm[1]],
      simplex.perm[jjplus1pluskkplus1perm[2]],
      simplex.perm[jjplus1pluskkplus1perm[3]]
    ])
    iiplusjjpluskpermperm = cast[array[4, int32]](
      mm_add_epi32(ii, jjpluskpermperm)
    )
    iii1plusjjj1pluskkk1permperm = cast[array[4, int32]](
      mm_add_epi32(mm_add_epi32(ii, mm_cvtps_epi32(i1)), jjj1pluskkk1permperm)
    )
    iii2plusjjj2pluskkk2permperm = cast[array[4, int32]](
      mm_add_epi32(mm_add_epi32(ii, mm_cvtps_epi32(i2)), jjj2pluskkk2permperm)
    )
    iiplus1plusjjplus1pluskkplus1permperm = cast[array[4, int32]](
      mm_add_epi32(mm_add_epi32(ii, vec1i), jjplus1pluskkplus1permperm)
    )
    gx0 = cast[M128]([
      grad3[simplex.permMod12[iiplusjjpluskpermperm[0]]][0],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[1]]][0],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[2]]][0],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[3]]][0]
    ])
    gy0 = cast[M128]([
      grad3[simplex.permMod12[iiplusjjpluskpermperm[0]]][1],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[1]]][1],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[2]]][1],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[3]]][1]
    ])
    gz0 = cast[M128]([
      grad3[simplex.permMod12[iiplusjjpluskpermperm[0]]][2],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[1]]][2],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[2]]][2],
      grad3[simplex.permMod12[iiplusjjpluskpermperm[3]]][2]
    ])
    gx1 = cast[M128]([
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[0]]][0],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[1]]][0],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[2]]][0],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[3]]][0]
    ])
    gy1 = cast[M128]([
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[0]]][1],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[1]]][1],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[2]]][1],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[3]]][1]
    ])
    gz1 = cast[M128]([
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[0]]][2],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[1]]][2],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[2]]][2],
      grad3[simplex.permMod12[iii1plusjjj1pluskkk1permperm[3]]][2]
    ])
    gx2 = cast[M128]([
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[0]]][0],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[1]]][0],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[2]]][0],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[3]]][0]
    ])
    gy2 = cast[M128]([
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[0]]][1],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[1]]][1],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[2]]][1],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[3]]][1]
    ])
    gz2 = cast[M128]([
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[0]]][2],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[1]]][2],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[2]]][2],
      grad3[simplex.permMod12[iii2plusjjj2pluskkk2permperm[3]]][2]
    ])
    gx3 = cast[M128]([
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[0]]][0],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[1]]][0],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[2]]][0],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[3]]][0]
    ])
    gy3 = cast[M128]([
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[0]]][1],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[1]]][1],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[2]]][1],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[3]]][1]
    ])
    gz3 = cast[M128]([
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[0]]][2],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[1]]][2],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[2]]][2],
      grad3[simplex.permMod12[iiplus1plusjjplus1pluskkplus1permperm[3]]][2]
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

  m128(32) * (n0 + n1 + n2 + n3)

func column4(simplex: Simplex, x, y, z, step: float32): array[4, M128] =
  for i in 0 ..< 4:
    result[i] = simplex.layer4(x, y + i.float32 * step, z, step)

func row4(simplex: Simplex, x, y, z, step: float32): array[4, array[4, M128]] =
  for i in 0 ..< 4:
    result[i] = simplex.column4(x + i.float32 * step, y, z, step)

func grid4(
  simplex: Simplex, x, y, z: float32
): array[4, array[4, M128]] =
  ## Generates a 4x4 2D noise grid based on the Simplex parameters.
  ## Starts at (x, y) and moves by +1 in the x and y directions.
  ## Uses SSE2 SIMD insructions.

  if simplex.octaves == 0:
    failOctaves()

  var
    totals: array[4, array[4, M128]]
    amplitude = m128(simplex.amplitude)
    gain = m128(simplex.gain)
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
    let octaves = m128(simplex.octaves.float32)
    for i in 0 ..< 4:
      for j in 0 ..< 4:
        totals[i][j] /= octaves

  totals

func grid*(
  simplex: Simplex, start: (float32, float32), dimens: (int, int)
): Grid =
  ## Beginning at position start, generate a grid of 2D noise based on the
  ## Simplex parameters. The width and height of the grid is set by the dimens
  ## parameter.

  let
    (x, y) = start
    (width, height) = dimens

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

func grid*(
  simplex: Simplex, start: (int, int), dimens: (int, int)
): Grid {.inline.} =
  ## Helper for working with ints.
  simplex.grid((start[0].float32, start[1].float32), dimens)

func grid*(
  simplex: Simplex, start: (float32, float32, float32), dimens: (int, int, int)
): Grid =
  ## Beginning at position start, generate a grid of 3D noise based on the
  ## Simplex parameters. The width, depth, and height of the grid is set by
  ## the dimens parameter.

  let
    (x, y, z) = start
    (width, height, depth) = dimens

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

func grid*(
  simplex: Simplex, start: (int, int, int), dimens: (int, int, int)
): Grid {.inline.} =
  ## Helper for working with ints.
  simplex.grid((start[0].float32, start[1].float32, start[2].float32), dimens)

when defined(release):
  {.pop.}

when isMainModule:
  ## Draw an image to see what the noise looks like

  import pixie

  var s = initSimplex(1988)
  s.frequency = 0.1
  let
    image = newImage(480, 480)
    g = s.grid((0, 0), (480, 480))
  for x in 0 ..< 480:
    for y in 0 ..< 480:
      # for z in 0 ..< 1:
      let
        # v = s.value(x.float32, y.float32)#, z.float32)
        v = g[x, y] #, z]
        c = (((v + 1) / 2) * 255).uint8
      image[x, y] = rgba(c, c, c, 255)

  image.writeFile("examples/noise.png")
