import math, random

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
    perm*: array[256, uint8]
    permMod12*: array[256, uint8]

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
    result.perm[i] = i.uint8
  var r = initRand(seed)
  shuffle(r, result.perm)
  for i in 0 ..< result.perm.len:
    result.permMod12[i] = result.perm[i] mod 12

func dot(g: array[3, float32], x, y: float32): float32 {.inline.} =
  g[0] * x + g[1] * y

func dot(g: array[3, float32], x, y, z: float32): float32 {.inline.} =
  g[0] * x + g[1] * y + g[2] * z

import noisy/simd/sse2

func column4(simplex: Simplex, x, y, step: float32): m128 =
  let
    F2 = mm_set1_ps(F2)
    G2 = mm_set1_ps(G2)
    steps = cast[m128]([0.float32, step, step * 2, step * 3])
    x = mm_set1_ps(x.float32)
    y = mm_set1_ps(y.float32) + steps
    vec0 = mm_set1_ps(0)
    vec1 = mm_set1_ps(1)
    vec2 = mm_set1_ps(2)
    v0dot5 = mm_set1_ps(0.5)
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
    ii = mm_cvtps_epi32(i) and vec255
    jj = mm_cvtps_epi32(j) and vec255
    t0 = v0dot5 - x0 * x0 - y0 * y0
    t1 = v0dot5 - x1 * x1 - y1 * y1
    t2 = v0dot5 - x2 * x2 - y2 * y2
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
      grad3[simplex.permMod12[iii[0].uint8 + simplex.perm[jji[0].uint8]]][0],
      grad3[simplex.permMod12[iii[1].uint8 + simplex.perm[jji[1].uint8]]][0],
      grad3[simplex.permMod12[iii[2].uint8 + simplex.perm[jji[2].uint8]]][0],
      grad3[simplex.permMod12[iii[3].uint8 + simplex.perm[jji[3].uint8]]][0],
    ])
    gy0 = cast[m128]([
      grad3[simplex.permMod12[iii[0].uint8 + simplex.perm[jji[0].uint8]]][1],
      grad3[simplex.permMod12[iii[1].uint8 + simplex.perm[jji[1].uint8]]][1],
      grad3[simplex.permMod12[iii[2].uint8 + simplex.perm[jji[2].uint8]]][1],
      grad3[simplex.permMod12[iii[3].uint8 + simplex.perm[jji[3].uint8]]][1],
    ])
    gx1  = cast[m128]([
      grad3[
        simplex.permMod12[iii[0].uint8 + i1i[0].uint8 +
        simplex.perm[jji[0].uint8 + j1i[0].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[1].uint8 + i1i[1].uint8 +
        simplex.perm[jji[1].uint8 + j1i[1].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[2].uint8 + i1i[2].uint8 +
        simplex.perm[jji[2].uint8 + j1i[2].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[3].uint8 + i1i[3].uint8 +
        simplex.perm[jji[3].uint8 + j1i[3].uint8]]
      ][0]
    ])
    gy1  = cast[m128]([
      grad3[
        simplex.permMod12[iii[0].uint8 + i1i[0].uint8 +
        simplex.perm[jji[0].uint8 + j1i[0].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[1].uint8 + i1i[1].uint8 +
        simplex.perm[jji[1].uint8 + j1i[1].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[2].uint8 + i1i[2].uint8 +
        simplex.perm[jji[2].uint8 + j1i[2].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[3].uint8 + i1i[3].uint8 +
        simplex.perm[jji[3].uint8 + j1i[3].uint8]]
      ][1]
    ])
    gx2  = cast[m128]([
      grad3[
        simplex.permMod12[iii[0].uint8 + 1.uint8 +
        simplex.perm[jji[0].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[1].uint8 + 1.uint8 +
        simplex.perm[jji[1].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[2].uint8 + 1.uint8 +
        simplex.perm[jji[2].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[3].uint8 + 1.uint8 +
        simplex.perm[jji[3].uint8 + 1.uint8]]
      ][0]
    ])
    gy2  = cast[m128]([
      grad3[
        simplex.permMod12[iii[0].uint8 + 1.uint8 +
        simplex.perm[jji[0].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[1].uint8 + 1.uint8 +
        simplex.perm[jji[1].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[2].uint8 + 1.uint8 +
        simplex.perm[jji[2].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[3].uint8 + 1.uint8 +
        simplex.perm[jji[3].uint8 + 1.uint8]]
      ][1]
    ])

  let
    n0 = blend(vec0, vec1, t0gt) * t0 * t0 * t0 * t0 * (gx0 * x0 + gy0 * y0)
    n1 = blend(vec0, vec1, t1gt) * t1 * t1 * t1 * t1 * (gx1 * x1 + gy1 * y1)
    n2 = blend(vec0, vec1, t2gt) * t2 * t2 * t2 * t2 * (gx2 * x2 + gy2 * y2)

  mm_set1_ps(70.float32) * (n0 + n1 + n2)

func row4(simplex: Simplex, x, y, step: float32): array[4, m128] =
  for i in 0 ..< 4:
    result[i] = simplex.column4(x + i.float32 * step, y, step)

func grid4*(
  simplex: Simplex, x, y, step: float32
): array[4, array[4, float32]] =
  ## Generates a 4x4 2D noise grid based on the Simplex parameters.
  ## Starts at (x, y) and moves by step in the x and y directions.
  ## Uses SSE2 SIMD insructions.

  if simplex.octaves == 0:
    failOctaves()

  var
    totals: array[4, m128]
    amplitude = mm_set1_ps(simplex.amplitude)
    gain = mm_set1_ps(simplex.gain)
    frequency = simplex.frequency

  for _ in 0 ..< simplex.octaves:
    let rows = simplex.row4(x, y, step * frequency)
    for i in 0 ..< 4:
      totals[i] = totals[i] + rows[i] * amplitude

    amplitude = amplitude * gain
    frequency *= simplex.lacunarity

  let octaves = mm_set1_ps(simplex.octaves.float32)
  for i in 0 ..< 4:
    totals[i] = totals[i] / octaves

  cast[array[4, array[4, float32]]](totals)

import noisy/simd/avx

func column8(simplex: Simplex, x, y, step: float32): array[8, float32] =
  let
    F2 = mm256_set1_ps(F2)
    G2 = mm256_set1_ps(G2)
    steps = cast[m256]([
      0.float32, step, step * 2, step * 3,
      step * 4, step * 5, step * 6, step * 7
    ])
    x = mm256_set1_ps(x.float32)
    y = mm256_set1_ps(y.float32) + steps
    vec0 = mm256_set1_ps(0)
    vec1 = mm256_set1_ps(1)
    vec2 = mm256_set1_ps(2)
    v0dot5 = mm256_set1_ps(0.5)
    vec255 = mm256_set1_epi32(255)
    # vec255 = mm_set1_epi32(255)

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
    # iArr = cast[array[8, float32]](i)
    # jArr = cast[array[8, float32]](j)
    # ii0 = mm_cvtps_epi32(mm_loadu_ps(iArr[0].unsafeAddr)) and vec255
    # ii1 = mm_cvtps_epi32(mm_loadu_ps(iArr[4].unsafeAddr)) and vec255
    # jj0 = mm_cvtps_epi32(mm_loadu_ps(jArr[0].unsafeAddr)) and vec255
    # jj1 = mm_cvtps_epi32(mm_loadu_ps(jArr[4].unsafeAddr)) and vec255
    # ii = [ii0, ii1]
    # jj = [jj0, jj1]
    ii = mm256_cvtps_epi32(i) and vec255
    jj = mm256_cvtps_epi32(j) and vec255
    t0 = v0dot5 - x0 * x0 - y0 * y0
    t1 = v0dot5 - x1 * x1 - y1 * y1
    t2 = v0dot5 - x2 * x2 - y2 * y2
    t0gt = t0 > vec0
    t1gt = t1 > vec0
    t2gt = t2 > vec0

  # Set up the gradient vectors
  let
    i1i = cast[array[8, int32]](mm256_cvtps_epi32(i1))
    iii = cast[array[8, int32]](ii)
    j1i = cast[array[8, int32]](mm256_cvtps_epi32(j1))
    jji = cast[array[8, int32]](jj)

    gx0 = cast[m256]([
      grad3[simplex.permMod12[iii[0].uint8 + simplex.perm[jji[0].uint8]]][0],
      grad3[simplex.permMod12[iii[1].uint8 + simplex.perm[jji[1].uint8]]][0],
      grad3[simplex.permMod12[iii[2].uint8 + simplex.perm[jji[2].uint8]]][0],
      grad3[simplex.permMod12[iii[3].uint8 + simplex.perm[jji[3].uint8]]][0],
      grad3[simplex.permMod12[iii[4].uint8 + simplex.perm[jji[4].uint8]]][0],
      grad3[simplex.permMod12[iii[5].uint8 + simplex.perm[jji[5].uint8]]][0],
      grad3[simplex.permMod12[iii[6].uint8 + simplex.perm[jji[6].uint8]]][0],
      grad3[simplex.permMod12[iii[7].uint8 + simplex.perm[jji[7].uint8]]][0],
    ])
    gy0 = cast[m256]([
      grad3[simplex.permMod12[iii[0].uint8 + simplex.perm[jji[0].uint8]]][1],
      grad3[simplex.permMod12[iii[1].uint8 + simplex.perm[jji[1].uint8]]][1],
      grad3[simplex.permMod12[iii[2].uint8 + simplex.perm[jji[2].uint8]]][1],
      grad3[simplex.permMod12[iii[3].uint8 + simplex.perm[jji[3].uint8]]][1],
      grad3[simplex.permMod12[iii[4].uint8 + simplex.perm[jji[4].uint8]]][1],
      grad3[simplex.permMod12[iii[5].uint8 + simplex.perm[jji[5].uint8]]][1],
      grad3[simplex.permMod12[iii[6].uint8 + simplex.perm[jji[6].uint8]]][1],
      grad3[simplex.permMod12[iii[7].uint8 + simplex.perm[jji[7].uint8]]][1],
    ])
    gx1  = cast[m256]([
      grad3[
        simplex.permMod12[iii[0].uint8 + i1i[0].uint8 +
        simplex.perm[jji[0].uint8 + j1i[0].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[1].uint8 + i1i[1].uint8 +
        simplex.perm[jji[1].uint8 + j1i[1].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[2].uint8 + i1i[2].uint8 +
        simplex.perm[jji[2].uint8 + j1i[2].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[3].uint8 + i1i[3].uint8 +
        simplex.perm[jji[3].uint8 + j1i[3].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[4].uint8 + i1i[4].uint8 +
        simplex.perm[jji[4].uint8 + j1i[4].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[5].uint8 + i1i[5].uint8 +
        simplex.perm[jji[5].uint8 + j1i[5].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[6].uint8 + i1i[6].uint8 +
        simplex.perm[jji[6].uint8 + j1i[6].uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[7].uint8 + i1i[7].uint8 +
        simplex.perm[jji[7].uint8 + j1i[7].uint8]]
      ][0]
    ])
    gy1  = cast[m256]([
      grad3[
        simplex.permMod12[iii[0].uint8 + i1i[0].uint8 +
        simplex.perm[jji[0].uint8 + j1i[0].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[1].uint8 + i1i[1].uint8 +
        simplex.perm[jji[1].uint8 + j1i[1].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[2].uint8 + i1i[2].uint8 +
        simplex.perm[jji[2].uint8 + j1i[2].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[3].uint8 + i1i[3].uint8 +
        simplex.perm[jji[3].uint8 + j1i[3].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[4].uint8 + i1i[4].uint8 +
        simplex.perm[jji[4].uint8 + j1i[4].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[5].uint8 + i1i[5].uint8 +
        simplex.perm[jji[5].uint8 + j1i[5].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[6].uint8 + i1i[6].uint8 +
        simplex.perm[jji[6].uint8 + j1i[6].uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[7].uint8 + i1i[7].uint8 +
        simplex.perm[jji[7].uint8 + j1i[7].uint8]]
      ][1]
    ])
    gx2  = cast[m256]([
      grad3[
        simplex.permMod12[iii[0].uint8 + 1.uint8 +
        simplex.perm[jji[0].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[1].uint8 + 1.uint8 +
        simplex.perm[jji[1].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[2].uint8 + 1.uint8 +
        simplex.perm[jji[2].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[3].uint8 + 1.uint8 +
        simplex.perm[jji[3].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[4].uint8 + 1.uint8 +
        simplex.perm[jji[4].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[5].uint8 + 1.uint8 +
        simplex.perm[jji[5].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[6].uint8 + 1.uint8 +
        simplex.perm[jji[6].uint8 + 1.uint8]]
      ][0],
      grad3[
        simplex.permMod12[iii[7].uint8 + 1.uint8 +
        simplex.perm[jji[7].uint8 + 1.uint8]]
      ][0]
    ])
    gy2  = cast[m256]([
      grad3[
        simplex.permMod12[iii[0].uint8 + 1.uint8 +
        simplex.perm[jji[0].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[1].uint8 + 1.uint8 +
        simplex.perm[jji[1].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[2].uint8 + 1.uint8 +
        simplex.perm[jji[2].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[3].uint8 + 1.uint8 +
        simplex.perm[jji[3].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[4].uint8 + 1.uint8 +
        simplex.perm[jji[4].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[5].uint8 + 1.uint8 +
        simplex.perm[jji[5].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[6].uint8 + 1.uint8 +
        simplex.perm[jji[6].uint8 + 1.uint8]]
      ][1],
      grad3[
        simplex.permMod12[iii[7].uint8 + 1.uint8 +
        simplex.perm[jji[7].uint8 + 1.uint8]]
      ][1]
    ])

  let
    n0 = blend(vec0, vec1, t0gt) * t0 * t0 * t0 * t0 * (gx0 * x0 + gy0 * y0)
    n1 = blend(vec0, vec1, t1gt) * t1 * t1 * t1 * t1 * (gx1 * x1 + gy1 * y1)
    n2 = blend(vec0, vec1, t2gt) * t2 * t2 * t2 * t2 * (gx2 * x2 + gy2 * y2)

  # debugEcho cast[array[8, float32]](mm256_set1_ps(70.float32) * (n0 + n1 + n2))

  # debugEcho cast[array[8, float32]](r)
  cast[array[8, float32]](mm256_set1_ps(70.float32) * (n0 + n1 + n2))

func noise(simplex: Simplex, x, y: float32): float32 =
  let
    s = (x + y) * F2
    i = floor(x + s)
    j = floor(y + s)
    t = (i + j) * G2
    x0 = x - (i - t)
    y0 = y - (j - t)
    gt = (x0 > y0).uint8
    i1 = gt
    j1 = (not gt) and 1

  let
    x1 = x0 - i1.float32 + G2
    y1 = y0 - j1.float32 + G2
    x2 = x0 - 1.float32 + 2.float32 * G2
    y2 = y0 - 1.float32 + 2.float32 * G2
    ii = (i.int32 and 255).uint8
    jj = (j.int32 and 255).uint8
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

  var i1, j1, k1, i2, j2, k2: uint8
  if x0 >= y0:
    if y0 >= z0:
      i1 = 1
      j1 = 0
      k1 = 0
      i2 = 1
      j2 = 1
      k2 = 0
    elif x0 >= z0:
      i1 = 1
      j1 = 0
      k1 = 0
      i2 = 1
      j2 = 0
      k2 = 1
    else:
      i1 = 0
      j1 = 0
      k1 = 1
      i2 = 1
      j2 = 0
      k2 = 1
  else:
    if y0 < z0:
      i1 = 0
      j1 = 0
      k1 = 1
      i2 = 0
      j2 = 1
      k2 = 1
    elif x0 < z0:
      i1 = 0
      j1 = 1
      k1 = 0
      i2 = 0
      j2 = 1
      k2 = 1
    else:
      i1 = 0
      j1 = 1
      k1 = 0
      i2 = 1
      j2 = 1
      k2 = 0

  let
    x1 = x0 - i1.float32 + G3
    y1 = y0 - j1.float32 + G3
    z1 = z0 - k1.float32 + G3
    x2 = x0 - i2.float32 + 2.float32 * G3
    y2 = y0 - j2.float32 + 2.float32 * G3
    z2 = z0 - k2.float32 + 2.float32 * G3
    x3 = x0 - 1.float32 + 3.float32 * G3
    y3 = y0 - 1.float32 + 3.float32 * G3
    z3 = z0 - 1.float32 + 3.float32 * G3
    ii = (i.int32 and 255).uint8
    jj = (j.int32 and 255).uint8
    kk = (k.int32 and 255).uint8
    t0 = 0.6.float32 - x0 * x0 - y0 * y0 - z0 * z0
    t1 = 0.6.float32 - x1 * x1 - y1 * y1 - z1 * z1
    t2 = 0.6.float32 - x2 * x2 - y2 * y2 - z2 * z2
    t3 = 0.6.float32 - x3 * x3 - y3 * y3 - z3 * z3

  var n: array[4, float32]
  if t0 >= 0:
    let gi0 = simplex.permMod12[
      ii + simplex.perm[jj + simplex.perm[kk]]
    ]
    n[0] = t0 * t0 * t0 * t0 * dot(grad3[gi0], x0, y0, z0)

  if t1 >= 0:
    let gi1 = simplex.permMod12[
      ii + i1 + simplex.perm[jj + j1 + simplex.perm[kk + k1]]
    ]
    n[1] = t1 * t1 * t1 * t1 * dot(grad3[gi1], x1, y1, z1)

  if t2 >= 0:
    let gi2 = simplex.permMod12[
      ii + i2 + simplex.perm[jj + j2 + simplex.perm[kk + k2]]
    ]
    n[2] = t2 * t2 * t2 * t2 * dot(grad3[gi2], x2, y2, z2)

  if t3 >= 0:
    let gi3 = simplex.permMod12[
      ii + 1 + simplex.perm[jj + 1 + simplex.perm[kk + 1]]
    ]
    n[3] = t3 * t3 * t3 * t3 * dot(grad3[gi3], x3, y3, z3)

  32.float32 * (n[0] + n[1] + n[2] + n[3])

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

when defined(release):
  {.pop.}

when isMainModule:
  import fidget/opengl/perf

  var s = initSimplex(1988)
  # s.octaves = 3
  # s.frequency = 4
  # s.amplitude = 0.2
  # s.lacunarity = 1.5
  # s.gain = 4.3

  # let g = s.grid4(0.float32, 0.float32, 1.float32)

  # echo g[1][2]
  # echo s.value(1, 2)

  # timeIt "normal":
  #   var c: int
  #   var z: float32
  #   for y in -24000000 ..< 24000000:
  #     # z +=
  #     discard s.value(0, y)
  #     inc c
  #   echo c, " ", z

  timeIt "sse2":
    var c: int
    var z: float32
    for y in countup(-24000000, 24000000-1, 4):
      let tmp = cast[array[4, float32]](s.column4(0, y.float32, 1))
      # z = z + tmp[0] + tmp[1] + tmp[2] + tmp[3]
      inc(c, 4)
    echo c, " ", z

  timeIt "avx":
    var c: int
    var z: float32
    for y in countup(-24000000, 24000000-1, 8):
      let tmp = s.column8(0, y.float32, 1)
      # echo sizeof(tmp)
      # z = z + tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
      inc(c, 8)
    echo c, " ", z

  # for x in countup(-9513, -9513+7, 8):
  # let x = -9513
  # let a = cast[array[8, float32]](s.column8(x.float32, 0, 1))
  # echo a
  # for i in 0 ..< 8:
  #   echo s.value(x, i)
    # if a[i] != sv:
    #   debugEcho "not equal @", x + i, " ", a[i], " ", sv

  # import chroma, flippy

  # var s = initSimplex(1988)
  # let img = newImage(256, 256, 3)
  # for x in 0 ..< 256:
  #   for y in 0 ..< 256:
  #     let
  #       v0 = s.value(x.float32 * 0.1.float32, y.float32 * 0.1.float32, 0.float32)
  #       c0 = (((v0 + 1) / 2) * 255).uint8
  #     img.putRgba(x, y, rgba(c0, c0, c0, 255))

  # img.save("noise.png")
