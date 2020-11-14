import math, random

const
  F2 = (0.5 * (sqrt(3.0) - 1)).float32
  G2 = ((3 - sqrt(3.0)) / 6).float32
  F3 = (1.0 / 3.0).float32
  G3 = (1.0 / 6.0).float32

  grad3 = [
    [1.int8, 1, 0], [-1.int8, 1, 0], [1.int8, -1, 0], [-1.int8, -1, 0],
    [1.int8, 0, 1], [-1.int8, 0, 1], [1.int8, 0, -1], [-1.int8, 0, -1],
    [0.int8, 1, 1], [0.int8, -1, 1], [0.int8, 1, -1], [0.int8, -1, -1]
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

template fastFloor(f: float32): int =
  if f >= 0: f.int else: (f - 1).int

func dot(g: array[3, int8], x, y: float32): float32 {.inline.} =
  g[0].float32 * x + g[1].float32 * y

func dot(g: array[3, int8], x, y, z: float32): float32 {.inline.} =
  g[0].float32 * x + g[1].float32 * y + g[2].float32 * z

func noise(simplex: Simplex, x, y: float32): float32 =
  let
    s = (x + y) * F2
    i = fastFloor(x + s)
    j = fastFloor(y + s)
    t = (i + j).float32 * G2
    x0 = x - (i.float32 - t)
    y0 = y - (j.float32 - t)

  var i1, j1: uint8
  if x0 > y0:
    i1 = 1
    j1 = 0
  else:
    i1 = 0
    j1 = 1

  let
    x1 = x0 - i1.float32 + G2
    y1 = y0 - j1.float32 + G2
    x2 = x0 - 1.float32 + 2.float32 * G2
    y2 = y0 - 1.float32 + 2.float32 * G2
    ii = (i and 255).uint8
    jj = (j and 255).uint8
    t0 = 0.5.float32 - x0 * x0 - y0 * y0
    t1 = 0.5.float32 - x1 * x1 - y1 * y1
    t2 = 0.5.float32 - x2 * x2 - y2 * y2

  var n0, n1, n2: float32
  if t0 >= 0:
    n0 = t0 * t0 * t0 * t0 * dot(
      grad3[simplex.permMod12[ii + simplex.perm[jj]]], x0, y0
    )

  if t1 >= 0:
    n1 = t1 * t1 * t1 * t1 * dot(
      grad3[simplex.permMod12[ii + i1 + simplex.perm[jj + j1]]], x1, y1
    )

  if t2 >= 0:
    n2 = t2 * t2 * t2 * t2 * dot(
      grad3[simplex.permMod12[ii + 1 + simplex.perm[jj + 1]]], x2, y2
    )

  70.float32 * (n0 + n1 + n2)

func noise(simplex: Simplex, x, y, z: float32): float32 =
  let
    s = (x + y + z) * F3
    i = fastFloor(x + s)
    j = fastFloor(y + s)
    k = fastFloor(z + s)
    t = (i + j + k).float32 * G3
    x0 = x - (i.float32 - t)
    y0 = y - (j.float32 - t)
    z0 = z - (k.float32 - t)

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
    ii = (i and 255).uint8
    jj = (j and 255).uint8
    kk = (k and 255).uint8
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
    raise newException(NoisyError, "Octaves must be > 0")

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
  import chroma, flippy

  var s = initSimplex(1988)
  let img = newImage(256, 256, 3)
  for x in 0 ..< 256:
    for y in 0 ..< 256:
      let
        v0 = s.value(x.float32 * 0.1.float32, y.float32 * 0.1.float32, 0.float32)
        c0 = (((v0 + 1) / 2) * 255).uint8
      img.putRgba(x, y, rgba(c0, c0, c0, 255))

  img.save("noise.png")
