import noisy, std/monotimes, strformat

var simplex = initSimplex(1988) # 1988 is the random seed, generate this however
simplex.octaves = 3
simplex.frequency = 4
simplex.amplitude = 0.2
simplex.lacunarity = 1.5
simplex.gain = 4.3

block basic_2d:
  let start = getMonoTime().ticks
  var c: float
  for x in 0 ..< 4096:
    for y in 0 ..< 4096:
      c += simplex.value(x, y)
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  4096x4096: {delta:.4f}s [{c}]"

block simd_2d:
  let start = getMonoTime().ticks
  var c: float
  let g = simplex.grid(0, 0, 4096, 4096)
  for value in g.values:
    c += value
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  4096x4096: {delta:.4f}s [{c}]"

block basic_3d:
  let start = getMonoTime().ticks
  var c: float
  for x in 0 ..< 256:
    for y in 0 ..< 256:
      for z in 0 ..< 256:
        c += simplex.value(x, y, z)
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  256x256x256: {delta:.4f}s [{c}]"

block simd_3d:
  let start = getMonoTime().ticks
  var c: float
  let g = simplex.grid(0, 0, 0, 256, 256, 256)
  for value in g.values:
    c += value
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  256x256x256: {delta:.4f}s [{c}]"
