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
  for x in 0 ..< 524288:
    for y in 0 ..< 4:
      c += simplex.value(x, y)
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  524288x4: {delta:.4f}s [{c}]"

block simd_2d:
  let start = getMonoTime().ticks
  var c: float
  for x in countup(0, 524288 - 1, 4):
    let grid = simplex.grid4(x, 0)
    for i in 0 ..< 4:
      for j in 0 ..< 4:
        c += grid[i][j]
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  524288x4: {delta:.4f}s [{c}]"

block basic_3d:
  let start = getMonoTime().ticks
  var c: float
  for x in 0 ..< 131072:
    for y in 0 ..< 4:
      for z in 0 ..< 4:
        c += simplex.value(x, y, z)
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  131072x4x4: {delta:.4f}s [{c}]"

block simd_3d:
  let start = getMonoTime().ticks
  var c: float
  for x in countup(0, 131072 - 1, 4):
    let grid = simplex.grid4(x, 0, 0)
    for i in 0 ..< 4:
      for j in 0 ..< 4:
        for k in 0 ..< 4:
          c += grid[i][j][k]
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  131072x4x4: {delta:.4f}s [{c}]"
