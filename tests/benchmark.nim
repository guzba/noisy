import benchy, noisy, strformat

var simplex = initSimplex(1988) # 1988 is the random seed, generate this however
simplex.octaves = 3
simplex.frequency = 4
simplex.amplitude = 0.2
simplex.lacunarity = 1.5
simplex.gain = 4.3

timeIt "2d 4096x4096":
  var c: float
  for x in 0 ..< 4096:
    for y in 0 ..< 4096:
      c += simplex.value(x, y)
  keep(c)

timeIt "2d simd 4096x4096":
  var c: float
  let g = simplex.grid((0, 0), (4096, 4096))
  for value in g.values:
    c += value
  keep(c)

timeIt "3d 256x256x256":
  var c: float
  for x in 0 ..< 256:
    for y in 0 ..< 256:
      for z in 0 ..< 256:
        c += simplex.value(x, y, z)
  keep(c)

timeIt "3d simd 256x256x256":
  var c: float
  let g = simplex.grid((0, 0, 0), (256, 256, 256))
  for value in g.values:
    c += value
  keep(c)
