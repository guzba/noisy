import noisy, perlin, std/monotimes, strformat

# These are both running 2 octaves to compare properly. PerlinNim has an issue
# where it cannot do just 1 octave: https://github.com/Nycto/PerlinNim/issues/8

block guzba_noisy:
  var simplex = initSimplex(1988)
  simplex.octaves = 2
  simplex.gain = 1.0
  let start = getMonoTime().ticks
  var c: float
  for x in 0 ..< 300:
    for y in 0 ..< 300:
      for z in 0 ..< 300:
        c += (simplex.value(x, y, z) * 0.5 + 0.5)
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  300x300x300: {delta:.4f}s [{c}]"

block nycto_perlinnim:
  let
    noise = newNoise(1988)
    start = getMonoTime().ticks
  var c: float
  for x in 0 ..< 300:
    for y in 0 ..< 300:
      for z in 0 ..< 300:
        c += noise.pureSimplex(x, y, z)
  let delta = float64(getMonoTime().ticks - start) / 1000000000.0
  echo &"  300x300x300: {delta:.4f}s [{c}]"
