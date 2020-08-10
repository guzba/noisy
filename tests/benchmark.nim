import fidget/opengl/perf, noisy, perlin

timeIt "noisy":
  let simplex = initSimplex(1988)

  for x in 0 ..< 750:
    for y in 0 ..< 750:
      for z in 0 ..< 500:
        assert simplex.value(x, y) != 2

timeIt "PerlinNim":
  let noise = newNoise(1988, 1, 1.0)
  for x in 0 ..< 750:
    for y in 0 ..< 750:
      for z in 0 ..< 500:
        assert noise.pureSimplex(x, y) != 2
