import noisy, perlin, fidget/opengl/perf

var s = 0.float32
timeIt "noisy":
  let simplex = initSimplex(1988)

  for x in 0 ..< 500:
    for y in 0 ..< 500:
      for z in 0 ..< 500:
        s = s + simplex.value(x, y, z)

if s > 0:
  echo "positive"
else:
  echo "negative"

# timeIt "PerlinNim":
#   let noise = newNoise(1988, 1, 1.0)
#   for x in 0 ..< 500:
#     for y in 0 ..< 500:
#       for z in 0 ..< 500:
#         assert noise.pureSimplex(x, y, z) != 2
