import noisy

var simplex = initSimplex(1988) # 1988 is the random seed, generate this however
simplex.octaves = 3
simplex.frequency = 4
simplex.amplitude = 0.2
simplex.lacunarity = 1.5
simplex.gain = 4.3

block basic_2d:
  let grid = simplex.grid4(0, 0)
  for x in 0 ..< 4:
    for y in 0 ..< 4:
      let value = simplex.value(x, y)
      doAssert value >= -1.0 and value <= 1.0
      doAssert grid[x][y] == value


block basic_3d:
  let grid = simplex.grid4(0, 0, 0)
  for x in 0 ..< 4:
    for y in 0 ..< 4:
      for z in 0 ..< 4:
        let value = simplex.value(x, y, z)
        doAssert value >= -1.0 and value <= 1.0
        doAssert grid[x][y][z] == value

block range_2d:
  var total: float32
  for offset in countup(-1200, 1200-1, 4):
    let simd = simplex.grid4(offset.float32, offset.float32)
    for x in 0 ..< 4:
      for y in 0 ..< 4:
        let n = simplex.value(x + offset, y + offset)
        total += n
        assert simd[x][y] == n
  echo total

block range_3d:
  var total: float32
  for offset in countup(-1200, 1200-1, 4):
    let simd = simplex.grid4(offset.float32, offset.float32, offset.float32)
    for x in 0 ..< 4:
      for y in 0 ..< 4:
        for z in 0 ..< 4:
          let n = simplex.value(x + offset, y + offset, z + offset)
          total += n
          assert simd[x][y][z] == n
  echo total
