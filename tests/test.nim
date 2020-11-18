import noisy

let simplex = initSimplex(1988) # 1988 is the random seed, generate this however

block basic_2d:
  let grid = simplex.grid(0, 0, 19, 17)
  for x in 0 ..< 19:
    for y in 0 ..< 17:
      let value = simplex.value(x, y)
      doAssert value >= -1.0 and value <= 1.0
      if grid[x, y] != value:
        echo x, " ", y, " ", grid[x, y], " ", value
      # doAssert grid[x, y] == value

block basic_3d:
  let grid = simplex.grid(0, 0, 0, 19, 19, 17)
  for x in 0 ..< 19:
    for y in 0 ..< 19:
      for z in 0 ..< 17:
        let value = simplex.value(x, y, z)
        doAssert value >= -1.0 and value <= 1.0
        doAssert grid[x, y, z] == value

block range_2d:
  var total: float32
  for offset in countup(-1200, 1200-1, 4):
    let simd = simplex.grid(offset, offset, 4, 4)
    for x in 0 ..< 4:
      for y in 0 ..< 4:
        let n = simplex.value(x + offset, y + offset)
        total += n
        assert simd[x, y] == n
  echo total

block range_3d:
  var total: float32
  for offset in countup(-1200, 1200-1, 4):
    let simd = simplex.grid(offset, offset, offset, 4, 4, 4)
    for x in 0 ..< 4:
      for y in 0 ..< 4:
        for z in 0 ..< 4:
          let n = simplex.value(x + offset, y + offset, z + offset)
          total += n
          assert simd[x, y, z] == n
  echo total
