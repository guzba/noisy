import noisy, strformat

var simplex = initSimplex(1988)
simplex.frequency = 0.1

# Starting at (0, 0) generate a 16x16 grid of 2D noise values.
let values = simplex.grid((0, 0), (16, 16))
for x in 0 ..< 16:
  for y in 0 ..< 16:
    let value = values[x, y]
    echo &"({x},{y}): {value}"
