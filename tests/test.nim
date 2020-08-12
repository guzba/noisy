import noisy

let simplex = initSimplex(1988)

for y in 0 ..< 10:
    for x in 0 .. 10:
        let value = simplex.value(x, y)
        assert value >= -1.0 and value <= 1.0
