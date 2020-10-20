# Noisy

`nimble install noisy`

Noisy is a pure Nim implementation of Simplex (Perlin) noise.

Noisy works well using Nim's relatively new `--gc:arc` and `--gc:orc` as well as the default garbage collector. This library also works using both `nim c` and `nim cpp`, in addition to `--cc:vcc` on Windows.

I have also verified that Noisy builds with `--experimental:strictFuncs` on Nim 1.4.0.

# Example

```nim
import noisy

let simplex = initSimplex(1988)

for y in 0 ..< 10:
    for x in 0 .. 10:
        let value = simplex.value(x, y)
        echo value
```

# API: noisy

```nim
import noisy
```

## **type** Simplex


```nim
Simplex = object
 octaves*: int
 amplitude*, frequency*, lacunarity*, gain*: float32
 perm*: array[256, uint8]
 permMod12*: array[256, uint8]
```

## **type** NoisyError


```nim
NoisyError = object of ValueError
```

## **func** initSimplex


```nim
func initSimplex(seed: int): Simplex
```

## **func** value

Generates the 2D noise value at (x, y) based on the Simplex parameters.

```nim
func value(simplex: Simplex; x, y: float32): float32 {.raises: [NoisyError], tags: [].}
```

## **func** value

Generates the 3D noise value at (x, y, z) based on the Simplex parameters.

```nim
func value(simplex: Simplex; x, y, z: float32): float32 {.raises: [NoisyError], tags: [].}
```

## **template** value


```nim
template value(simplex: Simplex; x, y: int): float32
```

## **template** value


```nim
template value(simplex: Simplex; x, y, z: int): float32
```
