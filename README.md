# Noisy

Noisy is a pure Nim implementation of Simplex (Perlin) noise.

Noisy works well using Nim's relatively new `--gc:arc` and `--gc:orc` as well as the default garbage collector. This library also works using both `nim c` and `nim cpp`, in addition to `--cc:vcc` on Windows.

# API: noisy

```nim
import noisy
```

## **type** Simplex


```nim
Simplex = object
 octaves*: int
 amplitude*, frequency*, lacunarity*, gain*: float32
```

## **proc** initSimplex


```nim
proc initSimplex(seed: int): Simplex
```

## **proc** value


```nim
proc value(simplex: Simplex; x, y: float32): float32
```

## **proc** value


```nim
proc value(simplex: Simplex; x, y, z: float32): float32
```

## **template** value


```nim
template value(simplex: Simplex; x, y: int): float32
```

## **template** value


```nim
template value(simplex: Simplex; x, y, z: int): float32
```
