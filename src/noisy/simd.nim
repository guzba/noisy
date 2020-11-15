# when defined(vcc):
#   discard
# else:
# {.passC: "-mavx2".}
# {.passL: "-mavx2".}

{.passC: "-msse4.2".}
{.passL: "-msse4.2".}

type
  m128* {.importc: "__m128", header: "xmmintrin.h".} = object
  m128i* {.importc: "__m128i", header: "emmintrin.h".} = object
  # m256i* {.importc: "__m256i", header: "immintrin.h".} = object

func mm_set1_ps*(a: float32): m128
  {.importc: "_mm_set1_ps", header: "xmmintrin.h".}

func mm_and_si128*(a, b: m128i): m128i
  {.importc: "_mm_and_si128", header: "emmintrin.h".}

func `and`*(a, b: m128i): m128i {.inline.} =
  mm_and_si128(a, b)

func mm_set1_epi32*(a: int32): m128i
  {.importc: "_mm_set1_epi32", header: "emmintrin.h".}

func mm_cmpgt_ps*(a, b: m128): m128
  {.importc: "_mm_cmpgt_ps", header: "xmmintrin.h".}

# func mm_movemask_ps*(a: m128): int32
#   {.importc: "_mm_movemask_ps", header: "xmmintrin.h".}

func mm_mul_ps(a: m128, b: m128): m128
  {.importc: "_mm_mul_ps", header: "xmmintrin.h".}

func mm_add_ps(a: m128, b: m128): m128
  {.importc: "_mm_add_ps", header: "xmmintrin.h".}

# func mm_add_epi32(a: m128i, b: m128i): m128i
#   {.importc: "_mm_add_epi32", header: "emmintrin.h".}

func mm_cvtps_epi32*(a: m128): m128i
  {.importc: "_mm_cvtps_epi32", header: "emmintrin.h".}

func mm_sub_ps(a: m128, b: m128): m128
  {.importc: "_mm_sub_ps", header: "xmmintrin.h".}

# func mm_sub_epi32(a: m128i, b: m128i): m128i
#   {.importc: "_mm_sub_epi32", header: "emmintrin.h".}

func `+`*(a, b: m128): m128 {.inline.} =
  mm_add_ps(a, b)

# func `+`*(a, b: m128i): m128i =
#   mm_add_epi32(a, b)

# func `-`*(a, b: m128i): m128i =
#   mm_sub_epi32(a, b)

func `-`*(a, b: m128): m128 {.inline.} =
  mm_sub_ps(a, b)

func `*`*(a, b: m128): m128 {.inline.} =
  mm_mul_ps(a, b)

# func mm_loadu_ps*(p: pointer): m128
#   {.importc: "_mm_loadu_ps", header: "xmmintrin.h".}

# func mm_loadu_si128*(dst: pointer): m128i
#   {.importc: "_mm_loadu_si128", header: "emmintrin.h".}

# func mm_storeu_si128*(dst: pointer, v: m128i)
#   {.importc: "_mm_storeu_si128", header: "emmintrin.h".}

# proc mm_cmpeq_epi8*(a, b: m128i): m128i
#   {.importc: "_mm_cmpeq_epi8", header: "emmintrin.h".}

# proc mm_movemask_epi8*(a: m128i): int32
#   {.importc: "_mm_movemask_epi8", header: "emmintrin.h".}

# proc mm256_loadu_si256*(p: pointer): m256i
#   {.importc: "_mm256_loadu_si256", header: "immintrin.h".}

# proc mm256_cmpeq_epi8*(a: m256i, b: m256i): m256i
#   {.importc: "_mm256_cmpeq_epi8", header: "immintrin.h".}

# proc mm256_movemask_epi8*(a: m256i): int32
#   {.importc: "_mm256_movemask_epi8", header: "immintrin.h".}
# when defined(amd64):
#   import simd

#   template copy128*(dst: var seq[uint8], src: openarray[uint8], op, ip: int) =
#     when nimvm:
#       copy64(dst, src, op + 0, ip + 0)
#       copy64(dst, src, op + 8, ip + 8)
#     else:
#       mm_storeu_si128(dst[op].addr, mm_loadu_si128(src[ip].unsafeAddr))
# else:
#   template copy128*(dst: var seq[uint8], src: openarray[uint8], op, ip: int) =
#     copy64(dst, src, op + 0, ip + 0)
#     copy64(dst, src, op + 8, ip + 8)

func mm_floor_ps(a: m128): m128
  {.importc: "_mm_floor_ps", header: "smmintrin.h".}

func floor*(a: m128): m128 {.inline.} =
  mm_floor_ps(a)

func mm_blendv_ps*(a, b, mask: m128): m128
  {.importc: "_mm_blendv_ps", header: "smmintrin.h".}
