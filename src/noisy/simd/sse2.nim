type m128* {.importc: "__m128", header: "xmmintrin.h".} = object

# xmmintrin.h

func mm_set1_ps*(a: float32): m128
  {.importc: "_mm_set1_ps", header: "xmmintrin.h".}

func mm_and_ps*(a, b: m128): m128
  {.importc: "_mm_and_ps", header: "xmmintrin.h".}

func mm_or_ps*(a, b: m128): m128
  {.importc: "_mm_or_ps", header: "xmmintrin.h".}

func mm_andnot_ps*(a, b: m128): m128
  {.importc: "_mm_andnot_ps", header: "xmmintrin.h".}

func mm_cmpgt_ps*(a, b: m128): m128
  {.importc: "_mm_cmpgt_ps", header: "xmmintrin.h".}

func mm_cmplt_ps*(a, b: m128): m128
  {.importc: "_mm_cmplt_ps", header: "xmmintrin.h".}

func mm_add_ps(a: m128, b: m128): m128
  {.importc: "_mm_add_ps", header: "xmmintrin.h".}

func mm_sub_ps(a: m128, b: m128): m128
  {.importc: "_mm_sub_ps", header: "xmmintrin.h".}

func mm_mul_ps(a: m128, b: m128): m128
  {.importc: "_mm_mul_ps", header: "xmmintrin.h".}

# emmintrin.h

type m128i* {.importc: "__m128i", header: "emmintrin.h".} = object

func mm_set1_epi32*(a: int32): m128i
  {.importc: "_mm_set1_epi32", header: "emmintrin.h".}

func mm_cvttps_epi32*(a: m128): m128i
  {.importc: "_mm_cvttps_epi32", header: "emmintrin.h".}

func mm_cvtps_epi32*(a: m128): m128i
  {.importc: "_mm_cvtps_epi32", header: "emmintrin.h".}

func mm_cvtepi32_ps*(a: m128i): m128
  {.importc: "_mm_cvtepi32_ps", header: "emmintrin.h".}

func mm_and_si128*(a, b: m128i): m128i
  {.importc: "_mm_and_si128", header: "emmintrin.h".}

# Func for readability

func `and`*(a, b: m128): m128 {.inline} =
  mm_and_ps(a, b)

func `and`*(a, b: m128i): m128i {.inline.} =
  mm_and_si128(a, b)

func `or`*(a, b: m128): m128 {.inline} =
  mm_or_ps(a, b)

func `>`*(a, b: m128): m128 =
  mm_cmpgt_ps(a, b)

func `<`*(a, b: m128): m128 =
  mm_cmplt_ps(a, b)

func `+`*(a, b: m128): m128 {.inline.} =
  mm_add_ps(a, b)

func `-`*(a, b: m128): m128 {.inline.} =
  mm_sub_ps(a, b)

func `*`*(a, b: m128): m128 {.inline.} =
  mm_mul_ps(a, b)

func floor*(a: m128): m128 {.inline.} =
  const one = [1.float32, 1, 1, 1]
  let tmp =  mm_cvtepi32_ps(mm_cvttps_epi32(a))
  tmp - (mm_cmplt_ps(a, tmp) and cast[m128](one))

func blend*(a, b, mask: m128): m128 {.inline.} =
  mm_andnot_ps(mask, a) or (mask and b)
