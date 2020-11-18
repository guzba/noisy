type m128* {.importc: "__m128", header: "xmmintrin.h".} = object

# xmmintrin.h

func mm_loadu_ps*(p: pointer): m128
  {.importc: "_mm_loadu_ps", header: "xmmintrin.h".}

func mm_storeu_ps*(p: pointer, a: m128)
  {.importc: "_mm_storeu_ps", header: "xmmintrin.h".}

func mm_set1_ps*(a: float32): m128
  {.importc: "_mm_set1_ps", header: "xmmintrin.h".}

func mm_and_ps(a, b: m128): m128
  {.importc: "_mm_and_ps", header: "xmmintrin.h".}

func mm_or_ps(a, b: m128): m128
  {.importc: "_mm_or_ps", header: "xmmintrin.h".}

func mm_xor_ps(a, b: m128): m128
  {.importc: "_mm_xor_ps", header: "xmmintrin.h".}

func mm_andnot_ps*(a, b: m128): m128
  {.importc: "_mm_andnot_ps", header: "xmmintrin.h".}

func mm_cmpgt_ps(a, b: m128): m128
  {.importc: "_mm_cmpgt_ps", header: "xmmintrin.h".}

func mm_cmpge_ps(a, b: m128): m128
  {.importc: "_mm_cmpge_ps", header: "xmmintrin.h".}

func mm_cmplt_ps(a, b: m128): m128
  {.importc: "_mm_cmplt_ps", header: "xmmintrin.h".}

func mm_cmple_ps(a, b: m128): m128
  {.importc: "_mm_cmple_ps", header: "xmmintrin.h".}

func mm_add_ps(a: m128, b: m128): m128
  {.importc: "_mm_add_ps", header: "xmmintrin.h".}

func mm_sub_ps(a: m128, b: m128): m128
  {.importc: "_mm_sub_ps", header: "xmmintrin.h".}

func mm_mul_ps(a: m128, b: m128): m128
  {.importc: "_mm_mul_ps", header: "xmmintrin.h".}

func mm_div_ps(a: m128, b: m128): m128
  {.importc: "_mm_div_ps", header: "xmmintrin.h".}

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

# For readability

template `and`*(a, b: m128): m128 =
  mm_and_ps(a, b)

template `or`*(a, b: m128): m128 =
  mm_or_ps(a, b)

template `xor`*(a, b: m128): m128 =
  mm_xor_ps(a, b)

template `>`*(a, b: m128): m128 =
  mm_cmpgt_ps(a, b)

template `>=`*(a, b: m128): m128 =
  mm_cmpge_ps(a, b)

template `<`*(a, b: m128): m128 =
  mm_cmplt_ps(a, b)

template `<=`*(a, b: m128): m128 =
  mm_cmple_ps(a, b)

template `+`*(a, b: m128): m128 =
  mm_add_ps(a, b)

template `-`*(a, b: m128): m128 =
  mm_sub_ps(a, b)

template `*`*(a, b: m128): m128 =
  mm_mul_ps(a, b)

template `/`*(a, b: m128): m128 =
  mm_div_ps(a, b)

template `+=`*(a: var m128, b: m128) =
  a = a + b

template `-=`*(a: var m128, b: m128) =
  a = a - b

template `*=`*(a: var m128, b: m128) =
  a = a * b

template `/=`*(a: var m128, b: m128) =
  a = a / b

func floor*(a: m128): m128 {.inline.} =
  const one = [1.float32, 1, 1, 1]
  let tmp = mm_cvtepi32_ps(mm_cvttps_epi32(a))
  tmp - (mm_cmplt_ps(a, tmp) and cast[m128](one))

template blend*(a, b, mask: m128): m128 =
  ((a xor b) and mask) xor a
