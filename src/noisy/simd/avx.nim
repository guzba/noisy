{.passC: "-mavx2".}
{.passL: "-mavx2".}

type m256* {.importc: "__m256", header: "immintrin.h".} = object

const
  CMP_EQ_OQ* = 0x00 # Equal (ordered, non-signaling)
  CMP_LT_OS* = 0x01 # Less-than (ordered, signaling)
  CMP_LE_OS* = 0x02 # Less-than-or-equal (ordered, signaling)
  CMP_UNORD_Q* = 0x03 # Unordered (non-signaling)
  CMP_NEQ_UQ* = 0x04 # Not-equal (unordered, non-signaling)
  CMP_NLT_US* = 0x05 # Not-less-than (unordered, signaling)
  CMP_NLE_US* = 0x06 # Not-less-than-or-equal (unordered, signaling)
  CMP_ORD_Q* = 0x07 # Ordered (nonsignaling)
  CMP_EQ_UQ* = 0x08 # Equal (unordered, non-signaling)
  CMP_NGE_US* = 0x09 # Not-greater-than-or-equal (unord, signaling)
  CMP_NGT_US* = 0x0a # Not-greater-than (unordered, signaling)
  CMP_FALSE_OQ* = 0x0b # False (ordered, non-signaling)
  CMP_NEQ_OQ* = 0x0c # Not-equal (ordered, non-signaling)
  CMP_GE_OS* = 0x0d # Greater-than-or-equal (ordered, signaling)
  CMP_GT_OS* = 0x0e # Greater-than (ordered, signaling)
  CMP_TRUE_UQ* = 0x0f # True (unordered, non-signaling)
  CMP_EQ_OS* = 0x10 # Equal (ordered, signaling)
  CMP_LT_OQ* = 0x11 # Less-than (ordered, non-signaling)
  CMP_LE_OQ* = 0x12 # Less-than-or-equal (ordered, non-signaling)
  CMP_UNORD_S* = 0x13 # Unordered (signaling)
  CMP_NEQ_US* = 0x14 # Not-equal (unordered, signaling)
  CMP_NLT_UQ* = 0x15 # Not-less-than (unordered, non-signaling)
  CMP_NLE_UQ* = 0x16 # Not-less-than-or-equal (unord, non-signaling)
  CMP_ORD_S* = 0x17 # Ordered (signaling)
  CMP_EQ_US* = 0x18 # Equal (unordered, signaling)
  CMP_NGE_UQ* = 0x19 # Not-greater-than-or-equal (unord, non-sign)
  CMP_NGT_UQ* = 0x1a # Not-greater-than (unordered, non-signaling)
  CMP_FALSE_OS* = 0x1b # False (ordered, signaling)
  CMP_NEQ_OS* = 0x1c # Not-equal (ordered, signaling)
  CMP_GE_OQ* = 0x1d # Greater-than-or-equal (ordered, non-signaling)
  CMP_GT_OQ* = 0x1e # Greater-than (ordered, non-signaling)
  CMP_TRUE_US* = 0x1f # True (unordered, signaling)

func mm256_loadu_ps*(a: pointer): m256
  {.importc: "_mm256_loadu_ps", header: "xmmintrin.h".}

func mm256_set1_ps*(a: float32): m256
  {.importc: "_mm256_set1_ps", header: "immintrin.h".}

func mm256_and_ps(a, b: m256): m256
  {.importc: "_mm256_and_ps", header: "immintrin.h".}

func mm256_or_ps(a, b: m256): m256
  {.importc: "_mm256_or_ps", header: "immintrin.h".}

func mm256_xor_ps(a, b: m256): m256
  {.importc: "_mm256_xor_ps", header: "immintrin.h".}

func mm256_cmp_ps(a, b: m256, imm8: int32): m256
  {.importc: "_mm256_cmp_ps", header: "immintrin.h".}

func mm256_add_ps(a: m256, b: m256): m256
  {.importc: "_mm256_add_ps", header: "immintrin.h".}

func mm256_sub_ps(a: m256, b: m256): m256
  {.importc: "_mm256_sub_ps", header: "immintrin.h".}

func mm256_mul_ps(a: m256, b: m256): m256
  {.importc: "_mm256_mul_ps", header: "immintrin.h".}

func mm256_div_ps(a: m256, b: m256): m256
  {.importc: "_mm256_div_ps", header: "immintrin.h".}

func mm256_floor_ps*(a: m256): m256
  {.importc: "_mm256_floor_ps", header: "immintrin.h".}

func mm256_blendv_ps*(a, b, mask: m256): m256
  {.importc: "_mm256_blendv_ps", header: "immintrin.h".}

type m256i* {.importc: "__m256i", header: "immintrin.h".} = object

func mm256_set1_epi32*(a: int32): m256i
  {.importc: "_mm256_set1_epi32", header: "immintrin.h".}

func mm256_cvttps_epi32*(a: m256): m256i
  {.importc: "_mm256_cvttps_epi32", header: "immintrin.h".}

func mm256_cvtps_epi32*(a: m256): m256i
  {.importc: "_mm256_cvtps_epi32", header: "immintrin.h".}

func mm256_cvtepi32_ps*(a: m256i): m256
  {.importc: "_mm256_cvtepi32_ps", header: "immintrin.h".}

# For readability

template `and`*(a, b: m256): m256 =
  mm256_and_ps(a, b)

template `or`*(a, b: m256): m256 =
  mm256_or_ps(a, b)

template `xor`*(a, b: m256): m256 =
  mm256_xor_ps(a, b)

template `>`*(a, b: m256): m256 =
  mm256_cmp_ps(a, b, CMP_GT_OS)

template `<`*(a, b: m256): m256 =
  mm256_cmp_ps(a, b, CMP_LE_OQ)

template `+`*(a, b: m256): m256 =
  mm256_add_ps(a, b)

template `-`*(a, b: m256): m256 =
  mm256_sub_ps(a, b)

template `*`*(a, b: m256): m256 =
  mm256_mul_ps(a, b)

template `/`*(a, b: m256): m256 =
  mm256_div_ps(a, b)

template floor*(a: m256): m256 =
  mm256_floor_ps(a)

template blend*(a, b, mask: m256): m256 =
  mm256_blendv_ps(a, b, mask)



func mm256_and_si256(a, b: m256i): m256i
  {.importc: "_mm256_and_si256", header: "immintrin.h".}

template `and`*(a, b: m256i): m256i =
  mm256_and_si256(a, b)
