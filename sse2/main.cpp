#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <tmmintrin.h>
#include <smmintrin.h>

#define SIMD

typedef struct {
    uint8_t r, g, b, a;
} __attribute__((packed)) Pixel32;

Pixel32 mix_pixels(Pixel32 a32, Pixel32 b32)
{
    const float a32_alpha = a32.a / 255.0;
    const float b32_alpha = b32.a / 255.0;
    const float r_alpha = b32_alpha + a32_alpha * (1.0f - b32_alpha);

    Pixel32 r = {};

    r.r = (uint8_t) ((b32.r * b32_alpha + a32.r * a32_alpha * (1.0f - b32_alpha)) / r_alpha);
    r.g = (uint8_t) ((b32.g * b32_alpha + a32.g * a32_alpha * (1.0f - b32_alpha)) / r_alpha);
    r.b = (uint8_t) ((b32.b * b32_alpha + a32.b * a32_alpha * (1.0f - b32_alpha)) / r_alpha);
    r.a = (uint8_t) (r_alpha * 255.0);

    return r;
}

const __m128i _swap_mask =
    _mm_set_epi8(7,  6,   5,  4,
                 3,  2,   1,  0,
                 15, 14, 13, 12,
                 11, 10,  9,  8
                 );

const __m128i _aa =
    _mm_set_epi8( 15,15,15,15,
                  11,11,11,11,
                  7,7,7,7,
                  3,3,3,3 );

const __m128i _mask1 = _mm_set_epi16(-1,0,0,0, -1,0,0,0);
const __m128i _mask2 = _mm_set_epi16(0,-1,-1,-1, 0,-1,-1,-1);
const __m128i _v1 = _mm_set1_epi16( 1 );

// https://stackoverflow.com/questions/53643637/simd-for-alpha-blending-how-to-operate-on-every-nth-byte
void mix_pixels_sse(Pixel32 *src, Pixel32 *dst, Pixel32 *c)
{
    __m128i _src = _mm_loadu_si128((__m128i*)src);
    __m128i _src_a = _mm_shuffle_epi8(_src, _aa);

    __m128i _dst = _mm_loadu_si128((__m128i*)dst);
    __m128i _dst_a = _mm_shuffle_epi8(_dst, _aa);
    __m128i _one_minus_src_a = _mm_subs_epu8(_mm_set1_epi8(-1), _src_a);

    ////////////////////
    // __m128i _swapped_src = _mm_shuffle_epi8(_src, _swap_mask);
    // __m128i _extended_swapped_src = _mm_cvtepu8_epi16(_swapped_src);
    ////////////////////
    __m128i _out = {};
    {
        __m128i _s_a = _mm_cvtepu8_epi16( _src_a );
        __m128i _s = _mm_cvtepu8_epi16( _src );
        __m128i _d = _mm_cvtepu8_epi16( _dst );
        __m128i _d_a = _mm_cvtepu8_epi16( _one_minus_src_a );
        _out = _mm_adds_epu16(
            _mm_mullo_epi16(_s, _s_a),
            _mm_mullo_epi16(_d, _d_a));
        _out = _mm_srli_epi16(
            _mm_adds_epu16(
                _mm_adds_epu16( _v1, _out ),
                _mm_srli_epi16( _out, 8 ) ), 8 );
        _out = _mm_or_si128(
            _mm_and_si128(_out,_mask2),
            _mm_and_si128(
                _mm_adds_epu16(
                    _s_a,
                    _mm_cvtepu8_epi16(_dst_a)), _mask1));
    }

    // compute _out2 using high quadword of of the _src and _dst
    //...
    __m128i _out2 = {};
    {
        __m128i _s_a = _mm_cvtepu8_epi16(_mm_shuffle_epi8(_src_a, _swap_mask));
        __m128i _s = _mm_cvtepu8_epi16(_mm_shuffle_epi8(_src, _swap_mask));
        __m128i _d = _mm_cvtepu8_epi16(_mm_shuffle_epi8(_dst, _swap_mask));
        __m128i _d_a = _mm_cvtepu8_epi16(_mm_shuffle_epi8(_one_minus_src_a, _swap_mask));
        _out2 = _mm_adds_epu16(
            _mm_mullo_epi16(_s, _s_a),
            _mm_mullo_epi16(_d, _d_a));
        _out2 = _mm_srli_epi16(
            _mm_adds_epu16(
                _mm_adds_epu16( _v1, _out2 ),
                _mm_srli_epi16( _out2, 8 ) ), 8 );
        _out2 = _mm_or_si128(
            _mm_and_si128(_out2,_mask2),
            _mm_and_si128(
                _mm_adds_epu16(
                    _s_a,
                    _mm_cvtepu8_epi16(_dst_a)), _mask1));
    }

    __m128i _ret = _mm_packus_epi16( _out, _out2 );

    _mm_storeu_si128( (__m128i_u*) c, _ret );
}

int main(int argc, char *argv[])
{
    Pixel32 a[] = {
        {1, 2, 3, 0},
        {5, 6, 7, 255},
        {9, 10, 11, 255},
        {13, 14, 15, 255},

        // {1, 2, 3, 4},
        // {5, 6, 7, 8},
        // {9, 10, 11, 12},
        // {13, 14, 15, 16},
    };
    Pixel32 b[] = {
        {17, 18, 19, 255},
        {21, 22, 23, 255},
        {25, 26, 27, 255},
        {29, 30, 31, 255},

        // {17, 18, 19, 20},
        // {21, 22, 23, 24},
        // {25, 26, 27, 28},
        // {29, 30, 31, 32},
    };
    Pixel32 c[4] = {};

#ifndef SIMD
    for (size_t i = 0; i < 4; ++i) {
        c[i] = mix_pixels(a[i], b[i]);
    }
#else
    mix_pixels_sse(a, b, c);
#endif  // SIMD

    return 0;
}
