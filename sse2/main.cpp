#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>

#include <algorithm>

#include <tmmintrin.h>
#include <smmintrin.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#include "./stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb_image_write.h"

#define SIMD

struct Pixel32
{
    uint8_t r, g, b, a;
};

struct Image32
{
    Pixel32 *pixels;
    size_t width;
    size_t height;
};

const size_t SIMD_PIXEL_PACK_SIZE = sizeof(__m128i) / sizeof(Pixel32);

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

// NOTE: Stolen from https://stackoverflow.com/a/53707227
void mix_pixels_sse(Pixel32 *src, Pixel32 *dst, Pixel32 *c)
{
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

    __m128i _src = _mm_loadu_si128((__m128i*)src);
    __m128i _src_a = _mm_shuffle_epi8(_src, _aa);

    __m128i _dst = _mm_loadu_si128((__m128i*)dst);
    __m128i _dst_a = _mm_shuffle_epi8(_dst, _aa);
    __m128i _one_minus_src_a = _mm_subs_epu8(_mm_set1_epi8(-1), _src_a);

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

void slap_image32_onto_image32(Image32 src, Image32 dst,
                              size_t x0, size_t y0)
{
    size_t x1 = std::min(x0 + src.width, dst.width);
    size_t y1 = std::min(y0 + src.height, dst.height);
    for (size_t y = y0; y < y1; ++y) {
        for (size_t x = x0; x < x1; ++x) {
            dst.pixels[y * dst.width + x] =
                mix_pixels(
                    dst.pixels[y * dst.width + x],
                    src.pixels[(y - y0) * src.width + (x - x0)]);
        }
    }
}

void slap_image32_onto_image32_simd(Image32 src, Image32 dst,
                               size_t x0, size_t y0)
{
    Pixel32 out[SIMD_PIXEL_PACK_SIZE] = {};

    size_t x1 = std::min(x0 + src.width, dst.width);
    size_t y1 = std::min(y0 + src.height, dst.height);
    for (size_t y = y0; y < y1; ++y) {
        for (size_t x = x0; x < x1; x += SIMD_PIXEL_PACK_SIZE) {
            mix_pixels_sse(
                &src.pixels[(y - y0) * src.width + (x - x0)],
                &dst.pixels[y * dst.width + x],
                &dst.pixels[y * dst.width + x]);
            // TODO: tail of the row is not taken into account
        }
    }
}

Image32 load_image32(const char *filepath)
{
    Image32 result = {};
    int x, y, n;
    result.pixels = (Pixel32*) stbi_load(filepath, &x, &y, &n, 4);
    result.width = x;
    result.height = y;
    return result;
}

int main(int argc, char *argv[])
{
    static_assert(sizeof(Pixel32) == sizeof(uint32_t),
                  "Size of Pixel32 is scuffed on your platform lol");

#ifdef SIMD
    printf("SIMD ON\n");
#else
    printf("SIMD OFF\n");
#endif

    const char * const DST_FILENAME = "maxresdefault.jpg";
    Image32 dst = load_image32(DST_FILENAME);

    const char * const SRC_FILENAME = "tsodinFeels.png";
    Image32 src = load_image32(SRC_FILENAME);

    for (size_t i = 0; i < src.width * src.height; ++i) {
        src.pixels[i].a = src.pixels[i].a >> 1;
    }

    size_t pos_x = (dst.width >> 1) - (src.width >> 1);
    size_t pos_y = (dst.height >> 1) - (src.height >> 1);
    const size_t N = 100'000;
    clock_t begin = clock();
    for (size_t i = 0; i < N; ++i) {
#ifdef SIMD
        slap_image32_onto_image32_simd(src, dst, pos_x, pos_y);
#else
        slap_image32_onto_image32(src, dst, pos_x, pos_y);
#endif
    }
    printf("    %fs\n", (float)(clock() - begin) / (float) CLOCKS_PER_SEC);

    const char * const OUT_FILENAME = "output.png";
    int ret = stbi_write_png(OUT_FILENAME, dst.width, dst.height, 4, dst.pixels, dst.width * 4);
    printf("    ret = %d\n", ret);
    return 0;
}
