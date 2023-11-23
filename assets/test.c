#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <immintrin.h>

void print(const __m256i *ax, bool is_binary) {
    int8_t *ptr = (int8_t *)ax;
    for (size_t i = 0; i < 32; ++i) {
        if (is_binary) {
            for (int j = 7; j >= 0; --j) {
                printf("%d", (ptr[i] >> j) & 1);
            }
            printf(" ");
            // printf("%x ", ptr[i]);
        } else {
            printf("%d ", ptr[i]);
        }
        
    }
    printf("\n");
}

// gcc -O3 -mavx2 test.c -o test
// gcc -mavx2 -mfma -ffast-math -fpermissive -DQM_x86 test.c -o test
int main() {
    uint8_t packed_int4_0 = 0b11100011;
    printf("%d\n", (packed_int4_0 & 0x0F));
    printf("%d\n", (packed_int4_0 >> 4));

    signed char w_de_0 = (packed_int4_0 & 0x0F) - 8.0; // weight 0
    signed char w_de_16 = (packed_int4_0 >> 4) - 8.0; 
    printf("%d\n", w_de_0);
    printf("%d\n", w_de_16);

    printf("---------------- Test AVX -----------------\n");
    int8_t weight[32];
    for (size_t i = 0; i < 32; ++i) {
        weight[i] = 0b00010010;
    }
    const __m256i *w_start = (__m256i *)&weight;
    const __m256i lowMask = _mm256_set1_epi8(0x0F);
    __m256i raw_w = _mm256_loadu_si256(w_start);
    __m256i lower_w = _mm256_and_si256(raw_w, lowMask);

    int8_t *ptr = (int8_t *)&lower_w;
    for (size_t i = 0; i < 32; ++i) {
        printf("%x ", ptr[i]);
    }
    printf("\n");

    // 0b0001001000010010 >> 4 = 0b0000 0001 0010 0001
    __m256i higher_w = _mm256_srli_epi16(raw_w, 4);
    higher_w = _mm256_and_si256(higher_w, lowMask);
    ptr = (int8_t *)&higher_w;
    for (size_t i = 0; i < 32; ++i) {
        printf("%x ", ptr[i]);
    }
    printf("\n");

    // convert the range from (0, 15) to (-8, 7)
    const __m256i zero_point = _mm256_set1_epi8(8);
    __m256i w_0, w_128;
    w_0 = _mm256_sub_epi8(lower_w, zero_point);
    w_128 = _mm256_sub_epi8(higher_w, zero_point);
    ptr = (int8_t *)&w_0;
    for (size_t i = 0; i < 32; ++i) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
    ptr = (int8_t *)&w_128;
    for (size_t i = 0; i < 32; ++i) {
        printf("%d ", ptr[i]);
    }
    printf("\n");

    // we need to do signed product, but _mm256_maddubs_epi16 only take unsigned
    int8_t activation[32];
    for (size_t i = 0; i < 32; ++i) {
        activation[i] = 0b00000010;
    }

    const __m256i *a_start = (__m256i *)&activation;

    const __m256i ax = _mm256_sign_epi8(w_0, w_0);
    print(&ax, true);

    // Load activation
    __m256i act = a_start[0];
    // Sign the values of the y vectors
    // if w_0 < 0, activation < 0, result > 0
    // if w_0 < 0, activation > 0, result < 0
    const __m256i sy = _mm256_sign_epi8(act, w_0); 
    print(&sy, true);

    __m256i dot = _mm256_maddubs_epi16(ax, sy);
    ptr = (int8_t *)&dot;
    for (size_t i = 0; i < 32; ++i) {
        printf("%d ", ptr[i]);
    }
    printf("\n");
    
    return 0;
}
