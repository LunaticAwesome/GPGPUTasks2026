#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__device__ __forceinline__ int merge_path(
    const unsigned int* A, int a_count,
    const unsigned int* B, int b_count,
    int diag)
{
    int begin = max(0, diag - b_count);
    int end = min(diag, a_count);

    while (begin < end) {
        int mid = (begin + end) >> 1;
        if (A[mid] <= B[diag - mid - 1]) { // трюк -- смотрим на diag - 1, а не на diag,
            // чтобы понять, куда из diag - 1 он перешел в diag
            begin = mid + 1;
        } else {
            end = mid;
        }
    }
    return begin;
}


__global__ void merge_sort(
    const unsigned int* input_data,
          unsigned int* output_data,
                   int  sorted_k,
                   int  n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= static_cast<unsigned int>(n)) {
        return;
    }

    const unsigned int pair_size = 2 * sorted_k;
    const unsigned int start = (i / pair_size) * pair_size;
    const unsigned int mid = min(start + sorted_k, n);
    const unsigned int end = min(start + 2 * sorted_k, n);
    const unsigned int k = i - start;

    const unsigned int a_count = mid - start;
    const unsigned int b_count = end - mid;

    const unsigned int* A = input_data + start;
    const unsigned int* B = input_data + mid;

    const unsigned int i_pos = merge_path(A, a_count, B, b_count, k); // получает часть пары i + j == k

    const unsigned int j_pos = k - i_pos; // вторая часть пары

    unsigned int v;
    if (i_pos >= a_count) {
        v = B[j_pos];
    } else if (j_pos >= b_count) {
        v = A[i_pos];
    } else {
        v = min(A[i_pos], B[j_pos]); // какой элемент надо вставить на диагонали
    }

    output_data[i] = v;
}

namespace cuda {
void merge_sort(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::merge_sort<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(input_data.cuptr(), output_data.cuptr(), sorted_k, n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
