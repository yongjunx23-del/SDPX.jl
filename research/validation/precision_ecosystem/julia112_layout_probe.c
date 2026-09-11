/* Independent installed-header check. Not an allocator or C-scratch bound.
 * cc -I<Julia runtime>/include/julia julia112_layout_probe.c -o <temporary output>
 */
#include <stddef.h>
#include <stdio.h>
#include "julia.h"

int main(void)
{
    printf("tag=%zu memory_body=%zu memory_ref=%zu array_fixed=%zu "
           "memory_length_offset=%zu memory_pointer_offset=%zu array_dims_offset=%zu\n",
           sizeof(jl_taggedvalue_t), sizeof(jl_genericmemory_t),
           sizeof(jl_genericmemoryref_t), sizeof(jl_array_t),
           offsetof(jl_genericmemory_t, length), offsetof(jl_genericmemory_t, ptr),
           offsetof(jl_array_t, dimsize));
    return !(sizeof(jl_taggedvalue_t) == 8 && sizeof(jl_genericmemory_t) == 16 &&
             sizeof(jl_genericmemoryref_t) == 16 && sizeof(jl_array_t) == 16 &&
             offsetof(jl_genericmemory_t, length) == 0 &&
             offsetof(jl_genericmemory_t, ptr) == 8 && offsetof(jl_array_t, dimsize) == 16);
}
