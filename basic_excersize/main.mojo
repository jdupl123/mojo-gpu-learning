from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer
from gpu.memory import AddressSpace
from memory import stack_allocation
from layout import Layout, LayoutTensor
from math import iota
from sys import sizeof

def main():
    var ctx = DeviceContext()
    alias dtype = DType.float32
    alias blocks = 8
    alias threads = 4
    alias elements_in = blocks * threads # one element per thread

    var in_buffer = ctx.enqueue_create_buffer[dtype](elements_in)

    with in_buffer.map_to_host() as host_buffer:
        iota(host_buffer.unsafe_ptr(), elements_in)
        print(host_buffer)

    alias layout = Layout.row_major(blocks, threads)

    var in_tensor = LayoutTensor[dtype, layout](in_buffer)
    alias InTensor = LayoutTensor[dtype, layout, MutableAnyOrigin]

    var out_buffer = ctx.enqueue_create_buffer[dtype](blocks)

    # Zero the values on the device as they'll be used to accumulate results
    ctx.enqueue_memset(out_buffer, 0)

    alias out_layout = Layout.row_major(elements_in)
    alias OutTensor = LayoutTensor[dtype, out_layout, MutableAnyOrigin]

    var out_tensor = OutTensor(out_buffer)

    fn print_values_kernel(out_tensor: OutTensor):
        var bid = block_idx.x
        var tid = thread_idx.x
        print("block:", bid, "thread:", tid, "val:", out_tensor[bid, tid])

    ctx.enqueue_function[print_values_kernel](
        out_tensor, grid_dim=blocks, block_dim=threads,
    )
    ctx.synchronize()

    fn sum_reduce_kernel(
        in_tensor: InTensor, out_tensor: OutTensor
    ):
        # This allocates memory to be shared between threads in a block prior to the
        # kernel launching. Each kernel gets a pointer to the allocated memory.
        var shared = stack_allocation[
            threads,
            Scalar[dtype],
            address_space = AddressSpace.SHARED,
        ]()

        # Place the corresponding value into shared memory
        shared[thread_idx.x] = in_tensor[block_idx.x, thread_idx.x][0]

        # Await all the threads to finish loading their values into shared memory
        barrier()

        # If this is the first thread, sum and write the result to global memory
        if thread_idx.x == 0:
            for i in range(threads):
                out_tensor[block_idx.x] += shared[i]

    ctx.enqueue_function[sum_reduce_kernel](
        in_tensor,
        out_tensor,
        grid_dim=blocks,
        block_dim=threads,
    )

    # Copy the data back to the host and print out the buffer
    with out_buffer.map_to_host() as host_buffer:
        print(host_buffer)
        
    ctx.enqueue_function[print_values_kernel](
        out_tensor, grid_dim=blocks, block_dim=threads,
    )
    ctx.synchronize()