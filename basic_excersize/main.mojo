from gpu import thread_idx, block_idx, warp, barrier
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
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
        print("block:", bid, "thread:", tid, "val:", in_tensor[bid, tid])

    ctx.enqueue_function[print_values_kernel](
        out_tensor, grid_dim=blocks, block_dim=threads,
    )
    ctx.synchronize()