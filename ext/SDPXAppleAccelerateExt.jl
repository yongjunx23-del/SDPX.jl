module SDPXAppleAccelerateExt

using AppleAccelerate
using SDPX

function __init__()
    SDPX._register_blas_thread_controller!(
        () -> Int(AppleAccelerate.get_num_threads()),
        count -> Int(AppleAccelerate.set_num_threads(count)),
        :apple_accelerate,
    )
end

end
