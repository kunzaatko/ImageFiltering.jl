using ImageFiltering, FFTW, LinearAlgebra, Profile, Random
# using ProfileView
using ComputationalResources

FFTW.set_num_threads(parse(Int, get(ENV, "FFTW_NUM_THREADS", "1")))
BLAS.set_num_threads(parse(Int, get(ENV, "BLAS_NUM_THREADS", string(Threads.nthreads() ÷ 2))))

function benchmark(mats)
    kernel = ImageFiltering.factorkernel(Kernel.LoG(1))
    Threads.@threads for mat in mats
        frame_filtered = deepcopy(mat[:, :, 1])
        r_cached = CPU1(ImageFiltering.planned_fft(frame_filtered, kernel))
        for i in axes(mat, 3)
            frame = @view mat[:, :, i]
            imfilter!(r_cached, frame_filtered, frame, kernel)
        end
        return
    end
end

function test(mats)
    kernel = ImageFiltering.factorkernel(Kernel.LoG(1))
    for mat in mats
        f1 = deepcopy(mat[:, :, 1])
        r_cached = CPU1(ImageFiltering.planned_fft(f1, kernel))
        f2 = deepcopy(mat[:, :, 1])
        r_noncached = CPU1(Algorithm.FFT())
        for i in axes(mat, 3)
            frame = @view mat[:, :, i]
            @info "imfilter! noncached"
            imfilter!(r_noncached, f2, frame, kernel)
            @info "imfilter! cached"
            imfilter!(r_cached, f1, frame, kernel)
            @show f1[1:4] f2[1:4]
            f1 ≈ f2 || error("f1 !≈ f2")
        end
        return
    end
end

function profile()
    Random.seed!(1)
    nmats = 10
    mats = [rand(Float32, rand(80:100), rand(80:100), rand(2000:3000)) for _ in 1:nmats]
    GC.gc(true)

    # benchmark(mats)

    # for _ in 1:3
    #     @time "warm run of benchmark(mats)" benchmark(mats)
    # end

    test(mats)

    # Profile.clear()
    # @profile benchmark(mats)

    # Profile.print(IOContext(stdout, :displaysize => (24, 200)); C=true, combine=true, mincount=100)
    # # ProfileView.view()
    # GC.gc(true)
end

profile()

using ImageFiltering
using ImageFiltering.RFFT

function mwe()
    a = rand(Float64, 10, 10)
    out1 = rfft(a)

    buf = RFFT.RCpair{Float64}(undef, size(a))
    rfft_plan = RFFT.plan_rfft!(buf)
    copy!(buf, a)
    out2 = complex(rfft_plan(buf))

    return out1 ≈ out2
end
mwe()