using ImageFiltering, ImageCore, OffsetArrays, FFTViews, ComputationalResources
using LinearAlgebra
using Test
using ImageFiltering: IdentityUnitRange
using ImageFiltering: borderinstance, filtfft

@testset "tiling" begin
    m = zeros(UInt8, 20, 20)
    for i = -2:2; m[diagind(m,i)] .= 0xff; end
    kernel = KernelFactors.prewitt((true,true), 1)
    kp = broadcast(*, kernel...)
    # target is the result we want to get
    target = zeros(20, 20)
    dv = [42.5, 85, 85, 42.5]
    for i = 1:4
        target[diagind(target, i)] .= dv[i]
        target[diagind(target, -i)] .= -dv[i]
    end
    mf = imfilter(m, kernel)
    @test mf[2:19, 2:19] ≈ target[2:19, 2:19]
    for r in (CPU1(Algorithm.FIR()), CPU1(Algorithm.FIRTiled()),
              CPUThreads(Algorithm.FIR()), CPUThreads(Algorithm.FIRTiled()))
        mf = @inferred(imfilter(r, m, kernel))
        @test mf[2:19, 2:19] ≈ target[2:19, 2:19]
    end
    mf = imfilter(m, (kp,))
    @test mf[2:19, 2:19] ≈ target[2:19, 2:19]
    # A big image
    img = rand(1000,999)
    kf = kernelfactors((rand(7), rand(7)))
    k = broadcast(*, kf...)
    imgf = imfilter(m, k, Algorithm.FIR())
    for r in (CPU1(Algorithm.FIR()), CPU1(Algorithm.FIRTiled()),
              CPUThreads(Algorithm.FIR()), CPUThreads(Algorithm.FIRTiled()))
        @test imfilter(r, m, kf) ≈ imgf
    end
end

@testset "FIR/FFT" begin
    f32type(img) = f32type(eltype(img))
    f32type(::Type{C}) where {C<:Colorant} = base_colorant_type(C){Float32}
    f32type(::Type{T}) where {T<:Number} = Float32
    zerona!(a) = (a[isnan.(a)] .= zero(eltype(a)); a)
    zerona!(a, nanflag) = (a[nanflag] .= zero(eltype(a)); a)
    ## Images for which the boundary conditions will be irrelevant
    imgf = zeros(5, 7); imgf[3,4] = 1
    imgi = zeros(Int, 5, 7); imgi[3,4] = 1
    imgg = fill(Gray{N0f8}(0), 5, 7); imgg[3,4] = 1
    imgc = fill(RGB{N0f8}(0,0,0), 5, 7); imgc[3,4] = RGB(1,0,0)
    # Dense inseparable kernel
    kern = [0.1 0.2; 0.4 0.5]
    kernel = OffsetArray(kern, -1:0, 1:2)
    for img in (imgf, imgi, imgg, imgc)
        targetimg = zeros(typeof(img[1]*kern[1]), size(img))
        targetimg[3:4,2:3] = rot180(kern) .* img[3,4]
        @test @inferred(imfilter(img, kernel)) ≈ targetimg
        @test @inferred(imfilter(img, (kernel,))) ≈ targetimg
        @test @inferred(imfilter(f32type(img), img, kernel)) ≈ float32.(targetimg)
        ret = imfilter(img, kernel)
        fill!(ret, zero(eltype(ret)))
        @test @inferred(imfilter!(ret, img, kernel)) ≈ targetimg
        fill!(ret, zero(eltype(ret)))
        @test @inferred(imfilter!(CPU1(Algorithm.FIR()), ret, img, kernel)) ≈ targetimg
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(img))))
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32.(targetimg)
            fill!(ret, zero(eltype(ret)))
            @test @inferred(imfilter!(ret, img, kernel, border)) ≈ targetimg
            for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(img, (kernel,), border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32.(targetimg)
                fill!(ret, zero(eltype(ret)))
                @test @inferred(imfilter!(CPU1(alg), ret, img, kernel, border)) ≈ targetimg
            end
            @test_throws MethodError imfilter!(CPU1(Algorithm.FIR()), ret, img, kernel, border, Algorithm.FFT())
        end
        targetimg_inner = OffsetArray(targetimg[2:end, 1:end-2], 2:5, 1:5)
        @test @inferred(imfilter(img, kernel, Inner())) ≈ targetimg_inner
        @test @inferred(imfilter(f32type(img), img, kernel, Inner())) ≈ float32.(targetimg_inner)
        for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
            @test @inferred(imfilter(img, kernel, Inner(), alg)) ≈ targetimg_inner
            @test @inferred(imfilter(f32type(img), img, kernel, Inner(), alg)) ≈ float32.(targetimg_inner)
            @test @inferred(imfilter(CPU1(alg), img, kernel, Inner())) ≈ targetimg_inner
        end
    end
    # Factored kernel
    kernel = (OffsetArray([0.2,0.8], -1:0), OffsetArray([0.3 0.6], 0:0, 1:2))
    kern = parent(kernel[1]).*parent(kernel[2])
    for img in (imgf, imgi, imgg, imgc)
        targetimg = zeros(typeof(img[1]*kern[1]), size(img))
        targetimg[3:4,2:3] = rot180(kern) .* img[3,4]
        ret = similar(targetimg)
        @test @inferred(imfilter(img, kernel)) ≈ targetimg
        @test @inferred(imfilter(f32type(img), img, kernel)) ≈ float32.(targetimg)
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(img))))
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32.(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32.(targetimg)
            end
            @test_throws MethodError imfilter!(CPU1(Algorithm.FIR()), ret, img, kernel, border, Algorithm.FFT())
            @test_throws ErrorException imfilter!(CPU1(Algorithm.FIR()), ret, img, kernel, borderinstance(border), axes(ret)) #167
        end
        targetimg_inner = OffsetArray(targetimg[2:end, 1:end-2], 2:5, 1:5)
        @test @inferred(imfilter(img, kernel, Inner())) ≈ targetimg_inner
        @test @inferred(imfilter(f32type(img), img, kernel, Inner())) ≈ float32.(targetimg_inner)
        for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
            @test @inferred(imfilter(img, kernel, Inner(), alg)) ≈ targetimg_inner
            @test @inferred(imfilter(f32type(img), img, kernel, Inner(), alg)) ≈ float32.(targetimg_inner)
        end
    end
    # Rational filter coefficients
    kernel = (centered([1//3, 1//3, 1//3]), centered([1//3, 1//3, 1//3]'))
    kern = kernel[1].*kernel[2]
    for img in (imgf, imgi, imgg, imgc)
        targetimg = zeros(typeof(img[1]*kern[1]), size(img))
        targetimg[2:4,3:5] .= img[3,4]*(1//9)
        @test @inferred(imfilter(img, kernel)) ≈ targetimg
        @test @inferred(imfilter(f32type(img), img, kernel)) ≈ float32.(targetimg)
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(img))))
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32.(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
                if alg == Algorithm.FFT() && eltype(img) == Int
                    @test @inferred(imfilter(Float64, img, kernel, border, alg)) ≈ targetimg
                else
                    @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                end
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32.(targetimg)
            end
        end
        targetimg_inner = OffsetArray(targetimg[2:end-1, 2:end-1], 2:4, 2:6)
        @test @inferred(imfilter(img, kernel, Inner())) ≈ targetimg_inner
        @test @inferred(imfilter(f32type(img), img, kernel, Inner())) ≈ float32.(targetimg_inner)
        for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
            if alg == Algorithm.FFT() && eltype(img) == Int
                @test @inferred(imfilter(Float64, img, kernel, Inner(), alg)) ≈ targetimg_inner
            else
                @test @inferred(imfilter(img, kernel, Inner(), alg)) ≈ targetimg_inner
            end
            @test @inferred(imfilter(f32type(img), img, kernel, Inner(), alg)) ≈ float32.(targetimg_inner)
        end
    end
    ## Images for which the boundary conditions matter
    imgf = zeros(5, 7); imgf[1,2] = 1
    imgi = zeros(Int, 5, 7); imgi[1,2] = 1
    imgg = fill(Gray(0), 5, 7); imgg[1,2] = 1
    imgc = fill(RGB(0,0,0), 5, 7); imgc[1,2] = RGB(1,0,0)
    # Dense kernel
    kern = [0.1 0.2; 0.4 0.5]
    kernel = OffsetArray(kern, -1:0, 1:2)
    function target1(img::AbstractArray{T}, kern, border) where T
        if border ∈ ("replicate", "symmetric")
            ret = float64.(zero(img))
            x = img[1,2]
            ret[1,1] = (kern[1,1]+kern[2,1])*x
            ret[2,1] = kern[1,1]*x
            ret
        elseif border == "circular"
            a = FFTView(float64.(zero(img)))
            x = img[1,2]
            a[0:1, -1:0] = rot180(kern).*x
            parent(a)
        elseif border == "reflect" || isa(border, Fill)
            ret = float64.(zero(img))
            x = img[1,2]
            ret[1,1] = kern[2,1]*x
            ret[2,1] = kern[1,1]*x
            ret
        end
    end
    function target1(img::AbstractArray{T}, kern, ::NA) where T
        ret = float64.(zero(img))
        x = img[1,2]
        ret[1,1] = kern[2,1]*x/(kern[2,1]+kern[2,2])
        ret[2,1] = kern[1,1]*x/sum(kern)
        ret[:,end] .= nan(eltype(ret))  # for the last column, this kernel is entirely in the padding region
        ret
    end
    for img in (imgf, imgi, imgg, imgc)
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(img))))
            targetimg = target1(img, kern, border)
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32.(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32.(targetimg)
            end
        end
        border = NA()
        targetimg0 = target1(img, kern, border)
        nanflag = isnan.(targetimg0)
        targetimg = zerona!(copy(targetimg0))
        @test @inferred(zerona!(imfilter(img, kernel, border))) ≈ targetimg
        @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border))) ≈ float32.(targetimg)
        for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
            @test @inferred(zerona!(imfilter(img, kernel, border, alg), nanflag)) ≈ targetimg
            @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border, alg), nanflag)) ≈ float32.(targetimg)
        end
    end
    # Factored kernel
    kernel = (OffsetArray([0.2,0.8], -1:0), OffsetArray([0.3 0.6], 0:0, 1:2))
    kern = parent(kernel[1]).*parent(kernel[2])
    for img in (imgf, imgi, imgg, imgc)
        for border in ("replicate", "circular", "symmetric", "reflect", Fill(zero(eltype(img))))
            targetimg = target1(img, kern, border)
            @test @inferred(imfilter(img, kernel, border)) ≈ targetimg
            @test @inferred(imfilter(f32type(img), img, kernel, border)) ≈ float32.(targetimg)
            for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
                @test @inferred(imfilter(img, kernel, border, alg)) ≈ targetimg
                @test @inferred(imfilter(f32type(img), img, kernel, border, alg)) ≈ float32.(targetimg)
            end
        end
        border = NA()
        targetimg0 = target1(img, kern, border)
        nanflag = isnan.(targetimg0)
        targetimg = zerona!(copy(targetimg0))
        @test @inferred(zerona!(imfilter(img, kernel, border))) ≈ targetimg
        @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border))) ≈ float32.(targetimg)
        for alg in (Algorithm.FIR(), Algorithm.FIRTiled(), Algorithm.FFT())
            @test @inferred(zerona!(imfilter(img, kernel, border, alg), nanflag)) ≈ targetimg
            @test @inferred(zerona!(imfilter(f32type(img), img, kernel, border, alg), nanflag)) ≈ float32.(targetimg)
        end
    end
    # filtering with a 0d kernel
    a = rand(3,3)
    # @test imfilter(a, reshape([2])) ≈ 2a
    # imfilter! as a copy
    ret = zeros(3, 3)
    b = @inferred(imfilter!(CPU1(Algorithm.FIR()), ret, a, (), NoPad()))
    @test b == a
    @test !(b === a)
    # OffsetArrays
    img = OffsetArray(rand(RGB{N0f8}, 80, 100), (-5, 3))
    imgf = imfilter(img, Kernel.gaussian((3,3)))
    @test axes(imgf) == axes(img)
end

@testset "Borders (issue #85)" begin
    A = ones(8, 8)
    r1 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(0))
    r2 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(10))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(0, (3,3)))
    r2 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(10, (3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(0, (3,3),(3,3)))
    r2 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(10, (3,3), (3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(0, [3,3],[3,3]))
    r2 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(10, [3,3], [3,3]))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(0, Kernel.gaussian((1,1),(3,3))))
    r2 = imfilter(A, Kernel.gaussian((1,1),(3,3)), Fill(10, Kernel.gaussian((1,1),(3,3))))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    B = fill!(similar(A), 0)
    C = fill!(similar(A), 0)
    r1 = imfilter!(B, A, Kernel.gaussian((1,1),(3,3)), Fill(0))
    r2 = imfilter!(C, A,  Kernel.gaussian((1,1),(3,3)), Fill(10))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    fill!(B, 0); fill!(C, 0)
    r1 = imfilter!(B, A, Kernel.gaussian((1,1),(3,3)), Fill(0, (3,3)))
    r2 = imfilter!(C, A,  Kernel.gaussian((1,1),(3,3)), Fill(10, (3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    fill!(B, 0); fill!(C, 0)
    r1 = imfilter!(B, A, Kernel.gaussian((1,1),(3,3)), Fill(0, (3,3),(3,3)))
    r2 = imfilter!(C, A,  Kernel.gaussian((1,1),(3,3)), Fill(10, (3,3),(3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    fill!(B, 0); fill!(C, 0)
    r1 = imfilter!(B, A, Kernel.gaussian((1,1),(3,3)), Fill(0, [3,3],[3,3]))
    r2 = imfilter!(C, A,  Kernel.gaussian((1,1),(3,3)), Fill(10, [3,3],[3,3]))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]


    g = collect(KernelFactors.gaussian(1,3))
    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(0))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(10))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(0, (3,3)))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(10, (3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(0, (3,3),(3,3)))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(10, (3,3),(3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(0, [3,3],[3,3]))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(10, [3,3],[3,3]))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(0))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,0}(centered(g)),), Fill(10))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(0, (3,3)))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(10, (3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(0, (3,3),(3,3)))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(10, (3,3),(3,3)))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    r1 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(0, [3,3],[3,3]))
    r2 = imfilter(A, ( ImageFiltering.ReshapedOneD{2,1}(centered(g)),), Fill(10, [3,3],[3,3]))
    @test r1[4,4] == r2[4,4]
    @test r1[1,1] != r2[1,1]

    err = ArgumentError("$(typestring(Fill{Int,1}))(0, (3,), (3,)) lacks the proper padding sizes for an array with 2 dimensions")
    kern = Kernel.gaussian((1,1),(3,3))
    @test_throws err imfilter(CPU1(), A, kern, Fill(0, (3,)))
    kernf = ImageFiltering.factorkernel(kern)
    err = DimensionMismatch("output indices (OffsetArrays.IdOffsetRange(values=0:9, indices=0:9), OffsetArrays.IdOffsetRange(values=1:8, indices=1:8)) disagree with requested indices (1:8, 0:9)")
    @test_throws err imfilter(CPU1(), A, kern, Fill(0, (1, 0)))
    @test_throws DimensionMismatch imfilter(CPU1(), A, kern, Fill(0, (0, 1)))
    @test_throws DimensionMismatch imfilter(CPU1(), A, kern, Fill(0, (0, 0)))
end

@testset "Complex FFT" begin

    A = rand(10, 10)
    B = rand(10, 10)
    @test filtfft(A, B) ≈ filtfft(ComplexF32.(A), B)
    @test filtfft(A, B) ≈ filtfft(A, ComplexF32.(B))
    @test filtfft(A, B) ≈ filtfft(ComplexF32.(A), ComplexF32.(B))

    C = rand(9, 9)
    D = rand(9, 9)
    @test filtfft(C, D) ≈ filtfft(ComplexF32.(C), D)
    @test filtfft(C, D) ≈ filtfft(C, ComplexF32.(D))
    @test filtfft(C, D) ≈ filtfft(ComplexF32.(C), ComplexF32.(D))
end
