module GaborKernels

export Gabor

"""
    Gabor(size_or_axes, wavelength, orientation; kwargs...)
    Gabor(size_or_axes; wavelength, orientation, kwargs...)

Generate the 2-D Gabor kernel in the spatial space.

# Arguments

## `kernel_size::Dims{2}` or `kernel_axes::NTuple{2,<:AbstractUnitRange}`

Specifies either the size or axes of the output kernel. The axes at each dimension will be
`-r:r` if the size is odd.

## `wavelength::Real`(λ>=2)

This is the wavelength of the sinusoidal factor of the Gabor filter kernel and herewith the
preferred wavelength of this filter. Its value is specified in pixels.

The value `λ=2` should not be used in combination with phase offset `ψ=-π/2` or `ψ=π/2`
because in these cases the Gabor function is sampled in its zero crossings.

In order to prevent the occurence of undesired effects at the image borders, the wavelength
value should be smaller than one fifth of the input image size.

## `orientation::Real`(θ∈[0, 2pi]):

This parameter specifies the orientation of the normal to the parallel stripes of a Gabor
function [3].

# Keywords

## `bandwidth=1` (b>0)

The half-response spatial frequency bandwidth b (in octaves) of a Gabor filter is related to
the ratio σ / λ, where σ and λ are the standard deviation of the Gaussian factor of the
Gabor function and the preferred wavelength, respectively, as follows:

```math
b = \\log_2\\frac{\\frac{\\sigma}{\\lambda}\\pi + \\sqrt{\\frac{\\ln 2}{2}}}{\\frac{\\sigma}{\\lambda}\\pi - \\sqrt{\\frac{\\ln 2}{2}}}
```

## `aspect_ratio=0.5`(γ>0)

This parameter, called more precisely the spatial aspect ratio, specifies the ellipticity of
the support of the Gabor function [3]. For `γ = 1`, the support is circular. For γ < 1 the
support is elongated in orientation of the parallel stripes of the function.

## `phase_offset=0` (ψ∈[-π, π])

The values `0` and `π` correspond to center-symmetric 'center-on' and 'center-off'
functions, respectively, while -π/2 and π/2 correspond to anti-symmetric functions. All
other cases correspond to asymmetric functions.

# Examples

There are two ways to use gabor filters: 1) via [`imfilter`](@ref), 2) via `fft` and
convolution theorem. Usually `imfilter` requires a small kernel to work efficiently, while
using `fft` can be more efficient when the kernel has the same size of the image.

## imfilter

```jldoctest gabor_example
julia> using ImageFiltering, FFTW, TestImages, ImageCore

julia> img = TestImages.shepp_logan(256);

julia> kern = Kernel.Gabor((19, 19), 3, 0); # usually a small kernel size

julia> g_img = imfilter(img, real.(kern));

```

## convolution theorem

The convolution theorem is stated as `fft(conv(A, K)) == fft(A) .* fft(K)`:

```jldoctest gabor_example
julia> kern = Kernel.Gabor(size(img), 3, 0);

julia> g_img = ifft(fft(channelview(img)) .* ifftshift(fft(kern))); # apply convolution theorem

julia> @. Gray(abs(g_img));

```

See the [gabor filter demo](@ref demo_gabor_filter) for more examples with images.

# Extended help

Mathematically, gabor filter is defined in spatial space:

```math
g(x, y) = \\exp(-\\frac{x'^2 + \\gamma^2y'^2}{2\\sigma^2})\\exp(i(2\\pi\\frac{x'}{\\lambda} + \\psi))
```

where ``x' = x\\cos\\theta + y\\sin\\theta`` and ``y' = -x\\sin\\theta + y\\cos\\theta``.

# References

- [1] [Wikipedia: Gabor filter](https://en.wikipedia.org/wiki/Gabor_filter)
- [2] [Wikipedia: Gabor transformation](https://en.wikipedia.org/wiki/Gabor_transform)
- [3] [Wikipedia: Gabor atom](https://en.wikipedia.org/wiki/Gabor_atom)
- [4] [Gabor filter for image processing and computer vision](http://matlabserver.cs.rug.nl/edgedetectionweb/web/edgedetection_params.html)
"""
struct Gabor{T<:Complex, TP<:Real, R<:AbstractUnitRange} <: AbstractMatrix{T}
    ax::Tuple{R,R}
    λ::TP
    θ::TP
    ψ::TP
    σ::TP
    γ::TP

    # cache values
    sc::Tuple{TP, TP} # sincos(θ)
    λ_scaled::TP # 2π/λ
    function Gabor{T,TP,R}(ax::Tuple{R,R}, λ::TP, θ::TP, ψ::TP, σ::TP, γ::TP) where {T,TP,R}
        λ > 0 || throw(ArgumentError("`λ` should be positive: $λ"))
        new{T,TP,R}(ax, λ, θ, ψ, σ, γ, sincos(θ), 2π/λ)
    end
end
function Gabor(size_or_axes::Tuple; wavelength, orientation, kwargs...)
    Gabor(size_or_axes, wavelength, orientation; kwargs...)
end
function Gabor(
        size_or_axes::Tuple,
        wavelength::Real,
        orientation::Real;
        bandwidth::Real=1.0f0,
        phase_offset::Real=0.0f0,
        aspect_ratio::Real=0.5f0,
    )
    bandwidth > 0 || throw(ArgumentError("`bandwidth` should be positive: $bandwidth"))
    aspect_ratio > 0 || throw(ArgumentError("`aspect_ratio` should be positive: $aspect_ratio"))
    wavelength >= 2 || @warn "`wavelength` should be equal to or greater than 2" wavelength
    # we still follow the math symbols in the implementation
    λ, γ, ψ = wavelength, aspect_ratio, phase_offset
    # for orientation: Julia follow column-major order, thus here we compensate the orientation by π/2
    θ = convert(float(typeof(orientation)), mod2pi(orientation+π/2))
    # follows reference [4]
    σ = convert(float(typeof(λ)), (λ/π)*sqrt(0.5log(2)) * (2^bandwidth+1)/(2^bandwidth-1))

    params = float.(promote(λ, θ, ψ, σ, γ))
    T = typeof(params[1])

    ax = _to_axes(size_or_axes)
    all(map(length, ax) .> 0) || throw(ArgumentError("Kernel size should be positive: $size_or_axes"))
    if any(r->5λ > length(r), ax)
        expected_size = @. 5λ * length(ax)
        @warn "for wavelength `λ=$λ` the expected kernel size is expected to be larger than $expected_size."
    end

    Gabor{Complex{T}, T, typeof(first(ax))}(ax, params...)
end

@inline Base.size(kern::Gabor) = map(length, kern.ax)
@inline Base.axes(kern::Gabor) = kern.ax
@inline function Base.getindex(kern::Gabor, x::Int, y::Int)
    ψ, σ, γ = kern.ψ, kern.σ, kern.γ
    s, c = kern.sc # sincos(θ)
    λ_scaled = kern.λ_scaled # 2π/λ

    xr = x*c + y*s
    yr = -x*s + y*c
    yr = γ * yr
    return exp(-(xr^2 + yr^2)/(2σ^2)) * cis(xr*λ_scaled + ψ)
end

# Utils

function _to_axes(sz::Dims)
    map(sz) do n
        r=n÷2
        isodd(n) ? UnitRange(-r, n-r-1) : UnitRange(-r+1, n-r)
    end
end
_to_axes(ax::NTuple{N, T}) where {N, T<:AbstractUnitRange} = ax

end
