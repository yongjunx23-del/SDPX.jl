using SDPX, MultiFloats, MultiFloatLinearAlgebra, BigFloatLinearAlgebra
using Serialization, SHA, TOML, LinearAlgebra
include(joinpath(@__DIR__,"basis.jl"))
BLAS.set_num_threads(1)
input=ENV["N14_INPUT"]
expected="19514baaf8e8a15f028d22f368c9bbd2a9765685916ef3dc6696ac91ad7d3e30"
@assert open(io->bytes2hex(sha256(io)),input)==expected
rows=deserialize(input);T=Float64x4
@assert size(rows.real)==size(rows.imag)==(9300,65)
c=zeros(T,65);c[1]=-T(3);c[2]=T(3)
timed=@timed sdpx_basis(rows.real,rows.imag,c)
out=ENV["CAMPAIGN_RESULT"];path=joinpath(out,"reference_basis.jls")
@assert !ispath(path)
open(io->serialize(io,timed.value),path*".part","w");mv(path*".part",path)
receipt=Dict("status"=>"prepared_not_solved","input_sha256"=>expected,
 "basis_sha256"=>open(io->bytes2hex(sha256(io)),path),"source_bits"=>1024,"basis_bits"=>1024,
 "seconds"=>timed.time,"bytes"=>timed.bytes,"peak_rss"=>Sys.maxrss())
open(io->TOML.print(io,receipt;sorted=true),joinpath(out,"basis_receipt.toml"),"w")
