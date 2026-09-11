using SDPX
path=joinpath(pkgdir(SDPX),"benchmark","lifecycle","regression_guard.jl")
text=read(path,String)
needle="\"/tmp/csdr-alpha9-twice/solve-alpha3.bin\""
@assert length(findall(needle,text))==1
# Path adaptation only; reference values, six-field fingerprint and gates unchanged.
text=replace(text,needle=>repr(ENV["CSDR_INPUT"]);count=1)
include_string(Main,text,path)
