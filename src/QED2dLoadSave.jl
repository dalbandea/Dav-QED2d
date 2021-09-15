function save_gauge(U, file::String, prm::LattParm)
	n_links = prm.iL[1]*prm.iL[2]*2
	U_h = ones(ComplexF64, prm.iL[1], prm.iL[2], 2)
	CUDA.copyto!(U_h, U)
	Ureal = reshape(real.(U_h), n_links)
	Uimag = reshape(imag.(U_h), n_links)
	DelimitedFiles.writedlm(file, hcat(Ureal, Uimag))
	
	return nothing
end

function load_gauge(U, file::String, prm::LattParm)
	tmp = DelimitedFiles.readdlm(file) |> x -> Complex.(x[:, 1], x[:, 2])
	tmp = reshape(tmp, (prm.iL[1], prm.iL[2], 2))
	# U = CUDA.CuArray{ComplexF64}(undef, (prm.iL[1], prm.iL[2], 2))
	CUDA.copyto!(U, tmp)

	return nothing
end
