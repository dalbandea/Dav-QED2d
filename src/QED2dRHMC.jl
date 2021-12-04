
k(eps) = sqrt(1.0-eps)
v(n, eps) = Elliptic.K(k(eps)^2)/(2.0*n+1.0)
a_r(r, n, eps) = Jacobi.cn(r*v(n,eps), k(eps)^2)^2/Jacobi.sn(r*v(n,eps), k(eps)^2)^2
c_r(r, n, eps) = Jacobi.sn(r*v(n,eps), k(eps)^2)^2


mu(j, n, eps, r_b) = r_b * a_r(2*j, n, eps)^(1/2)
nu(j, n, eps, r_b) = r_b * a_r(2*j-1, n, eps)^(1/2)

function rho_mu(j, k, l, n, eps, r_b)
	if(j<k || j>l)
		throw("j is not between k and l")
	end

	res = ( nu(j, n, eps, r_b)^2 - mu(j, n, eps, r_b)^2 )

	for m in k:l
		if m!=j
			res *= (nu(m, n, eps, r_b)^2 - mu(j, n, eps, r_b)^2)/(mu(m, n, eps, r_b)^2 - mu(j, n, eps, r_b)^2)
		end
	end

	return res
end

function P(k, l, n, eps, r_b, Y)
	dim = size(Y,2) # get dimensions of matrix
	res = LinearAlgebra.I(dim)

	# using LinearAlgebra.I(dim) makes it compute correctly when Y is just a
	# float number
	if(k!=0 && l!=0)
		for j in k:l
			res = res .+ rho_mu(j, k, l, n, eps, r_b)*(Y .+ mu(j, n, eps, r_b)^2*LinearAlgebra.I(dim) )^(-1)
		end
	end
	println(res)
	
	return res
end


function d(n, eps)
	res = k(eps)^(2*n+1)

	for i in 1:2:2*n+1
		res *= c_r(i, n, eps)^2
	end

	return res
end

delta(n, eps) = d(n, eps)^2 /(1+sqrt(1-d(n,eps)^2))^2

function A(n, eps)
	res = 2/(1+sqrt(1-d(n,eps)^2))

	for i in 1:2:2*n-1
		res *= c_r(i,n,eps)
	end

	for i in 2:2:2*n
		res *= 1/c_r(i,n,eps)
	end

	return(res)
end


function error(Y, Yapprox)
	return abs(1 .- sqrt(Y)*Yapprox[1])
end
