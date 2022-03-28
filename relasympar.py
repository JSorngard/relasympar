
from cmath import pi
import math
import numpy as np
from sympy.physics.wigner import wigner_3j, clebsch_gordan

def relativistic_asymmetry_parameter(n: int, kappa0: int, normalize=True):
	import sympy as sp
	
	if not isinstance(normalize, bool):
		raise TypeError("normalize must be a boolean")
	if not isinstance(kappa0, int):
		raise TypeError("initial kappa must be an integer")
	if kappa0 == 0:
		raise ValueError("initial kappa can not be 0")
	if not isinstance(n, int):
		raise TypeError("n must be an integer")
	if n < 0:
		raise ValueError("n must be non-negative")
	
	#Define symbols
	zeta = sp.symbols('zeta')

	#Determine initial state of the system
	l0,j0 = l_and_j_from_kappa(kappa0)
	normalization = sp.sqrt((2*n + 1)/(4*sp.pi))
	if normalize and n > 0:
		normalization /= relativistic_asymmetry_parameter(0, kappa0)

	sum = 0
	for mj in np.arange(-j0,j0+1,1):
		for k in [0,2]:
			for kp in [0,2]:
				for kf in final_kappas(kappa0):
					for kfp in final_kappas(kappa0):
						lf, j = l_and_j_from_kappa(kf)
						lfp, jp = l_and_j_from_kappa(kfp)
						lg = lf-int(math.copysign(1,kf))
						lgp = lf-int(math.copysign(1,kfp))

						prefactor = phase((j+k+j0)+(j-mj)+(jp+kp+j0)+(jp-mj))*wigner_3j(j,k,j0,-mj,0,mj)*wigner_3j(jp,kp,j0,-mj,0,mj)

						relsum = 0
						for ms in [-1/2,1/2]:
							for mlf in np.arange(-lf,lf+1,1):
								for mlfp in np.arange(-lfp,lfp+1,1):
									relsum += clebsch_gordan(lf,1/2,j,mlf,ms,mj)*clebsch_gordan(lfp,1/2,jp,mlfp,ms,mj)*phase(mlfp)*tripple_harmonic_integral(lf,lfp,n,mlf,-mlfp,0)
							
							for mlg in range(-lg, lg+1):
								for mlgp in range(-lgp,lgp+1):
									relsum += zeta**2*clebsch_gordan(lg,1/2,j,mlg,ms,mj)*clebsch_gordan(lgp,1/2,jp,mlgp,ms,mj)*phase(mlgp)*tripple_harmonic_integral(lg,lgp,n,mlg,-mlgp,0)
						
						relsum *= sp.symbols(f"M_{kf}")*sp.conjugate(sp.symbols(f"M_{kfp}"))

						sum += prefactor*relsum
	
	return sp.simplify(normalization*sum)

def tripple_harmonic_integral(j1,j2,j3,m1,m2,m3):
	return math.sqrt((2*j1+1)*(2*j2+1)*(2*j3+1)/(4*pi))*wigner_3j(j1,j2,j3,0,0,0)*wigner_3j(j1,j2,j3,m1,m2,m3)

def phase(x):
	if x%2 == 0:
		return 1
	else:
		return -1

def l_and_j_from_kappa(kappa):
	"""Returns the l and j quantum numbers that correspond to the input kappa quantum number"""
	if not isinstance(kappa,int):
		raise TypeError("kappa must be an integer")

	j = (2*abs(kappa)-1)/2
	l = 0
	if kappa < 0:
		l = -kappa-1
	else:
		l = kappa
	return l,j

def final_kappas(start_kappa):
	"""Returns the kappa quantum number of the final states reachable by two photons
	from the initial state with the given kappa quntum number"""

	if not isinstance(start_kappa,int):
		raise TypeError("initial kappa must be an integer")

	sig = int(start_kappa/abs(start_kappa))
	mag = abs(start_kappa)
	if mag == 1:
		return [sig*mag, -sig*(mag+1), sig*(mag+2)]
	elif mag == 2:
		return [-sig*(mag-1), sig*mag, -sig*(mag+1), sig*(mag+2)]
	else:
		return [sig*(mag-2), -sig*(mag-1), sig*mag, -sig*(mag+1), sig*(mag+2)]


def main():
	print(relativistic_asymmetry_parameter(2, -1))

if __name__ == '__main__':
	main()