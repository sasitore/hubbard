from qutip import (Qobj, about, basis, expect, fock, fock_dm, 
                   mesolve, smesolve, ssesolve, mcsolve, qeye, tensor, 
                   ket2dm, ptrace, isket)
import time
import numpy as np
import matplotlib.pyplot as plt
import itertools


#Eigenstates and eigenvalues
##Non-interacting one-particle eigenstates
def eigenstates0(L):
    eigs = []
    for k in range(1,L+1):
        eig_k = 0
        for n in range(1,L+1):
            eig_k += np.sin(np.pi*k*n/(L+1))*fock(L,n-1)
        eig_k = np.sqrt(2/(L+1))*eig_k  
        eigs.append(eig_k)
    return eigs


def energy_eigs(L, J=1):
	energies = np.zeros(L)
	for k in range(L):
		energies[k] = np.round(-2*J*np.cos(np.pi*(k+1)/(L+1)),6)
		if abs(energies[k])<1e-10:
			energies[k]=0
	return energies
	

##Minimum and maximum eigenvalues
def minmax_eigvals(t,psi):
    if isket(psi):
        rho = ket2dm(psi)
    else:
        rho=psi
    
    eigvals = rho.eigenenergies()
    
    return [min(eigvals),max(eigvals)]




#Purity and entropies
##Total purity
def purity(t, psi):
    psi = Qobj(psi)
    if isket(psi):
        rho = ket2dm(psi)
    else:
        rho=psi
    return (rho**2).tr()


##Von Neumann entropy of the full rho
def von_neumann_entropy(t, psi):
    if isket(psi):
        rho = ket2dm(psi)
    else:
        rho=psi
        
    eigvals = rho.eigenenergies()  # Eigenvalues of reduced density matrix
    
    # Filter out tiny negative or zero eigenvalues due to numerical noise
    eigvals = np.real_if_close(eigvals)
    eigvals = np.clip(eigvals, 0, 1)  # ensure within [0,1]
    eigvals = eigvals[eigvals > 1e-12]  # remove too-small values to avoid log(0)

    entropy = -np.sum(eigvals * np.log2(eigvals))
    
    return entropy


##Purity of the first particle
def purity_rho1(t, psi):
    if isket(psi):
        rho = ket2dm(psi)
    else:
        rho=psi
    rho1 = rho.ptrace(0)
    return (rho1**2).tr()    


##Von Neumann entropy for the first particle
def von_neumann_entropy_rho1(t, psi):
    if isket(psi):
        rho = ket2dm(psi)
    else:
        rho=psi
    rho1 = rho.ptrace(0)  # Partial trace over the second subsystem

    eigvals = rho1.eigenenergies()  # Eigenvalues of reduced density matrix
    
    # Filter out tiny negative or zero eigenvalues due to numerical noise
    eigvals = np.real_if_close(eigvals)
    eigvals = np.clip(eigvals, 0, 1)  # ensure within [0,1]
    eigvals = eigvals[eigvals > 1e-12]  # remove too-small values to avoid log(0)

    entropy = -np.sum(eigvals * np.log2(eigvals))
    
    return entropy





#Useful operators
##Permutation operators on the particle subspaces
def permute_particles(L,*idx):
    #If idx is NOT a series of elements, but a tuple, 
    #a list or a np.array -> Extract it and transform it to a list
    if len(idx) == 1 and isinstance(idx[0], (list, tuple, np.ndarray)):
        idx = list(idx[0])
    else:    
        #If the idx is given as a series of integers, 
        #then the input is a tuple -> Transform to a list
        idx = list(idx)
        
    
    res = 0
    N=len(idx)
    
    #For cycle over all combinations of Fock states
    for comb in itertools.product(range(L),repeat=N):
        comb = np.array(comb)
        comb_swap = comb[[(i-1) for i in idx]]
        vectors = [fock(L,i) for i in comb]
        vectors_swapped = [fock(L,i) for i in comb_swap]
        
        vec0  = tensor(*vectors)
        vec_swap = tensor(*vectors_swapped)
        res += vec0*vec_swap.dag()
        
    return res






##Operator variance (<A^2> - <A>^2)
def var_op(op):
    def f(t, st):
        rho = st if st.isoper else (st*st.dag())  # takes both pure state and density
        m1 = (op*rho).tr()                        # <A>
        m2 = ((op*op)*rho).tr()                   # <A^2>
        return float((m2 - m1*m1).real)
    return f






#Dealing with N particles
##Multi-particle tensor: it builds sum of an operator A over all particles
def multipart_tensor(op: Qobj, L: int, N: int, order=1, group_size=None):
	"""op: Qobj, single-particle operator to distribute in the particle slots
	L: int, number of sites
	N: int, number of particles
	order: int, how many times does the operator show up in each tensor product (ex. 2 in a 2-particle potential, 1 in kinetic energy)
	"""
	
	#Notice: Qutip can sum a tensor to 0, without needing redefinitions
	tot = 0
	
	#Applying the operator op order times, and the identity (N-order) times
	I = qeye(L)
	for idx_tuple in itertools.combinations(range(N), order):
		ops = [qeye(L)]*N
		for j in idx_tuple:
			ops[j] = op
		#Adding each new combination to the sum
		tot += tensor(*ops)
	
	return tot
	
	




#Simulation
##Simulate function
def simulate(H, rho0, tlist, e_ops, c_ops=None, sc_ops=None, gamma=0, gamma_m=0, ntraj=1, seed=None, solution_type=None):
	if seed==None:
		seed = int(time.time())

	#Mesolve if no probing; ssesolve if probing with no decoherence; smesolve if probing + decoherence
	if solution_type==None or solution_type=="default":
		if gamma_m==0:
			result = mesolve(H,rho0, tlist, e_ops=e_ops, c_ops=c_ops)
		elif gamma==0:
			result = ssesolve(H,rho0,tlist, sc_ops=sc_ops, e_ops=e_ops, ntraj=ntraj, seeds=seed)
		else:
			result = smesolve(H,rho0,tlist, sc_ops=sc_ops, c_ops = c_ops, e_ops=e_ops, ntraj=ntraj, seeds=seed)
	
	if solution_type=="master_eq" or solution_type=="master_equation":
		result = smesolve(H,rho0,tlist, sc_ops=sc_ops, c_ops = c_ops, e_ops=e_ops, ntraj=ntraj, seeds=seed)
	
	if solution_type=="montecarlo" or solution_type=="mcsolve":
		result = mcsolve(H, rho0, tlist, c_ops = sc_ops, e_ops=e_ops, ntraj=ntraj, seeds=seed)
	
	
	return result
	
		
	
##Result dictionary
    #This allows to avoid calling result like: 
    #result.expect[keys["P_fock1"]],but more simply like res["P_fock1"]
def result_dict(result, keys):
    out = {}
    for name, idxs in keys.items():
        if isinstance(idxs, int):
            # single operator → single time series
            out[name] = result.expect[idxs]
        else:
            # list of operators → stack into array (n_ops x n_times)
            out[name] = np.array([result.expect[i] for i in idxs])
    return out



##Extra observables that can be calculated from the primary ones of the result
def secondary_observables(result, N, L, J=1):
	out = dict(result)        #superficial copy of the result (doesn't change the original dictionary)

	#Expected positions
	for n in range(1,N+1):
		prob_focks = result[f"P_fock{n}"]
		out[f"x{n}"] = sum([(i+1)*prob_focks[i] for i in range(L)])
	
	
	#Expected energies
	##Calculating energies
	energies = np.zeros(L)
	for k in range(L):
		energies[k] = np.round(-2*J*np.cos(np.pi*(k+1)/(L+1)),6)
		if abs(energies[k])<1e-10:
			energies[k]=0

	for n in range(1,N+1):
		prob_eig = result[f"P_eig{n}"]
		out[f"energy{n}"] = sum([energies[k]*prob_eig[k] for k in range(L)])

	
	#Even and odd subspaces
	for n in range(1,N+1):
		prob_eig = result[f"P_eig{n}"]
		out[f"even{n}"] = sum([prob_eig[l] for l in range(L) if l%2==0])
		out[f"odd{n}"] = sum([prob_eig[l] for l in range(L) if l%2==1])
	
	
	return out
	



	
 	
 
 ##Plots
def plots(tlist, res, N, L, probed_n=None, draw=None):
	"""tlist: time array;
	res: result dictionary;
	N: number of particles;
	L: number of sites;
	draw: list of strings, tells what to draw:
		position: Expected positions of the N particles
		sites_occupations: Probability of occupation of each site for each particle
		eigen_probs: Probability of energy eigenstates for each particle
		probed_site_occ: Expected total occupation of central site
		probed_site_particles: Probability of each particle to be in the probed site
		parity: even and odd space decomposition for each particle
		
	"""
	
	
	def newfig(x=50,y=50,h=700, b=525):    #horizontal and vertical positions, height and base size of the figures
		fig = plt.figure()
		fig.canvas.manager.window.setGeometry(x, y, h, b)
		return fig
	
	if draw==None:
		draw = ["position", "sites_occupations", "eigen_probs", "probed_site_particles", "probed_site_occ",
		 "energy", "parity", "entropy_rho1", "bosonicness", "symmetries", "center_current"]
	
	if "everything" in draw:
		draw = ["position", "sites_occupations", "eigen_probs", "probed_site_particles", "probed_site_occ",
		 "energy", "parity", "entropy_tot", "entropy_rho1", "bosonicness", 
		 "symmetries", "occ_probed", "center_current", "center_current_var"]
	
	part_colors = ['r','b','g','k', 'm', 'c']
	
	energies = energy_eigs(L)
	
	if "position" in draw or "positions" in draw:
		newfig()
		legend = []
		for n in range(1,N+1):
			plt.plot(tlist, res[f"x{n}"], part_colors[n-1])
			legend.append(f"Particle {n}")
		plt.legend(legend)
	
	
	if "sites_occupations" in draw:
		for n in range(1,N+1):
			p_fock = res[f"P_fock{n}"]
			newfig()
			for k in range(L):
				plt.plot(tlist, p_fock[k])
			plt.legend(range(L))
			plt.title(f"Site occupation probabilities for particle {n}")
	
	
	if "eigen_probs" in draw or "eigenstates_probs" in draw or "prob_eigs" in draw or "P_eig" in draw:
		for n in range(1,N+1):
			p_eig = res[f"P_eig{n}"]
			newfig()
			for k in range(L):
				if k<int(L/2)+1:
					plt.plot(tlist, p_eig[k])
				else:
					plt.plot(tlist, p_eig[k], '--')
			plt.legend(energies)
			plt.title(f"Energy eigenstates probabilities for particle {n}")
				
    
	
	if "probed_site_particles" in draw and probed_n!=None:
		newfig()
		legend = []
		for n in range(1,N+1):
			p_fock = res[f"P_fock{n}"]
			plt.plot(tlist, p_fock[probed_n], part_colors[n-1])
			legend.append(f"Particle {n}")
		plt.title("Probability of being in the probed site for each particle")
		plt.legend(legend)


	if ("probed_site_occupation" in draw or "probed_site_occ" in draw) and probed_n!=None:
		newfig()
		legend = []
		plt.plot(tlist, res["occ_probed"])
		plt.title("Occupation of the probed site")
    

	
	
	if "energy" in draw or "energies" in draw:
		for n in range(1,N+1):
			newfig()
			plt.plot(tlist, res[f"energy{n}"])
			plt.title(fr"$\langle E_{n} (t) \rangle $")
		
		
		
	if "parity" in draw:
		for n in range(1,N+1):
			newfig()
			plt.plot(tlist, res[f"even{n}"], 'r')
			plt.plot(tlist, res[f"odd{n}"], 'g--')
			plt.legend([f"Even subspace (particle {n})", f"Odd subspace (particle {n})"])
	
	
	if "parity_op" in draw:
		for n in range(1,N+1):
			newfig()
			plt.plot(tlist, res[f"Parity_op{n}"])
			plt.title(f"Parity of particle {n}")
	
	if "entropy_tot" in draw or "entropy" in draw:
		newfig()
		plt.plot(tlist, res["entropy"])
		plt.title(r"Von Neumman entropy of $\rho$")
	
	
	if "entropy_rho1" in draw:
		newfig()
		plt.plot(tlist, res["entropy_rho1"])
		plt.title(r"Von Neumman entropy of $\rho_1$")
	
	
	if "bosonicness" in draw and "p_exc" in res:
		newfig()
		bosonicness = [0.5*(1+p) for p in res["p_exc"]]
		plt.plot(tlist, bosonicness)
		plt.ylim([-0.05,1.05])
		plt.title(r"Bosonicness: $\frac{1}{2}(1+P_{12})$")
	
	
	if N==3 and "symmetries" in draw:
		newfig()
		plt.plot(tlist, res["boson_sub"], 'r')
		plt.plot(tlist, res["fermion_sub"], 'b')
		plt.plot(tlist, res["immanon_sub"], 'g')
		plt.legend(["boson", "fermion", "(2,1)-immanon"])
		
	
	if "center_current" in draw:
		newfig()
		plt.plot(tlist, res["ct_current"])
		plt.title("Current from the central site")
		plt.ylabel(r"$j_c$")
	
	
	if "center_current_var" in draw:
		newfig()
		plt.plot(tlist, res["ct_current_var"])
		plt.title("Variance of the current from the central site")
		plt.ylabel(r"$\langle j_c\rangle^2 - \langle j_c^2\rangle$")
	

	return None


