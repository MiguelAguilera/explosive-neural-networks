import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import numba
from scipy.interpolate import interp1d
import scipy as sp
import os

plt.rc('text', usetex=True)
font = {'size': 20, 'family': 'serif', 'serif': ['latin modern roman']}
plt.rc('font', **font)
plt.rc('legend', **{'fontsize': 18})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

@njit
def log2cosh(x):
    ax = np.abs(x)
    return ax + np.log(1+np.exp(-2*ax))

@njit
def G(m,q,alpha,b,gamma):
    r = q / (1 - b * J * (1 - q))**2
    R = (1 / (b * J) - (1 - 2 * q)) / (1 - b * J * (1 - q))**2
    g = b * m * J
    sqrtD = b * J * np.sqrt(alpha * r)
    zmax=6
    z0=0
    Nint=1001
    x = np.linspace(-1., 1., Nint)*zmax + z0
    Gx = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2) * log2cosh(g + x * sqrtD)
    return np.sum(Gx)*(x[1]-x[0]) 
    
@njit(parallel=True)
def free_energy(m,q,b, alpha, beta, gamma):
    if q<0:
        return 100
    if alpha == 0 and gamma == 0:
        phi = -beta * J / 2 * m**2 + np.log(2 * np.cosh(beta * m * J))
    elif alpha == 0:
        if (1 + gamma / 2 * J * m**2) <= 0:
            return 100
        phi = beta / gamma * np.log(1 + gamma / 2 * J * m**2) - b * J * m**2 + np.log(2 * np.cosh(b * m * J))
    elif gamma == 0:
        if (1 - beta * J * (1 - q)) <0:
            return 100
        r = q / (1 - beta * J * (1 - q))**2
        R = (1 / (beta * J) - (1 - 2 * q)) / (1 - beta * J * (1 - q))**2
        Gmx = G(m,q,alpha,beta,gamma)
        phi = - 1 / 2 * beta * J * m**2 - 1 / 2 * alpha * (beta * J) * (1 + (beta * J) * r * (1 - q)) - 1 / 2 * alpha * (np.log(1 - beta * J * (1 - q)) - beta * J * np.sqrt(r * q))  + Gmx
    else:
        if (1 - b * J * (1 - q)) <=0:
            return 100
        if (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((1 - b * J * (1 - q**2)) / (1 - b * J * (1 - q))**2 - 1)) <= 0:
            return 100
        r = q / (1 - b * J * (1 - q))**2
        R = (1 - b * J * (1 - 2 * q)) / (b * J) / (1 - b * J * (1 - q))**2
        Gmx = G(m,q,alpha,b,gamma)
        if (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((b * J) * (R - r * q) - 1)) <=0:
            return 100
        phi = beta / gamma * np.log(1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((b * J) * (R - r * q) - 1)) - b * J * m**2 - 1 / 2 * alpha * (b * J) * (1 + (b * J) * r * (1 - q) + (b * J) * (R - r * q)) - 1 / 2 * alpha * (np.log(1 - b * J * (1 - q)) - b * J * np.sqrt(r * q)) + Gmx
    return -phi

def get_or_compute_free_energy(m, q, qsg, betas, alpha, betas0, betas0_, gamma, B, overwrite=False):
    """
    Compute free energy values f1 and f2.

    Parameters:
        m, q, qsg: Arrays of m, q, and q_sg values.
        betas, alpha, betas0, betas0_: Arrays of beta values.
        gamma: Scalar gamma value.
        A, B: Integers representing relevant parameters for the file naming.
        overwrite: If True, always compute and overwrite files. Otherwise, load if available.

    Returns:
        f1, f2: Computed or loaded free energy arrays.
    """
    f1 = calc_f(m, q, betas, alpha, betas0, gamma)
    f2 = calc_f(m * 0, qsg, betas, alpha, betas0_, gamma)
    return f1, f2

Nint=10001
    
A=1001
B=10001
betas=1/np.linspace(0.0001,1.5,B)[::-1]
betas=1/np.linspace(0,1.5,B)[::-1]
alphas = np.linspace(0.0, 0.2, A)
#betas=np.linspace(1,10,B)
J=1


gamma=-0.8
gamma2=0.8

overwrite = False

bM=np.ones(A)*np.nan
bM1=np.ones(A)*np.nan
bM2=np.ones(A)*np.nan

bM__=np.ones(A)*np.nan
bM1__=np.ones(A)*np.nan
bM2__=np.ones(A)*np.nan

bg=np.ones(A)
bg1=np.ones(A)
bg2=np.ones(A)

bE=np.zeros(A)*np.nan
bE1=np.zeros(A)*np.nan
bE2=np.zeros(A)*np.nan

bc=np.ones(A)*np.nan
bc1=np.ones(A)*np.nan
bc2=np.ones(A)*np.nan
bc_=np.ones(A)*np.nan
bc1_=np.ones(A)*np.nan
bc2_=np.ones(A)*np.nan


bAT=np.ones(A)*np.nan
bAT1=np.ones(A)*np.nan
bAT2=np.ones(A)*np.nan

mM=np.zeros(A)
qM=np.zeros(A)
mc=np.zeros(A)
qc=np.zeros(A)
mE=np.zeros(A)
qE=np.zeros(A)

bM_=np.ones(A)*np.inf


###    CALCULATE DETERMINISTIC MODEL    ###

filename = 'data/Fig4/beta0_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) +'.npz'
data=np.load(filename)
m0=data['m0']
alphas=data['alphas']



inds=np.where(m0>0.001)[0]
a0M = np.max(alphas[inds])
ia0M = np.argmax(alphas[inds])

x=sp.special.erfinv(m0)
f1= 0.5 * (sp.special.erf(x))**2 + 1 / np.pi * np.exp(-x**2) - 2 / np.pi * (np.exp(-x**2) + np.sqrt((alphas * np.pi) / 2)) * (x * np.sqrt(np.pi) * sp.special.erf(x) + np.exp(-x**2))

x=x*0
f2= 0.5 * (sp.special.erf(x))**2 + 1 / np.pi * np.exp(-x**2) - 2 / np.pi * (np.exp(-x**2) + np.sqrt((alphas * np.pi) / 2)) * (x * np.sqrt(np.pi) * sp.special.erf(x) + np.exp(-x**2))
print(sp.special.erfinv(0))

inds=np.where(f2>f1)[0]
a0c = np.max(alphas[inds])
ia0c = np.argmax(alphas[inds])
print(a0c,a0M)


@njit
def calc_f(m,q,b, alpha, betas, gamma):
    f=np.zeros(len(betas))
    for ib in range(len(betas)):
        f[ib] = free_energy(m[ib],q[ib],b[ib], alpha, betas[ib], gamma)
    return f
    
    
###    RUN OVER BETA VALUES    ###

aM=np.zeros(B)
ac=np.zeros(B)
b0M=np.zeros(B)

M=np.zeros((A,B))
Q=np.zeros((A,B))
Qsg=np.zeros((A,B))
F1=np.zeros((A,B))
F2=np.zeros((A,B))

for ia, alpha in enumerate(alphas):
    alpha = round(alpha, 6)
    filename = 'data/Fig4/alpha=' + str(alpha) + '_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) +'.npz'
    data=np.load(filename) 
    m=data['m']
    q=data['q'] 
    qsg=data['qsg']
    M[ia,:] = m
    Q[ia,:] = q
    Qsg[ia,:] = qsg
#    F1[ia,:]=calc_f(m,q,betas, alpha, betas, 0)
#    F2[ia,:]=calc_f(m*0,qsg,betas, alpha, betas, 0)
    f1,f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas, betas, 0, B, overwrite=overwrite)
    F1[ia,:]=f1
    F2[ia,:]=f2

for ib, beta in enumerate(betas[:-1]):
    beta = round(beta, 6)
    m=M[:,ib]
    q=Q[:,ib]
    qsg=Qsg[:,ib]
    f1=F1[:,ib]
    f2=F2[:,ib]

    if beta>1:
#        if ib>B-10:
#            plt.figure()
#            plt.plot(alphas,m)
#            plt.plot(alphas,q)
#            plt.show()
        inds=np.where(m>0.001)[0]
        if len(inds):
#            i = np.argmax(alphas[inds])
            aM[ib]=np.max(alphas[inds])
#            b0M[ib]= betas[ib] * (1 + gamma / 2 * J * m[ib]**2 + gamma / 2 * J * aM[ib] * ((1 - betas[ib] * J * (1 - q[ib]**2)) / (1 - betas[ib] * J * (1 - q[ib]))**2 - 1))    
        
        inds = np.where(f1-f2>1E-5)[0]
        if len(inds):
            ac[ib]=np.min(alphas[inds])
            
ac[-1]=a0c

plt.figure()
plt.plot(aM,1/betas)
plt.plot(ac,1/betas)



###    RUN OVER ALPHA VALUES    ###

for ia, alpha in enumerate(alphas):
    alpha = round(alpha, 6)
    filename = 'data/Fig4/alpha=' + str(alpha) + '_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) +'.npz'
    data=np.load(filename) 
    m=data['m']
    q=data['q'] 
    qsg=data['qsg']
    C=data['C']
    
#    if alpha>0:
#        print(qsg)
#        exit()
    q1=np.zeros(B)
    q1[1/betas<=(1+np.sqrt(alpha))] = (1-(1/betas)/(1+np.sqrt(alpha)))[1/betas<=(1+np.sqrt(alpha))]
    if ia==1:
        qsg=q1.copy()
    qsg[qsg<q1]=q1[qsg<q1]
    betas=data['betas']
    
    betas0 = betas * (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - q**2)) / (1 - betas * J * (1 - q))**2 - 1))
    betas0[-1]=np.inf

    betas02 = betas * (1 + gamma2 / 2 * J * m**2 + gamma2 / 2 * J * alpha * ((1 - betas * J * (1 - q**2)) / (1 - betas * J * (1 - q))**2 - 1))
    betas02[-1]=np.inf
    
    inds = (C*alpha*betas**2/(1-betas*(1-q))**2 < 1).nonzero()
    if len(inds):
        bAT[ia]=np.max(betas[inds])
        bAT1[ia]=np.max(betas0[inds])
        bAT2[ia]=np.max(betas02[inds])
            

    inds=np.where(m>0.001)[0]
    if len(inds):
        i = np.argmin(betas[inds])
        bM[ia]=betas[inds][i]
        bM1[ia]=betas0[inds][i]
        bM2[ia]=betas02[inds][i]
        
        mM[ia]=m[inds][i]
        qM[ia]=q[inds][i]
        
        s=np.zeros(B)
        s[inds]=1   
        ds=np.diff(s)
        inds1=(ds<0)
        inds2=(ds>0)
        if np.sum(inds2)>1:
            bM__[ia]=betas[:-1][inds1][0]
            bM1__[ia]=betas0[:-1][inds1][0]
            bM2__[ia]=betas02[:-1][inds1][0]

        if alpha>0 and alpha<a0c*3:

            f1,f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas, betas, 0, B, overwrite=False)
            inds = np.where(f1-f2>1E-4)[0]
            s=np.zeros(B)
            s[inds]=1   
            ds=np.diff(s)
            inds1=(ds<0)
            inds2=(ds>0)
            if len(inds1):
                bc[ia]=(np.min(betas[:-1][inds1])+np.min(betas[1:][inds1]))/2

            inds_p = inds.copy()
            betas0_ = betas * (1 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - qsg**2)) / (1 - betas * J * (1 - qsg))**2 - 1))   
            betas0_[-1] = np.inf
            
            

            print(alpha,1/bc1[ia])

            f1,f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas0, betas0_, gamma, B, overwrite=overwrite)
            interp_func1 = interp1d(betas0_, f2, bounds_error=False, fill_value="extrapolate")
            f2_interpolated1 = interp_func1(betas0)
            db = np.gradient(betas,betas0)
            inds = np.where((f1-f2_interpolated1>1E-4))[0]
            
            s=np.zeros(B)
            s[inds]=1   
            ds=np.diff(s)
            inds1=(ds<0)
            inds2=(ds>0)
            print()
            print(alpha)
            print(np.sum(ds>0))
            print(np.sum(ds<0))
            
            if len(inds1):
                bc1[ia]=(np.min(betas0[:-1][inds1])+np.min(betas0[1:][inds1]))/2
                bc1[ia]=max(bc1[ia],bM1[ia])
            elif alpha<a0c:
                bc1[ia]=bM1[ia]
            else:
                bc1[ia]=np.nan
                
            if np.sum(inds2)>1:# and alpha<a0c:
                bc1_[ia]=betas0[:-1][np.where(inds2)[0][1]]
                print('bc1_',bc1_[ia],alpha, 1/betas0[:-1][np.where(inds2)[0]])
                
            
            betas02_ = betas * (1 + gamma2 / 2 * J * alpha * ((1 - betas * J * (1 - qsg**2)) / (1 - betas * J * (1 - qsg))**2 - 1))   
            betas02_[-1] = np.inf
            f1,f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas02, betas02_, gamma2, B, overwrite=overwrite)
            
            # 1. Filter values to keep
            mask_keep = betas02 >= bM2[ia]
            betas02_keep = betas02[mask_keep]
            m_keep = m[mask_keep]
            q_keep = q[mask_keep]
            f1_keep = f1[mask_keep]

            # 2. Values to add from second vector
            mask_add = betas02_ < bM2[ia]
            betas02_add = betas02_[mask_add]
            m_add = np.zeros_like(betas02_add)  # m = 0 for added values
            q_add = qsg[mask_add]  # q = qsg for added values
            f2_add = f2[mask_add]

            # 3. Concatenate results
            betas02a = np.concatenate([betas02_add, betas02_keep])
            ma = np.concatenate([m_add,m_keep])
            qa = np.concatenate([q_add,q_keep])
            f1a = np.concatenate([f2_add,f1_keep])
            
            interp_func1 = interp1d(betas02_, f2, bounds_error=False, fill_value="extrapolate")
            f2_interpolated1 = interp_func1(betas02a)
            inds= (f1a-f2_interpolated1>1E-4).nonzero()[0]
            
            s=np.zeros(len(f1a))
            s[inds]=1   
            ds=np.diff(s)
            inds1=(ds<0)
            inds2=(ds>0)
            
            
            if len(betas02a[1:][inds1]):
                bc2[ia]=np.min(betas02a[1:][inds1])
            else:
                bc2[ia]=np.nan
                
            if np.sum(inds2)>2:# and alpha<a0c:
                bc2_[ia]=betas02a[:-1][np.where(inds2)[0][1]]

        elif alpha==0:
            bc[ia]=1
            bc1[ia]=1
            bc2[ia]=1
            

#    inds=np.where(m>0.001)[0]
    inds= (m>0.001).nonzero()[0]
    if len(inds):
        i = np.argmin(betas[inds])
        bE[ia]=betas[inds][i]
        bE1[ia]=betas0[inds][i]
        
        
        mE[ia]=m[inds][i]
        qE[ia]=q[inds][i]
        
    inds=np.where(q>0.001)[0]
    if len(inds):
        i = np.argmin(betas[inds])
        bg[ia]=betas[inds][i]
        bg1[ia]=betas0[inds][i]
        bg2[ia]=betas02[inds][i]


def mav(arr, window_size,st=0,end=-1):
    smooth_arr = np.convolve(arr, np.ones(window_size) / window_size, mode='same')
    smooth_arr[:st+1]=arr[:st+1]
    smooth_arr[-1:]=arr[-1:]
    return smooth_arr
    
def notnan(x):
    return x[~np.isnan(x)]

def mav2(arr, window_size,st=0,end=-1):
    arr2=arr.copy()
    inds = np.where(~np.isnan(arr2))[0]
    arr2[inds] = mav(arr[inds], window_size,st=0,end=-1)
    return arr2


plt.figure()
plt.plot(alphas,1/bc1)
plt.plot(alphas,1/bc1_,':')

ind0c = np.argmin((alphas-a0c)**2)

bc1[np.isinf(bc1)] = np.nan   
bc1_[np.isinf(bc1_)] = np.nan

inv_bc1_= 1/bc1_
inv_bc1_[np.isnan(bc1_)] = 0
ind = np.argmax(inv_bc1_)

print('ind',ind , alphas[ind],1/bc1_[ind])


bc1m=(bc1[ind]+bc1_[ind])/2
bc1_[ind+1]=bc1m
bc1[ind+1]=bc1m

bc1_[ind+2:]=np.nan
bc1[ind+2:]=np.nan

bc1_[1/bc1_<1E-5]=np.nan
bc1[1/bc1_<1E-3]=np.nan

bc2[0]=np.nan
inv_bc2= 1/bc2
inv_bc2[np.isnan(bc2)] = 0
ind= np.argmax(inv_bc2)
bc2[ind-1]=bM2[ind-1]
bc2[:ind-1]=np.nan

plt.figure()
plt.plot(alphas,1/bc1)
plt.plot(alphas,1/bc1_,':')
plt.figure()
plt.plot(alphas,1/bc2)
plt.plot(alphas,1/bc2_,':')
#plt.show()




print(alphas[ind0c-1])
# ~ bc1_[ind_notnan0-1]=np.inf
print(alphas[ind0c])


plt.figure(figsize=(6, 4),layout='constrained')
plt.plot(alphas,1/bM,'k--',label=r'$\gamma=0$')
plt.plot(alphas,1/bc,'b--')
plt.plot(alphas,1/bg,'c--')
#plt.plot(alphas,1/bE,'k:')
plt.axis([0,0.2,0,1/bg1[-1]])
#plt.title(r'$\gamma=0$')
plt.xlabel(r'$\alpha$')
plt.ylabel(r"$T$", rotation=0, labelpad=15)


plt.text(0.02, 1.5, r'P', size=25)
plt.text(0.02, .25, r'F', size=25)
plt.text(0.08, .1, r'M', size=25)
plt.text(0.085, 0.8, r'SG', size=25)


plt.figure()
plt.plot(alphas,1/bc,color='r')

from matplotlib import cm
cmap = cm.get_cmap('hot_r')

plt.figure(figsize=(6, 4),layout='constrained')
#plt.plot(alphas,1/bM__,'r')
#plt.plot(alphas,1/bc,color='r')
print(bc)

# ~ plt.plot(alphas,1/mav(bg,3),'k--')
#plt.plot(mav(aM,5),1/betas,color=cmap(1.0),label=r'$\gamma=0$')
ac[betas<1]=np.nan
plt.plot(ac,1/betas,'--',color=cmap(1.0))
plt.plot(alphas,1+np.sqrt(alphas),'--',color=cmap(1.0))

plt.xlabel(r'$\alpha$')
plt.ylabel(r"$T$", rotation=0, labelpad=15)

plt.text(0.02, 1.5, r'P', size=25)
plt.text(0.02, .25, r'F', size=25)
plt.text(0.08, .1, r'M', size=25)
plt.text(0.085, 0.8, r'SG', size=25)




inds=np.where(~np.isnan(bM1))[0]
#bM1[inds[-1]+1] = np.inf
# ~ bM1[inds[-1]] = np.inf
bM11=bM1.copy()
bM11[inds]=mav(bM11[inds],3)
bM22=bM2.copy()
bM22[inds]=mav(bM22[inds],3)
#plt.figure(figsize=(6, 4))
#plt.plot(alphas,1/bM1,'k',label=r'$\gamma='+str(gamma)+'$')
#plt.plot(alphas,1/bM1,'r',label=r'$\gamma='+str(gamma)+'$')
#plt.plot(alphas,1/bM1__,'r')
#plt.plot(alphas,1/mav2(bc1,3),'k')
#plt.plot(alphas,1/bc2,'-.r')
#plt.plot(alphas,1/bc1_,'k')
#plt.plot(alphas,1/bg1,'c')
plt.plot(alphas,1/bg1,color=cmap(0.66))
plt.plot(alphas,1/bg2,color=cmap(0.33))
plt.plot(alphas,1/bc1_,color=cmap(0.66))
#plt.plot(alphas,1/mav2(bc2_,3),'-.r')
plt.plot(alphas,1/bc2_,color=cmap(0.33))




bc2[1/bc2<1/bAT2]=np.nan
plt.plot(alphas,1/bc2,color=cmap(0.33))
plt.plot(alphas,1/bc1,color=cmap(0.66))


bAT[1/bAT>1/bM]=np.nan
bAT1[1/bAT1>1/bM1]=np.nan
bAT2[1/bAT2>1/bM2]=np.nan




plt.plot(alphas,1/bAT ,linestyle=(0,(1,3)),color=cmap(1.))
plt.plot(alphas,1/bAT1,linestyle=(1,(1,3)),color=cmap(0.66))
plt.plot(alphas,1/bAT2,linestyle=(2,(1,3)),color=cmap(0.33))


ind=np.where(1/bM<1E-2)[0][0]
bM[ind+1:]=np.nan
bM1[ind+1:]=np.nan
bM2[ind+1:]=np.nan
plt.plot(alphas,1/bM,'--',color=cmap(1.),label=r"$\gamma'=0$")
plt.plot(alphas,1/bM2,color=cmap(0.33),label=r"$\gamma'="+str(gamma2)+'$')
plt.plot(alphas,1/bM1,color=cmap(0.66),label=r"$\gamma'="+str(gamma)+'$')

#plt.plot(alphas,1/bE1,':')
plt.axis([0,0.2,0,1/bg1[-1]])
#plt.title(r'$\gamma='+str(gamma)+'$')
plt.xlabel(r'$\alpha$')
plt.ylabel(r"$T$", rotation=0, labelpad=15)
plt.legend( loc='center right', bbox_to_anchor=(1, 0.43),labelspacing=0.2)


plt.savefig('img/phase-diagram-memory-capacity.pdf', bbox_inches='tight')
#plt.show()


plt.figure()
plt.plot(alphas,mM)
plt.plot(alphas,qM)


    
plt.plot(alphas,mav(mM,3))
plt.plot(alphas,mav(qM,3))
#plt.show()

GN=101
#GN=41
#GN=31
gammas = -np.linspace(-1., 1.0, GN)
aM=np.zeros(GN)
ac=np.zeros(GN)
for ig, gamma in enumerate(gammas):
    gamma = round(gamma, 6)
    bM_ = bM * (1 + gamma / 2 * J * mM**2 + gamma / 2 * J * alpha * ((1 - bM * J * (1 - qM**2)) / (1 - bM * J * (1 - qM))**2 - 1))
    bc_ = bc * (1 + gamma / 2 * J * mc**2 + gamma / 2 * J * alpha * ((1 - bc * J * (1 - qc**2)) / (1 - bc * J * (1 - qc))**2 - 1))
    
#    interp_func = interp1d(bM_,alphas , bounds_error=False, fill_value="extrapolate")
#    aM[ig]= interp_func(2.)
#    
    ind=np.argmin((bM_[:np.argmax(bM_)]-2)**2)
    aM[ig]=alphas[ind]
    bc_=np.zeros(A)
    for ia, alpha in enumerate(alphas):
        alpha = round(alpha, 6)
        filename = 'data/Fig4/alpha=' + str(alpha) + '_A_' + str(A) + '_B_' + str(B) + '_N_' + str(Nint) +'.npz'
        data=np.load(filename) 
        m=data['m']
        q=data['q']
        qsg=data['qsg']
        betas=data['betas']
        
        betas0 = betas * (1 + gamma / 2 * J * m**2 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - q**2)) / (1 - betas * J * (1 - q))**2 - 1))  
        betas0_ = betas * (1 + gamma / 2 * J * alpha * ((1 - betas * J * (1 - qsg**2)) / (1 - betas * J * (1 - qsg))**2 - 1))     
        
        inds=np.where(m>1E-5)[0]
        if len(inds) and alpha>0:

#            f1=calc_f(m,q,betas, alpha, betas0, gamma)
#            f2=calc_f(m*0,qsg,betas, alpha, betas0_, gamma)
            f1,f2 = get_or_compute_free_energy(m, q, qsg, betas, alpha, betas0, betas0_, gamma, B, overwrite=overwrite)
            
            if gamma>0:
                # 1. Filter values to keep
                mask_keep = betas0 >= bM2[ia]
                betas0_keep = betas0[mask_keep]
                m_keep = m[mask_keep]
                q_keep = q[mask_keep]
                f1_keep = f1[mask_keep]

                # 2. Values to add from second vector
                mask_add = betas0_ < bM2[ia]
                betas0_add = betas0_[mask_add]
                m_add = np.zeros_like(betas0_add)  # m = 0 for added values
                q_add = qsg[mask_add]  # q = qsg for added values
                f2_add = f2[mask_add]

                # 3. Concatenate results
                betas0 = np.concatenate([betas0_add, betas0_keep])
                m = np.concatenate([m_add,m_keep])
                q = np.concatenate([q_add,q_keep])
                f1 = np.concatenate([f2_add,f1_keep])

            interp_func1 = interp1d(betas0, f1, bounds_error=False, fill_value="extrapolate")
            f1_interpolated = interp_func1(2.0)
            interp_func2 = interp1d(betas0_, f2, bounds_error=False, fill_value="extrapolate")
            f2_interpolated = interp_func2(2.0)
#            print(gamma,alpha,f1_interpolated,f2_interpolated,f1_interpolated-f2_interpolated)
            if f1_interpolated-f2_interpolated > 1E-5 or alpha>=aM[ig]:
                ac[ig]=alphas[ia-1]
                print(gamma,alphas[ia-1])
#                plt.figure()
#                plt.plot(1/betas0,f1)
#                plt.plot(1/betas0_,f2)
#                plt.show()
                break
#                inds1 = np.where((f1-f2_interpolated2>1E-5))[0]
#                if len(inds):
#                    bc_[ia]=np.max(betas0[inds1])
#                else:
#                    bc_[ia]=bM1[ia]
#                
#    ind=np.argmin((bc_[:np.argmax(bc_)]-2)**2)
#    ac[ig]=min(alphas[ind],aM[ig])


plt.figure(figsize=(6, 4))
#plt.plot(mav(aM,3),gammas,'-k')
#plt.plot(mav(ac,3),gammas,'-r')
plt.plot(aM,gammas,'k')
plt.plot(ac,gammas,'k')
#plt.plot(aE,gammas)
plt.axis([0, np.max(aM)*1.0, np.min(gammas), np.max(gammas)])
plt.ylim(max(gammas), min(gammas))
plt.xlabel(r'$\alpha$')
plt.ylabel(r"$\gamma'$", rotation=0, labelpad=15)

#plt.text(0.02, -.65, r'F', size=25)
#plt.text(0.0525, -.45, r'M', size=25)
#plt.text(0.085, -.25, r'SG', size=25)
plt.text(0.012, -.5, r'F', size=25)
plt.text(0.05, -.25, r'M', size=25)
plt.text(0.085, .0, r'SG', size=25)


plt.savefig('img/memory-capacity.pdf', bbox_inches='tight')

plt.show()


plt.show()
                
