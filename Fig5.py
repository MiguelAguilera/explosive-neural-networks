import numpy as np
from matplotlib import pyplot as plt
import skimage
import numba
from numba import njit

plt.rc('text', usetex=True)
font = {'size':15}
plt.rc('font',**font)
plt.rc('legend',**{'fontsize':16})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# ~ from kinetic_ising import ising





def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
def image(data,i):
    im = np.zeros((32,32,3))
    im[:,:,0] = np.reshape(data[b'data'][i,:1024] ,(32,32))
    im[:,:,2] = np.reshape(data[b'data'][i,2048:],(32,32))
    return im/256
#    im = (data[b'data'][0,:1024] + data[b'data'][0,1024:2048] + data[b'data'][0,2048:])/3
#    return np.reshape(im,(32,32))
    
def bw_im(im):
    return np.mean(im,axis=2)
    
    

#nM = 5
#Ms=(2**np.arange(4,4+nM)).astype(int)
#Ms=[16,20,24,28,32,36,40]
Ms=np.arange(1,15)*5

nM=len(Ms)
G=41
# ~ G=11
gammas=-np.linspace(-1,1,G)
beta=2

R=500
# ~ R=50
R=200    
# ~ R=1

@njit
def exp_g(x,gamma1):
    return np.exp((1/gamma1)*np.log(1+gamma1/1*x))  
        
@njit
def update_beta1(xi,s,gamma,beta):
    size=len(s)
    # ~ ov = np.einsum('ai,i->a',xi,s)/size
    ov = np.dot(xi, s) / size
    return beta/(1+gamma*(np.sum(ov**2)/2))
        
@njit
def SequentialGlauberStep(s, H, J, beta,gamma, xi, T, i=None):            # Execute step of the Glauber algorithm
    N=len(s)
    for i in range(N*T):
        i = np.random.randint(N)
        h = H[i] + np.dot(J[i,:],s)
        if gamma==0:
            s[i] = int(np.random.rand()*2-1 < np.tanh(beta*h))*2-1   # Glauber
        else:
            beta1 = update_beta1(xi,s,gamma,beta)
            tanh_g = (1-exp_g(-2*beta1*h,gamma/N))/(1+exp_g(-2*beta1*h,gamma/N))
            s[i] = int(np.random.rand()*2-1 < tanh_g)*2-1   # Glauber
    return s

numba.set_num_threads(4)
@njit(parallel=True)  
def calculate_overlap(gamma,beta,M,xi0,inds_cifar):

        # ~ I = ising(N)
        
        M0=xi0.shape[0]
        inds = np.arange(M0)
        np.random.shuffle(inds)

        inds_cifar = inds_cifar[inds[0:M]]
        xi = xi0[inds[0:M],:]
        
        
        # ~ print(xi.shape)
        # ~ J = np.einsum('ai,aj->ij',xi,xi)/N
        J = np.dot(xi.T, xi) / N
        np.fill_diagonal(J, 0.)
        # ~ I.xi = xi.copy()

        H=np.zeros(N)
        # ~ I.beta=beta
        # ~ I.gamma=gamma


        ind=0
        s = xi[ind,:].copy()
        beta1 = update_beta1(xi,s,gamma,beta)

        T=N//25
        T=N//100
        s = SequentialGlauberStep(s, H, J, beta,gamma, xi,T)
        # ~ overlap = np.einsum('ai,i->a',xi,s)/N
        overlap = np.dot(xi, s) / N
        return overlap[ind]

L=32
N=L*L*3

ms=np.zeros((G,nM,R))
m=np.zeros((G,nM))
m2=np.zeros((G,nM))
m1=np.zeros((G,nM))

ig = 0
#for ig,gamma in enumerate(gammas[0:6]):
#for ig,gamma in enumerate(gammas[6:11]):
#for ig,gamma in enumerate(gammas[11:16]):

data=np.load('data/cifar-patterns-200.npz')
xi0=data['xi']
inds_cifar0=(data['inds']).astype(int)
# ~ for ig,gamma in enumerate(gammas):
    # ~ if ig in range(6):
##    if ig in range(6,11):
#    if ig in range(11,16):
#    if ig in range(16,G):
for im,M in enumerate(Ms):
    for ig,gamma in enumerate(gammas):
        for r in range(R):
            gamma=np.round(gamma,6)
            m_ = calculate_overlap(gamma,beta,M,xi0.copy(),inds_cifar0.copy())
            m[ig,im] += m_/R
            m2[ig,im] += m_**2/R
            ms[ig,im,r] += m_
            
        print(gamma,M/N,m[ig,im])

        # ~ filename = 'data/mem_capacity_cifar_beta=' + str(beta) + '_G=' + str(G) + '_nM_' + str(nM) + '_ig_' + str(ig) + '_R_' + str(R) + '.npz'
        # ~ np.savez(filename, m=m[ig,:])
    
filename = 'data/Fig5/mem_capacity_cifar_beta=' + \
    str(beta) + '_nM_' + str(nM) + '_G_' + str(G) + '_R_' + str(R) + '.npz'
np.savez(filename, m=m,m2=m2,ms=ms)
    
plt.figure()
plt.imshow(m, extent=[Ms[0]/N,Ms[-1]/N,gammas[0],gammas[-1]],aspect='auto',interpolation="none",origin="lower")
plt.colorbar()

plt.figure()
plt.imshow(m2-m**2, extent=[Ms[0]/N,Ms[-1]/N,gammas[0],gammas[-1]],aspect='auto',interpolation="none",origin="lower")
plt.colorbar()
plt.show()
