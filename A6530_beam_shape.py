import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

def gaussian_vec(amp, mu, sigma, phivec):
    gaussvec = []
    for i in range(len(phivec)):
        gaussvec.append(amp*np.exp(-(phivec[i]-mu)**2/(2*sigma**2)))   # amp not normalized here
    return gaussvec

cf = 1.42e9  # center frequency, currently set to HI ~ 1.4 GHz
amp = 450*1000*1e7 # Arecibo transmitter is 450 kW
amp = 2e-15
fvec = np.linspace(cf-0.1e9, cf+0.1e9, 2048)
b = gaussian_vec(amp, cf, 0.02e9, fvec)

random_data = (1e-7)*np.random.uniform(0, 1, 50)
z = np.polyfit(np.linspace(-10,10,50), random_data, 10)
p = np.poly1d(z)
arr = np.random.uniform(-10, 10, 50)
#print(arr)
#p = Polynomial(arr)
alpha = np.polyval(arr, np.linspace(-5, 5, 2048)) # absorption coeff for Earth atmosphere, ~0 for HI line
alpha = p(np.linspace(-10, 10, 2048))
beta = 0.1*(6.65e-25) # absorption coeff for ISM, n = 0.1, cross-section = 6.65e-25
temp = 300 # reasonable atmospheric temp
d = 48500*3.1*10**18 # distance to exoplanet, 30 parsec to cm
z = 10000000 # height of atmosphere, 100 km to cm
c = 3e10 # speed of light
k = 1.4e-16 # boltzmann constant
h = 6.6e-27 # planck constant
ismtemp = 10

#print(alpha*z)
#print(beta*d)

#noise = np.random.uniform(-1e-8, 1e-8, 2048)
#alpha += noise

plt.plot(fvec, alpha*z, c='k')
plt.xlabel('frequency [Hz]')
plt.ylabel(r'$\alpha z$ ~ absorbance [unitless]')
plt.title('absorption spectrum (randomly generated)')
plt.savefig('sample_absorption_spectrum.pdf')
plt.show()

print(f'beta*d is {beta*d}.')

print((1-np.exp(-beta*d)))

print((1-np.exp(-beta*d))/np.exp(alpha.mean()*z))

print('==========')
print(h*(1.42e9)**3/(c**2))
print((1.42e9)**2*k*ismtemp/(c**2))
print('==========')
print(2*cf**2*k*temp/(c**2))
print(1-np.exp(-alpha.mean()*z))
print(1+np.exp(-(alpha.mean()*z+beta*d)))
print('======')
print(np.exp(-alpha.mean()*z))
print(1-np.exp(-beta*d))

print('-----------')
print(f'Size of first term is {np.max(b)*np.exp(-(2*alpha.mean()*z+beta*d))}.')
print(f'Size of second term is {(2*cf**2*k*temp/(c**2))*(1-np.exp(-alpha.mean()*z))*(1+np.exp(-(alpha.mean()*z+beta*d)))}.')
print(f'Size of third term is {(h*cf**3/(c**2))*(1-np.exp(-beta*d))/np.exp(alpha.mean()*z)}.')

#def beam_change(idx):
#    nb = b[idx]*np.exp(-(2*alpha*z+beta*d)) + (2*fvec[idx]**2*k*temp/(alpha**2*c**2))*(1-np.exp(-alpha*z))*(1+np.exp(-(alpha*z+beta*d))) + (h*fvec[idx]**3/(beta**2*c**2))*(1-np.exp(-beta*d))/np.exp(alpha*z)
#    return nb

def beam_change_exp(idx):
    eb = b[idx]*np.exp(-alpha[idx]*z) + (2*fvec[idx]**2*k*temp/(c**2))*(1-np.exp(-alpha[idx]*z))
    return eb

def beam_change_ism(idx):
    ib = b[idx]*np.exp(-(alpha[idx]*z+beta*d)) + (2*fvec[idx]**2*k*temp/(c**2))*(1-np.exp(-alpha[idx]*z))*np.exp(-beta*d) + (h*fvec[idx]**3/(c**2))*(1-np.exp(-beta*d))
    return ib

def beam_change(idx):
    nb = b[idx]*np.exp(-(2*alpha[idx]*z+beta*d)) + (2*fvec[idx]**2*k*temp/(c**2))*(1-np.exp(-alpha[idx]*z))*(1+np.exp(-(alpha[idx]*z+beta*d))) + (h*fvec[idx]**3/(c**2))*(1-np.exp(-beta*d))/np.exp(alpha[idx]*z)
    return nb

#def beam_change_2(idx):
#    mb = b[idx]*np.exp(-(2*alpha[idx]*z+beta*d)) + (2*fvec[idx]**2*k*temp/(c**2))*(1-np.exp(-alpha[idx]*z))*(1+np.exp(-(alpha[idx]*z+beta*d))) + (h*fvec[idx]**3/(c**2))*(1-np.exp(-beta*d))/np.exp(alpha[idx]*z)
#    nb = b[idx]*np.exp(-(2*alpha[idx]*z+beta*d)) + (2*fvec[idx]**2*k*temp/(c**2))*(1-np.exp(-alpha[idx]*z))*(1+np.exp(-(alpha[idx]*z+beta*d)))
#    return nb

b2 = []
for i in range(len(b)):
    j = beam_change_exp(i)
    b2.append(j)
b2 = np.array(b2)

b3 = []
for i in range(len(b)):
    j = beam_change_ism(i)
    b3.append(j)
b3 = np.array(b3)

b4 = []
for i in range(len(b)):
    j = beam_change(i)
    b4.append(j)
b4 = np.array(b4)

#plt.plot(fvec, b2)
#plt.xlabel('frequency [Hz]')
#plt.show()

#plt.plot(fvec, b)
#plt.show()

plt.plot(fvec, b, c='k', label=r'$I_\nu^{(0)}$')
plt.plot(fvec, b2, c='#547b86', label=r'$I_\nu^{(ex)}$')
plt.plot(fvec, b3, c='#abd37eff', label=r'$I_\nu^{(ISM)}$')
plt.plot(fvec, b4, c='#67c5e0', label=r'$I_\nu^\oplus$')
#plt.plot(fvec, b3, label='pulse at receiver')
plt.xlabel('frequency [Hz]')
plt.ylabel(r'intensity [erg s$^{-1}$ cm$^{-2}$]')
plt.title(r'beam shape change, $\alpha_\nu = 0$')
plt.legend()
plt.savefig('beam_shape_change.pdf')
plt.show()

#plt.plot(fvec, b3-b2)
#plt.ylim(-1e-17, 1e-17)
#plt.show()

#plt.plot(fvec, b-b2)
#plt.show()
