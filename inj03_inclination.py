import numpy as np
print 'using numpy version ', np.__version__

import lal
import lalsimulation as lalsim

import numexpr as ne
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.stats as stats
from scipy import signal
from scipy.optimize import newton
import sys

from astropy.table import Table, Column
import astropy.time as at

from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import ilwd

#import gwpy.timeseries as ts
#import gwpy

#from pylal import antenna as ant 


Ninj = 10


AmpOrder = -1
SpinOrder = -1
TidalOrder = 0
PhaseOrder = -1

nonGRparams = None

Approximant = lalsim.IMRPhenomPv2

def lalsim_wf_oldConv(f, m1, m2, D, phi_c, s1, s2, iota, nGRparam=None, A_order=0, P_order= -1,  \
                      f_ref=100.0, approx=lalsim.IMRPhenomPv2):
    
        dist = D * 1.0e6 * lal.PC_SI
        f_low = f[0]
        f_max = f[-1]
        df = f[1]-f[0]
        dt = 1./(2*f_max)
        z = 0.0
        lambda1 = 0.0
        lambda2 = 0.0
        waveFlags = None

        hplus, hcross = lalsim.SimInspiralFD(phi_c, df,\
                                            m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,\
                                            s1[0], s1[1], s1[2],\
                                            s2[0], s2[1], s2[2],\
                                            f_low, f_max, f_ref,\
                                            dist, z, iota,\
                                            lambda1, lambda2, waveFlags, nonGRparams,\
                                            A_order, P_order, approx)
        
        
        h_p = hplus.data.data
        h_c = hcross.data.data
        
        
 
        return h_p[int(f_low/df):-1], h_c[int(f_low/df):-1]

def innprod(h1, h2, freq, Sn, fLow=20.0, fHigh=2048):
    f_c = np.logical_and(np.greater_equal(freq, fLow), np.less_equal(freq, fHigh))
    
    h1c = h1[f_c]
    h2c = h2[f_c]
    freqc = freq[f_c]
    Snc = Sn[f_c]
    over = ne.evaluate('h1c*complex(h2c.real, -h2c.imag)/Snc')
    
    integral = integrate.simps(over, freqc)
    return 4.*integral.real

def chi_eff(m1, a1, tilt1, m2, a2, tilt2):
    return ((m1*a1*np.cos(tilt1))+ (m2*a2*np.cos(tilt2)))/(m1+m2)

def chi_precessing(m1, a1, tilt1, m2, a2, tilt2):
    q_inv = m1/m2
    A1 = 2. + (3.*q_inv/2.)
    A2 = 2. + 3./(2.*q_inv)
    S1_perp = a1*np.sin(tilt1)*m1*m1
    S2_perp = a2*np.sin(tilt2)*m2*m2
    Sp = np.maximum(A1*S2_perp, A2*S1_perp)
    chi_p = Sp/(A2*m1*m1)
    return chi_p

def change_spin_convention(theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, m1, m2, f_ref):
    iota, S1x, S1y, S1z, S2x, S2y, S2z = lalsim.SimInspiralTransformPrecessingNewInitialConditions(\
                                                 theta_jn, phi_jl, tilt1, tilt2, phi12, a1, a2, \
                                                 m1*lal.MSUN_SI, m2*lal.MSUN_SI, f_ref)
    
    spin1 = [S1x, S1y, S1z]
    spin2 = [S2x, S2y, S2z]
    
    return spin1, spin2, iota

def mc2ms(mc,eta):
    """
    Utility function for converting mchirp,eta to component masses. The
    masses are defined so that m1>m2. The rvalue is a tuple (m1,m2).
    """
    root = np.sqrt(0.25-eta)
    fraction = (0.5+root) / (0.5-root)
    invfraction = 1./fraction

    m2= mc * np.power((1+fraction),0.2) / np.power(fraction,0.6)

    m1= mc* np.power(1+invfraction,0.2) / np.power(invfraction,0.6)
    return np.array([m1,m2])

def coalescence_time(N, t_min=1135136606., t_max=1135140196.):
    return np.random.uniform(t_min, t_max, N)

def uniform_in_cos_angle(N,costheta_min=-1, costheta_max=1, offset=0.):
    return np.arccos(np.random.uniform(costheta_min,costheta_max,N)) + offset

def uniform_in_angle(N,theta_min=0., theta_max=2.*np.pi, offset=0.):
    return np.random.uniform(theta_min,theta_max,N) + offset

def compute_inj_SNR(IFOs, inj_params, freq, print_only=False):
    
    networkSNRsquared = 0.
    
    s1, s2, iota = change_spin_convention(inj_params['theta_jn'], inj_params['phi_jl'], \
                                          inj_params['tilt1'], inj_params['tilt1'], inj_params['phi12'], \
                                          inj_params['a1'], inj_params['a2'], \
                                          inj_params['mass1'], inj_params['mass2'], \
                                          inj_params['f_ref'])
    
    hplus, hcross = lalsim_wf_oldConv(freq, inj_params['mass1'], inj_params['mass2'], inj_params['distance'], \
                                          inj_params['phase'], s1, s2, iota, nGRparam=nonGRparams, A_order=AmpOrder, \
                                          P_order=PhaseOrder, f_ref=inj_params['f_ref'], approx=Approximant)
    
    for ifo in IFOs:
        ifoSNR = compute_detector_SNR(freq, hplus, hcross, ifo, \
                                      inj_params['time'], inj_params['ra'],\
                                      inj_params['dec'], inj_params['psi'])
        if print_only:
            print ifo, ifoSNR, 
        else:
            inj_params[ifo+'_SNR'] = ifoSNR
        networkSNRsquared += np.square(ifoSNR)
    if print_only:
        print np.sqrt(networkSNRsquared)
    else:
        inj_params['Network_SNR'] = np.sqrt(networkSNRsquared)
    
    
def compute_detector_SNR(freq, h_plus, h_cross, IFO, gps_time, RA, dec, psi):
    
    if IFO == 'H':
        IFO_diff = lal.LALDetectorIndexLHODIFF
        IFO_PSD = H1_psd
    elif IFO == 'L':
        IFO_diff = lal.LALDetectorIndexLLODIFF    
        IFO_PSD = L1_psd
    elif IFO == 'V':
        IFO_diff = lal.LALDetectorIndexVIRGODIFF
        IFO_PSD = V1_psd
    else:
        print 'Not a deifined detector'
        sys.exit(1)
        
    IFO_cached = lal.CachedDetectors[IFO_diff]
    fancy_time = lal.lal.LIGOTimeGPS(gps_time)
    gmst = lal.GreenwichMeanSiderealTime(fancy_time)
    timedelay = lal.TimeDelayFromEarthCenter(IFO_cached.location, RA, dec, fancy_time)
    fancy_timedelay = lal.lal.LIGOTimeGPS(timedelay)
    F_plus, F_cross = lal.ComputeDetAMResponse(IFO_cached.response, RA, dec, psi, gmst)
    
    timeshift = fancy_timedelay.gpsSeconds + 1e-9*fancy_timedelay.gpsNanoSeconds - 2.
    
    timeshift_vector = np.exp(-2.*1j*np.pi*timeshift*freq)
    
    h = ((F_plus*h_plus) + (h_cross*F_cross))*timeshift_vector[:h_cross.shape[0]]
            
    return np.sqrt(innprod(h, h, freq[:h_cross.shape[0]], IFO_PSD[:h_cross.shape[0]]))


GW170817_injections = Table()

#numbers taken as maxL values from https://geo2.arcca.cf.ac.uk/~spxcjh/LVC/GW170817/PEpaper/IMRPhenomPv2NRTidal_FixSky_lowSpin/lalinferencemcmc/IMRPhenomPv2_NRTidal/1187008882.45-0/V1H1L1/posplots.html
#but setting the tidal deformability parameters to 0

GW170817_injections['mchirp'] = np.ones(Ninj)*1.19750324719
GW170817_injections['eta'] = np.ones(Ninj)*0.249989635133

GW170817_injections['mass1'], GW170817_injections['mass2'] = mc2ms(GW170817_injections['mchirp'],GW170817_injections['eta'])

GW170817_injections['mtotal'] = GW170817_injections['mass1'] + GW170817_injections['mass2']
GW170817_injections['q'] = GW170817_injections['mass2']/GW170817_injections['mass1']

GW170817_injections['a1'] = np.ones(Ninj)*0.0398775638723
GW170817_injections['a2'] = np.ones(Ninj)*0.0228928075628
GW170817_injections['tilt1'] = np.ones(Ninj)*2.18320591417
GW170817_injections['tilt2'] = np.ones(Ninj)*0.888446053996
GW170817_injections['phi_jl'] = np.ones(Ninj)*3.7126027287
GW170817_injections['phi12'] = np.ones(Ninj)*3.1094349562

GW170817_injections['phase'] = np.ones(Ninj)*0.561403483023

Start_time = 1187007040. + 256. #The LOSC frames https://dcc.ligo.org/P1700349/public start at GPStime=1187007040, add some buffer to this so that the injections don't are effected by this
Stop_time  = 1187007040. + 1842.43 - 256. # GW170817 coalesces 1842.43s after the start of the LOSC frames, add some buffer before this  as well

GW170817_injections['time'] = np.ones(Ninj)*Start_time

GW170817_injections['theta_jn'] = np.arange(Ninj)*np.pi/2./Ninj
GW170817_injections['costheta_jn'] = np.cos(GW170817_injections['theta_jn'])

GW170817_injections['psi'] = np.ones(Ninj)*1.16419856771

#Hold the point fixed with respect to the detectors

time_sid_day = 86164.09

GW170817_injections['ra'] = -0.185539506068
GW170817_injections['dec'] = np.ones(Ninj)*(-0.935559373817)

GW170817_injections['f_ref'] = np.ones(Ninj)*100.0


GW170817_injections['H_SNR'] = np.zeros(Ninj)
GW170817_injections['L_SNR'] = np.zeros(Ninj)
GW170817_injections['V_SNR'] = np.zeros(Ninj)
GW170817_injections['Network_SNR'] = np.zeros(Ninj)


GW170817_injections['distance'] = np.ones(Ninj)*100.0

scaling = 20/24.
seglen = 128.*scaling
fseries = np.arange(20.0/scaling, (4096.0/scaling) + 1/seglen, 1/seglen)

H1_asd = np.genfromtxt('/home/tyson/O2/triggers/GW170817/C00_HLV_LALInference_PEPaper/H1_Cleaned_LALInference/BayesWave_PSD_H1_IFO0_asd_median.dat')
L1_asd = np.genfromtxt('/home/tyson/O2/triggers/GW170817/C00_HLV_LALInference_PEPaper/L1_Deglitched_Cleaned_LALInference/BayesWave_PSD_L1_IFO0_asd_median.dat')
V1_asd = np.genfromtxt('/home/tyson/O2/triggers/GW170817/C00_HLV_LALInference_PEPaper/V1_Cleaned_LALInference/BayesWave_PSD_V1_IFO0_asd_median.dat')

H1_psd = np.interp(fseries, H1_asd[:,0], np.square(H1_asd[:,1]), left=np.inf, right=np.inf)
L1_psd = np.interp(fseries, L1_asd[:,0], np.square(L1_asd[:,1]), left=np.inf, right=np.inf)
V1_psd = np.interp(fseries, V1_asd[:,0], np.square(V1_asd[:,1]), left=np.inf, right=np.inf)

ifos = ['H', 'L', 'V']

Target_SNR = 33.364307901

for i in xrange(0,Ninj):
   print i,

   compute_inj_SNR(ifos, GW170817_injections[i], fseries)
   print GW170817_injections['Network_SNR'][i],
 
   while np.logical_or(np.less_equal(GW170817_injections['Network_SNR'][i], Target_SNR - 0.05), \
                      np.greater_equal(GW170817_injections['Network_SNR'][i], Target_SNR + 0.05)):
       new_distance = GW170817_injections['distance'][i]*GW170817_injections['Network_SNR'][i]/Target_SNR

       GW170817_injections['distance'][i] = new_distance

       compute_inj_SNR(ifos, GW170817_injections[i], fseries)
       print GW170817_injections['Network_SNR'][i],
   print '\n'

now = at.Time.now()
now.format = 'gps'

savename = 'GW170817_injections_'+str(int(now.value))+'.dat'

GW170817_injections.write(savename, format='ascii')

print savename


approx_to_use = 'IMRPhenomPv2pseudoFourPN'
amp_order_to_use = -1

sim_inspiral_dt = [
        ('waveform','|S64'),
        ('taper','|S64'),
        ('f_lower', 'f8'),
        ('mchirp', 'f8'),
        ('eta', 'f8'),
        ('mass1', 'f8'),
        ('mass2', 'f8'),
        ('geocent_end_time', 'f8'),
        ('geocent_end_time_ns', 'f8'),
        ('distance', 'f8'),
        ('longitude', 'f8'),
        ('latitude', 'f8'),
        ('inclination', 'f8'),
        ('coa_phase', 'f8'),
        ('polarization', 'f8'),
        ('spin1x', 'f8'),
        ('spin1y', 'f8'),
        ('spin1z', 'f8'),
        ('spin2x', 'f8'),
        ('spin2y', 'f8'),
        ('spin2z', 'f8'),
        ('amp_order', 'i4'),
        ('numrel_data','|S64')
]

injections = np.zeros((Ninj,), dtype=sim_inspiral_dt)

ids = range(Ninj)

spin1 = np.zeros([Ninj,3])
spin2 = np.zeros([Ninj,3])
iota = np.zeros([Ninj])

for i in xrange(Ninj):
    spin1[i], spin2[i], iota[i] = change_spin_convention(GW170817_injections['theta_jn'][i], GW170817_injections['phi_jl'][i], \
                                          GW170817_injections['tilt1'][i], GW170817_injections['tilt1'][i], \
                                          GW170817_injections['phi12'][i], \
                                          GW170817_injections['a1'][i], GW170817_injections['a2'][i], \
                                          GW170817_injections['mass1'][i], GW170817_injections['mass2'][i], \
                                          GW170817_injections['f_ref'][i])

s1x = spin1[:,0]
s1y = spin1[:,1]
s1z = spin1[:,2]
s2x = spin2[:,0]
s2y = spin2[:,1]
s2z = spin2[:,2]

fLow = np.ones(Ninj)*fseries[0]

# Populate structured array
injections['waveform'] = [approx_to_use for i in xrange(Ninj)]
injections['taper'] = ['TAPER_NONE' for i in xrange(Ninj)]
injections['f_lower'] = fLow
injections['mchirp'] = GW170817_injections['mchirp']
injections['eta'] = GW170817_injections['eta']
injections['mass1'] = GW170817_injections['mass1']
injections['mass2'] = GW170817_injections['mass2']
injections['geocent_end_time'] = np.modf(GW170817_injections['time'])[1]
injections['geocent_end_time_ns'] = np.modf(GW170817_injections['time'])[0] * 10**9
injections['distance'] = GW170817_injections['distance']
injections['longitude'] = GW170817_injections['ra']
injections['latitude'] = GW170817_injections['dec']
injections['inclination'] = iota
injections['coa_phase'] = GW170817_injections['phase']
injections['polarization'] = GW170817_injections['psi']
injections['spin1x'] = s1x
injections['spin1y'] = s1y
injections['spin1z'] = s1z
injections['spin2x'] = s2x
injections['spin2y'] = s2y
injections['spin2z'] = s2z
injections['amp_order'] = [amp_order_to_use for i in xrange(Ninj)]
injections['numrel_data'] = [ "" for _ in xrange(Ninj)]

# Create a new XML document
xmldoc = ligolw.Document()
xmldoc.appendChild(ligolw.LIGO_LW())
sim_table = lsctables.New(lsctables.SimInspiralTable)
xmldoc.childNodes[0].appendChild(sim_table)

# Add empty rows to the sim_inspiral table
for inj in xrange(Ninj):
    row = sim_table.RowType()
    for slot in row.__slots__: setattr(row, slot, 0)
    sim_table.append(row)

# Fill in IDs
for i,row in enumerate(sim_table):
    row.process_id = ilwd.ilwdchar("process:process_id:{0:d}".format(i))
    row.simulation_id = ilwd.ilwdchar("sim_inspiral:simulation_id:{0:d}".format(ids[i]))

# Fill rows
for field in injections.dtype.names:
    vals = injections[field]
    for row, val in zip(sim_table, vals): setattr(row, field, val)

# Write file
output_file = open(savename.rstrip('.dat')+'.xml', 'w')
xmldoc.write(output_file)
output_file.close()

print "xml written!"






