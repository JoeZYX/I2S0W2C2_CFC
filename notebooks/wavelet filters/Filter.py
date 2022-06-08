import pywt
import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as F
from sklearn.cluster import KMeans

def PrepareWavelets(length=20):
    motherwavelets = []
    for family in pywt.families():
        for mother in pywt.wavelist(family):
            motherwavelets.append(mother)
    
    X = np.zeros([1,length])
    PSI = np.zeros([1,length])
    for mw_temp in motherwavelets:
        if mw_temp.startswith('gaus') or mw_temp.startswith('mexh') or mw_temp.startswith('morl') or mw_temp.startswith('cmor') or mw_temp.startswith('fbsp') or mw_temp.startswith('shan') or mw_temp.startswith('cgau'):
            pass
        else:
            param = pywt.Wavelet(mw_temp).wavefun(level=7)
            psi, x = param[1], param[-1]

            # normalization
            psi_sum = np.sum(psi)
            if np.abs(psi_sum) > 1:
                psi = psi / np.abs(psi_sum)
            x = x / max(x)

            # down sampling
            idx_ds = np.round(np.linspace(0, x.shape[0]-1, length)).astype(int)
            x = x[idx_ds]
            psi = psi[idx_ds]

            X = np.vstack((X, x.reshape(1,-1)))
            PSI = np.vstack((PSI, psi.reshape(1,-1)))

    X = X[1:,:]
    PSI = PSI[1:,:]

    # clustering
    FRE = np.zeros([1,length])
    for i in range(PSI.shape[0]):
        FRE = np.vstack((FRE, np.real(F.fft(PSI[i,:])).reshape(1,-1)))
    FRE = FRE[1:,:]
    PSI_extended = np.hstack((PSI, FRE))

    silhouette_result = np.zeros([2,1]).reshape(2,1)
    for k in range(5, 50):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(PSI_extended)
        label = kmeans.labels_
        silhouette_avg = silhouette_score(PSI_extended, label)
        silhouette_result = np.hstack((silhouette_result, np.array([k,silhouette_avg]).reshape(2,1)))
    
    best_idx = np.argmax(silhouette_result[1,:])
    best_K = int(result[0,best_idx])
    kmeans = KMeans(n_clusters=best_K, random_state=0).fit(PSI_extended)
    label = kmeans.labels_    

    # select one from each cluster
    SelectedWavelet = np.zeros([1,length])
    for k in range(best_K):
        wavesidx = np.where(label==k)[0][0]
        SelectedWavelet = np.vstack((SelectedWavelet, PSI[wavesidx,:]))            

    return torch.tensor(SelectedWavelet[1:,:])


def FiltersExtention(Filters):
    K, WS = Filters.shape
    
    N_ds = int(torch.log2(torch.tensor(WS-1)).floor()) - 2
    N_padding = int((WS-1)/2)
    
    Filter_temp = Filters.repeat(N_ds,1,1)
    m = torch.nn.ConstantPad1d(N_padding, 0)
    
    for n_ds in range(N_ds-1):
        filter_temp = Filter_temp[n_ds,:,:]
        # zero padding
        filter_temp_pad = m(filter_temp)
        # down sampling
        filter_ds = filter_temp_pad[:,::2]
        # save wavelets
        Filter_temp[n_ds+1,:,:] = filter_ds
    
    # formualte dimensionality
    Filter_temp = Filter_temp.view(K*N_ds,WS)
    Filter_temp = Filter_temp.repeat(1,1,1,1)
    Filter_temp = Filter_temp.permute(2,0,1,3)
    
    # normalization
    energy = torch.abs(torch.sum(Filter_temp, dim=3, keepdims=True))
    energy[energy<=1] = 1.
    Filter_temp = Filter_temp / energy
    return Filter_temp