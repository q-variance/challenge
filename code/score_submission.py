import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, poisson
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

# load the parquet files from data_loader.py

if os.path.isfile("dataset.parquet"):
    df= pd.read_parquet("dataset.parquet")  # READ SUBMISSION DATA
else:
    df = pd.concat([pd.read_parquet("dataset_part1.parquet"),pd.read_parquet("dataset_part2.parquet"),pd.read_parquet("dataset_part3.parquet")])


df["var"] = df.sigma**2
TICKERS1 = np.unique(df["ticker"])  # use original tickers

print(f"{len(df)} windows")
print(f"z has NaNs: {df['z'].isna().sum()}")  # → 0

df_orig = df.copy()

# divide dataframe into equal segments and rename 'ticker' column based on segment number
# note that parquet file has ticker on outer loop so this is like selecting certain stocks
def assign_segmented_tickers(df,N_SEGMENTS):
    nr = len(df)
   
    rows_per_segment = nr // N_SEGMENTS
    nextra = nr % N_SEGMENTS
    
    # Drop extra rows if necessary
    if nextra > 0:
        df = df.iloc[:nr - nextra].copy()
        nr = len(df)
    
    # Create an array of segment numbers
    segment_ids = np.arange(1, N_SEGMENTS + 1).repeat(rows_per_segment)
    
    # Use list comprehension to handle string concatenation in standard Python, 
    # then convert back to a NumPy array if needed, or just assign directly.
    new_tickers = ['V' + str(i) for i in segment_ids]
    
    # Assign the new ticker names to the 'ticker' column
    df['ticker'] = new_tickers
    
    return df

zmax = 0.6  #0.6 or 1
delz = 0.025*2
nbins = int(2*zmax/delz + 1)
bins = np.linspace(-zmax, zmax, nbins)         # fixed bins
ymax = 0.35

# create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
binned = (df.assign(z_bin=pd.cut(df.z, bins=bins, include_lowest=True))
               .groupby('z_bin',observed=False)
               .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
               .dropna())

def qvar(z, s0, zoff):    # define q-variance function, parameter is minimal volatility s0 and zoff
    return (s0**2 + (z - zoff)**2 / 2)

def qvar2(z, s0, s1):    # define q-variance function, parameter is minimal volatility s0 and qcoef
    return (s0**2 + s1*z**2 )

# curve_fit returns a value popt and a covariance pcov, the _ means we ignore the pcov
popt = [0.2586, 0.0214]  # same as optimized fit to data
popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.2, 0])  # fit this data

fitted = qvar(binned.z_mid, popt[0], popt[1])  # cols are z_bin, which is a range like (-0.601, -0.55], and qvar
r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)

print(f"σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")

# plot of all stocks
markfac = 1  # default is 1, can increase to 3 if less data points
plt.figure(figsize=(9,7))
plt.scatter(df.z, df['var'], c='steelblue', alpha=markfac*0.1, s=markfac*1, edgecolor='none')
numeric_array = (1 - df["T"]/130)
string_array = [str(x) for x in numeric_array]
plt.plot(binned.z_mid, binned['var'], 'b-', lw=3)     # label='binned'
plt.plot(binned.z_mid, fitted, 'red', lw=3, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')

plt.xlabel('z (scaled log return)', fontsize=12)
plt.ylabel('Annualised variance', fontsize=12)
plt.title('Q-Variance: all data T=1 to 26 weeks', fontsize=14)

plt.xlim(-zmax, zmax) 
plt.ylim(0.0, ymax)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
##plt.savefig("figure_1_1.png", dpi=300, bbox_inches='tight')


normcol = mcolors.Normalize(vmin=0, vmax=0.3)
cmap = cm.get_cmap('coolwarm')


# now change the fit for all and get coefficients qcopt for case T=5 ONLY
zmax = 1  # larger grid
ymax = 0.8

delz = 0.025*4  # wider bin for these noisy plots
nbins = int(2*zmax/delz + 1)
bins = np.linspace(-zmax, zmax, nbins)         # fixed bins

if len(np.unique(df_orig["ticker"])) == 1:     # set to >=1 to divide all data, or == 1 to divide model data only ###
    print("dividing into 500 separate runs") 
    df = assign_segmented_tickers(df_orig,500)  # divide and add tickers
else:
    print("data already divided into separate tickers") 
    df = df_orig


TICKERS = np.unique(df["ticker"])
r2vec = np.zeros(len(TICKERS))
q0vec = np.zeros(len(TICKERS))
q1vec = np.zeros(len(TICKERS))
i = 0
fig, ax = plt.subplots(figsize=(9,7))
for tickcur in TICKERS:
    dfcur = df[ (df["ticker"] == tickcur)  ].copy()   # & (df["T"] == 5)
    binned = (dfcur.assign(z_bin=pd.cut(dfcur.z, bins=bins, include_lowest=True))
                   .groupby('z_bin',observed=False)
                   .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
                   .dropna())

    popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.2, 0])  # custom fit
    qopt, _ = curve_fit(qvar2, binned.z_mid, binned["var"], p0=[0.2, 0.5])  # custom fit to quadratic term
    fitted = qvar(binned.z_mid, popt[0], popt[1])    # use fit for whole data set
    r2vec[i] = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)
    q0vec[i] = qopt[0]
    q1vec[i] = qopt[1]
    #print(f"ticker = {tickcur} σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")
    colcur = cmap(normcol(max(0,popt[0])))   # str(Tcur/100)
    ax.plot(binned.z_mid, binned['var'], '-', c=colcur, alpha = 0.5, lw=2, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2vec[i] :.3f}') 
    i = i+1

r2mean = np.mean(r2vec)
r2median = np.median(r2vec)
ax.set_xlabel('z (scaled log return)', fontsize=12)
ax.set_ylabel('Annualised variance', fontsize=12)
ax.set_title(f'Mean R2 for individual stocks = {r2mean:4f},  median = {r2median:4f}', fontsize=14)
#ax.legend(fontsize=12)

ymax2 = 1.2
ax.axis([-zmax+delz/2, zmax-delz/2, 0, ymax2])   # uses ymax2 for ensemble plot



# now check for time-invariant distribution
data = df_orig.copy()

# Quantum density function — returns plain array for curve_fit
def quantum_density(z, sig0, zoff=0.0):
    ns = np.arange(0, 6)
    qdn = np.zeros_like(z, dtype=float)
    sigvec = sig0 * np.sqrt(2 * ns + 1)
    means = zoff * np.ones_like(ns)  - sigvec**2/2 # no drift term in pure Q-Variance

    for n in ns:
        weight = poisson.pmf(n, mu=0.5)
        qdn += weight * norm.pdf(z, loc=means[n], scale=sigvec[n])
    return qdn

# Plot setup
zlim = 2
zbins = np.linspace(-zlim, zlim, 51)
zmid = (zbins[:-1] + zbins[1:]) / 2

# Histogram
counts, _ = np.histogram(data["z"], bins=zbins, density=True)

# Fit quantum model
p0 = [0.62, 0.0]  # initial guess: sig0 ≈ 0.62 → σ₀ ≈ 0.079 after √2 scaling
popt, _ = curve_fit(quantum_density, zmid, counts, p0=p0, bounds=(0, [2.0, 0.5]))
sig0_fit, zoff_fit = popt

# Predict on fine grid
z_fine = np.linspace(-zlim, zlim, 1000)
q_pred_fine = quantum_density(z_fine, *popt)

# Predict on histogram bin centers for R²
q_pred_hist = quantum_density(zmid, *popt)
r2 = r2_score(counts, q_pred_hist)

print(f"Fit: σ₀ = {sig0_fit:.4f}, zoff = {zoff_fit:.4f}, R² = {r2:.4f}")

# obtain histogram bars
##counts, bin_edges, _ = plt.hist(data["z"], bins=zbins, density=True, visible=False)



# now plot distn with periods

TVEC = np.unique(data["T"])  # case where T=5 only gives TVEC=5
if len(TVEC) > 1:
    TVEC = [5, 10, 20, 40, 80]


plt.figure(figsize=(9,7))
plt.plot(z_fine, q_pred_fine,
         color='red', lw=4,
         label=f'Q-Variance fit: σ₀ = {sig0_fit:.3f}, R² = {r2:.4f}')

for Tcur in TVEC:
    datacur = data[(data["T"] == Tcur)].copy()
    counts, bin_edges, _ = plt.hist(datacur["z"], bins=zbins, density=True, visible=False)
    r2 = r2_score(counts, q_pred_hist)    # use fit for whole data set
    colcur = str(Tcur/(max(TVEC)+20))
    #plt.plot(zmid, counts, c=colcur, lw=2,label=f'T = {Tcur/5:.0f}')  # , R² = {r2:.3f}' 
    plt.plot(zmid, counts, c=colcur, lw=2,label=f'T = {Tcur/5:.0f}, R² = {r2:.3f}' )

plt.title('Q-Variance: T dependence', fontsize=18, pad=20)
plt.xlabel('Scaled log-return z', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.xlim(-1.2, 1.2)
plt.legend(fontsize=10, loc='upper right')
plt.grid(alpha=0.3)
plt.tight_layout()


plt.show()

