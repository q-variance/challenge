# baseline_fit.py
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# load the parquet files from data_loader.py
df = pd.concat([
    pd.read_parquet("dataset_part1.parquet"),
    pd.read_parquet("dataset_part2.parquet"),
    pd.read_parquet("dataset_part3.parquet")
])

# Select S&P 500, T=5
# data = df[(df["ticker"] == "^GSPC") & (df["T"] == 5)].copy()
# data = df[(df["ticker"] == "^GSPC") ].copy()
data = df.copy()
data["var"] = data.sigma**2

#print(f"S&P 500 T=5: {len(data)} windows")
print(f"z has NaNs: {data['z'].isna().sum()}")  # → 0

zmax = 0.6
delz = 0.025
nbins = int(2*zmax/delz + 1)
#bins = np.linspace(-0.5, 0.5, 41)         # fixed bins
bins = np.linspace(-zmax, zmax, nbins)         # fixed bins

# create data frame with e.g. zbin = (-0.601, -0.55], z_mid, sigma
binned = (data.assign(z_bin=pd.cut(data.z, bins=bins, include_lowest=True))
               .groupby('z_bin',observed=False)
               .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
               .dropna())

# zmid = (bins[0:(nbins-1)] + bins[1:(nbins)])/2

def qvar(z, s0, zoff):    # define q-variance function, parameter is minimal volatility s0
    return (s0**2 + (z - zoff)**2 / 2)

# curve_fit returns a value popt and a covariance pcov, the _ means we ignore the pcov
popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.02, 0])

fitted = qvar(binned.z_mid, popt[0], popt[1])  # cols are z_bin, which is a range like (-0.601, -0.55], and qvar
r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)

print(f"σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")

# plot of all stocks
plt.figure(figsize=(9,7))
plt.scatter(data.z, data['var'], c='steelblue', alpha=0.1, s=1, edgecolor='none')
plt.plot(binned.z_mid, binned['var'], 'b-', lw=3)     # label='binned'
plt.plot(binned.z_mid, fitted, 'red', lw=4, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')

plt.xlabel('z (scaled log return)', fontsize=12)
plt.ylabel('Annualised variance', fontsize=12)
plt.title('All data T=1 to 26 weeks – Q-Variance', fontsize=14)

plt.xlim(-zmax, zmax) 
plt.ylim(0.0, 0.35)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
#plt.show()

# now do plot with periods
TVEC = [10, 20, 40, 80,]

plt.figure(figsize=(9,7))
#plt.plot(binned.z_mid, binned['var'], 'b-', lw=3,label='all T')  
plt.plot(binned.z_mid, fitted, 'red', lw=4, label=f'σ₀ = {popt[0]:.3f}, zoff = {popt[1]:.3f}, R² = {r2:.3f}')

for Tcur in TVEC:
    datacur = data[(data["T"] == Tcur)].copy()
    binned = (datacur.assign(z_bin=pd.cut(datacur.z, bins=bins, include_lowest=True))
                   .groupby('z_bin',observed=False)
                   .agg(z_mid=('z', 'mean'), var=('var', 'mean'))
                   .dropna())

    # popt, _ = curve_fit(qvar, binned.z_mid, binned["var"], p0=[0.02, 0])  # custom fit
    fitted = qvar(binned.z_mid, popt[0], popt[1])    # use fit for whole data set
    r2 = 1 - np.sum((binned["var"] - fitted)**2) / np.sum((binned["var"] - binned["var"].mean())**2)
    print(f"T = {Tcur} σ₀ = {popt[0]:.4f}  zoff = {popt[1]:.4f}  R² = {r2:.4f}")
    colcur = str(Tcur/100)
    plt.plot(binned.z_mid, binned['var'], c=colcur, lw=2,label=f'T = {Tcur/5:.0f}, R² = {r2:.3f}') 

plt.xlabel('z (scaled log return)', fontsize=12)
plt.ylabel('Annualised variance', fontsize=12)
plt.title('All stocks T=2, 4, 8, 16 weeks – Q-Variance', fontsize=14)

plt.xlim(-zmax, zmax) 
plt.ylim(0.0, 0.35)

plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# Optional: save as high-res PNG/PDF for the announcement
# plt.savefig("q_variance_8_assets.png", dpi=300, bbox_inches='tight')
# plt.savefig("q_variance_8_assets.pdf", bbox_inches='tight')



