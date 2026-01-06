def main():

    # 3 Free Parameters
    sigma0 = 0.259
    z0 = 0.021
    xi = 0.08
    
    # Structural settings
    dt = 1/252
    t_window = 15
    burn_in = 252
    
    # subtract final shift
    final_shift = 0.025
    
    n_paths = 500
    steps = 10000 + burn_in
    output_file = "q-variance/simulated_price_logreturns_nokappa.csv"
    
    all_data = []
    print(f"Final Calibration: Shifting Intercept by -{final_shift}...")
    
    for i in range(n_paths):
        prices, log_rets = np.ones(steps)*100, np.zeros(steps)
        # Start at the target baseline
        v = max(sigma0**2 - final_shift, 0.01)
        raw_ret_buffer = np.zeros(t_window)
    
        for t in range(1,steps):
            dw1, dw2 = np.random.normal(0,1,2)
    
            # 1. Predictive Logic (Locks Curvature)
            dx_est = np.sqrt(max(v,1e-6) * dt) * dw1
            temp_buffer = raw_ret_buffer.copy()
            temp_buffer[t % t_window] = dx_est
            z_pred = np.sum(temp_buffer) / np.sqrt(t_window/252)
    
            # 2. Adjusted Target (Locks Height)
            # We apply the final nudge here to counteract the 0.05 drift.
            target_v = (sigma0**2 + 0.5 * (z_pred  -z0)**2) - final_shift
    
            # 3. Variance SDE (Locked to target)
            v = max(target_v + xi * np.sqrt(max(target_v, 1e-7) * dt) * dw2, 1e-7)
    
            # 4. Price move
            dx = np.sqrt(max(v,1e-7) * dt) * dw1
            prices[t] = prices[t-1] * np.exp(dx)
            log_rets[t] = dx
    
            # 5. Buffer update
            raw_ret_buffer[t % t_window] = dx
    
    
    
        ticker = f"TKN{i}"
        all_data.append(pd.DataFrame({
            'Date': pd.date_range(end='2025-12-24', periods=steps - burn_in),
            'Ticker': ticker,
            'Price': prices[burn_in:],
            'logreturn': log_rets[burn_in:]
        }))
    
    pd.concat(all_data).to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
