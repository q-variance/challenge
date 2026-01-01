## ===================== ALL-IN-ONE (R): NO JUMPS, but Gaussian macro-moves executed in 2 steps + cooldown => beta2(T) ~ 0.5 =====================
rm(list=ls())

## -------------------- knobs --------------------
dt      <- 1/252
nDays   <- 10000L
seed    <- NULL
set.seed(seed)

S0 <- 100

## baseline diffusion (KEEP TINY or 0, otherwise beta2 -> 0 for large T)
mu_ann     <- 0.01
sigma_ann  <- 0.15
sigma_base <- 0.1        # <= set small (e.g. 0.01) if you want some noise

## 2-step "execution" of a Gaussian macro-move (NOT a jump: still Normal, just executed over 2 days)
p_exec    <- 1/800        # frequency of macro-moves
U_scale   <- 8.0          # strength relative to sigma_day; raise if beta2 < 0.5
cooldown  <- 1L           # gap after an execution (prevents overlap)

## q-variance estimation
hPlot   <- 10L
hGrid   <- seq(5L, 252L, by=5L)
TGrid   <- hGrid * dt
capProb <- 1L ## No
minN    <- 800L

## -------------------- helpers --------------------
cap_xy <- function(z, v, capProb) {
  ok <- is.finite(z) & is.finite(v) & (v > 0)
  z <- z[ok]; v <- v[ok]
  if (!length(z)) return(list(z=z, v=v))
  zCap <- unname(quantile(abs(z), probs=capProb, names=FALSE))
  vCap <- unname(quantile(v,     probs=capProb, names=FALSE))
  keep <- (abs(z) <= zCap) & (v <= vCap)
  list(z=z[keep], v=v[keep])
}

make_windows_fast <- function(ret, h, dt, mu_hat) {
  n <- length(ret); h <- as.integer(h)
  T <- h * dt
  cr  <- c(0, cumsum(ret))
  cr2 <- c(0, cumsum(ret*ret))
  s <- 1L:(n - h + 1L)
  e <- s + h - 1L
  R  <- cr[e+1L]  - cr[s]
  RV <- cr2[e+1L] - cr2[s]
  x <- R - mu_hat * T
  z <- x / sqrt(T)
  Vhat <- RV / T
  list(z=z, Vhat=Vhat, T=T)
}

beta2_ols_fast <- function(z, Vhat, capProb=0.995, minN=800L) {
  cc <- cap_xy(z, Vhat, capProb)
  z <- cc$z; Vhat <- cc$v
  if (length(z) < minN) return(NA_real_)
  X <- cbind(1, z, z*z)
  b <- qr.solve(crossprod(X), crossprod(X, Vhat))
  unname(b[3L])
}

plot_qvariance <- function(z, Vhat, capProb=0.995, main="") {
  cc <- cap_xy(z, Vhat, capProb)
  z <- cc$z; Vhat <- cc$v
  fit <- lm(Vhat ~ z + I(z^2))
  b2  <- unname(coef(fit)[["I(z^2)"]])
  xg <- seq(min(z), max(z), length.out=400L)
  yg <- coef(fit)[1L] + coef(fit)[2L]*xg + coef(fit)[3L]*xg^2
  plot(z, Vhat, pch=16, cex=0.22, col=gray(0,0.18),
       xlab="z", ylab="Vhat = RV/T", main=main)
  lines(xg, yg, lwd=2)
  mtext(sprintf("beta2=%.4f", b2), side=3, adj=1, line=0.2)
  invisible(b2)
}

beta2_curve <- function(ret, mu_hat) {
  vapply(hGrid, function(h) {
    w <- make_windows_fast(ret, h, dt, mu_hat)
    beta2_ols_fast(w$z, w$Vhat, capProb=capProb, minN=minN)
  }, numeric(1))
}

## -------------------- simulator: Gaussian 2-step macro-moves + cooldown (no instantaneous jumps) --------------------
simulate_path <- function() {
  sigma_day <- sigma_ann * sqrt(dt)
  mu_day    <- (mu_ann - 0.5*sigma_ann^2) * dt
  
  ret <- numeric(nDays)
  I   <- integer(nDays)      # macro-move start indicator
  last_end <- -1e9
  t <- 1L
  nExec <- 0L
  
  while (t <= nDays) {
    if (t <= (nDays-1L) && (t > (last_end + cooldown)) && runif(1) < p_exec) {
      ## one Gaussian macro-move U executed over 2 equal steps
      U <- (U_scale * sigma_day) * rnorm(1)
      
      ret[t]   <- mu_day + 0.5 * U
      ret[t+1] <- mu_day + 0.5 * U
      
      I[t] <- 1L
      nExec <- nExec + 1L
      
      last_end <- t + 1L
      t <- last_end + 1L
    } else {
      ## outside executions: keep iid noise tiny (or 0) so beta2 doesn't wash out
      ret[t] <- mu_day + (sigma_base * sigma_day) * rnorm(1)
      t <- t + 1L
    }
  }
  
  mu_hat <- mean(ret) / dt
  S <- exp(log(S0) + cumsum(c(0, ret)))
  list(ret=ret, S=S, mu_hat=mu_hat, I=I, nExec=nExec, sigma_day=sigma_day)
}

## ===================== ONE RUN: plots =====================
sim <- simulate_path()

par(mfrow=c(3,1), mar=c(4.5,4.5,2.5,1.5))

plot(sim$S, type="l", xlab="days", ylab="Price",
     main=sprintf("Price (2-step Gaussian execution; cooldown=%d; nExec=%d)", cooldown, sim$nExec))

wP <- make_windows_fast(sim$ret, hPlot, dt, sim$mu_hat)
bP <- plot_qvariance(wP$z, wP$Vhat, capProb=capProb,
                     main=sprintf("q-variance (h=%d, T=%.3f)", hPlot, wP$T))
abline(h=0.5, lty=2)

b2_one <- beta2_curve(sim$ret, sim$mu_hat)
plot(TGrid, b2_one, type="b", pch=16, ylim=c(0,1),
     xlab="T (years)", ylab="beta2 on z^2",
     main="beta2(T) from ONE path (target ~0.5)")
abline(h=0.5, lty=2)

cat(sprintf("ONE RUN: nExec=%d | sigma_day=%.6f | U_scale=%.2f | sigma_base=%.3f | cooldown=%d\n",
            sim$nExec, sim$sigma_day, U_scale, sigma_base, cooldown))
cat(sprintf("hPlot beta2=%.4f | mean(beta2)=%.4f\n", bP, mean(b2_one, na.rm=TRUE)))
