## 1. NB Model

### A. Model Structure

$$
\begin{align*}
Y_i &\sim NB(\mu_i,\alpha) \quad \text{ where }i=1,2,...,s \\
& \begin{cases} E(Y_i) = \mu_i \\
Var(Y_i) = \mu_i ( 1+\alpha \mu_i) \end{cases} 
\\

\eta_i & = X_i^t\beta \\
log(\mu_i) &= \eta_i 

\end{align*}
$$



### B. Likelihood Function

$$
\begin{align*}
f(Y \mid \beta , \alpha) & = \prod_i f(y_i \mid \mu_i,\alpha)  \\ 
&=   \prod_i \binom{y_i + 1/\alpha -1}{1/\alpha -1}\left(\frac{1}{1+\alpha \mu_i}\right)^{1/\alpha} \left( \frac{\alpha \mu_i}{1+ \alpha \mu_i} \right)^{y_i} \\ 
& = \prod_i \frac{\Gamma(y_i + 1/\alpha)}{\Gamma(y_i + 1) \Gamma(1/\alpha)} \left(\frac{1}{1+\alpha \mu_i}\right)^{1/\alpha} \left( 1- \frac{1}{1+ \alpha \mu_i} \right)^{y_i} \\ 
& = \prod_{i=1}^n \left[ \frac{1}{y_i!} \times \prod_{j=0} \left( j + 1/\alpha\right) \times (\alpha \mu_i)^{y_i} \times (1+\alpha \mu_i)^{-(y_i + 1/\alpha_i)}  \right]\\
\\ \\
l(\mu,\alpha) &= \sum_{i=1}^{n} \left[ \sum_{j=0}^{y_i -1} \log (j+ 1/\alpha) - \log y_i! + y_i \log \alpha \mu_i - ( y_i +1/\alpha ) \log (1+\alpha \mu_i)\right] \\ 
& = \sum_{i=1}^{n} \left[ \sum_{j=0}^{y_i -1} \log (j+ 1/\alpha) - \log y_i! + y_i \log \alpha + y_i x_i^t \beta - ( y_i +1/\alpha ) \log (1+\alpha \exp(x_i^t \beta))\right]  = l(\beta,\alpha)
\end{align*}
$$



### C. Gradient Function

$$
\begin{align*}
\frac{\partial l(\beta,\alpha)}{\partial \beta} & = \frac{\partial}{\partial \beta} \sum_{i=1}^n \left[ y_i x_i^t \beta - (y_i + 1/\alpha) \log (1+ \alpha \exp(x_i^t \beta)) \right]  \\
&= \sum_{i=1}^n \left( y_i x_i -(y_i +1/\alpha) \frac{\alpha \exp (x_i^t \beta)x_i}{1+\alpha \exp(x_i^t \beta)}  \right) \\
&=  \sum_{i=1}^n \left(\frac{y_i - \exp(x_i^t \beta)}{ 1+ \alpha \exp (x_i^t \beta)} \right) x_{i}^t \\
&=  \sum_{i=1}^n \left(\frac{y_i - \mu_i}{ 1+ \alpha \mu_i} \right) x_{i}^t \\

\\ \\
\frac{\partial l(\beta,\alpha)}{\partial \alpha} &= \sum_{i=1}^n \left[  -\alpha^{-2} \sum_{j=0}^{y_i-1} \frac{1}{j + \alpha ^{-1}} + \alpha^{-2} \log (1+ \alpha \mu_i)\right]
\end{align*}
$$





## 2. ZINB model

### A. Model Structure

$$
\begin{align*}
Y_i \sim &\begin{cases} 0 \text{ with probability }\psi_i \\ g(y_i) \text{ with probability } 1 - \psi_i \end{cases}

\\ \\
P(y_i=0 \mid x_i) & = \psi_i + (1-\psi_i) g(0;\mu_i) \\
P(y_i \mid x_i ) & = (1-\psi_i) g(y_i ; \mu_i) , \quad y_i > 0

\\ \\
\mu_i & = g^{-1}(x_i^t \theta) \\
logit(\psi_i) & = x_i^t \gamma \quad (\psi_i = \sigma(x_i^t\gamma))

\\ \\
Y_{ij} &= (1-S_{ij}) Z_{ij} \\
S_{ij} &\sim Bernoulli(\psi_{ij}) \\
Z_{ij} &\sim NB(\mu_{ij},\alpha_j)\\
Z_{ij} &\perp Z_{ij'}
\end{align*}
$$





## 3. CZINB (Consensus ZINB)
- Building Shared Zero model with some similar zero pattern gene sets
- High sparsity gene sets, Moderate sparsity gene sets, Low sparsity gene sets, Almost no sparsity gene sets

### A. Model Structure

$$
Y_{ij} \sim \begin{cases} 0 \text{ with prob } \psi_i  \\ g(y_{ij}) \text{ with prob } 1- \psi_i\end{cases} 

\\ \\

P(y_{ij}=0 \mid x_{i}) = \psi_i + (1-\psi_i) g( 0 ; \mu_{ij}) \\
P(y_i \mid x_i) = (1- \psi_i) g(y_{ij} ; \mu_{ij}) ,\qquad  y_{ij} > 0  

\\ \\

\mu_{ij} = g^{-1}(x_i^t \beta_j) \\
logit(\psi_i) = x_i^t \gamma \quad (\psi_i = \sigma(x_i^t\gamma))

\\ \\
Y_{ij} = (1-S_i) Z_{ij} \\
S_i \sim Bernoulli(\psi_i) \\
Z_{ij} \sim NB(\mu_{ij},\alpha_j)\\
Z_{ij} \perp Z_{ij'}
$$

### B. Likelihood Function (Core)

Define a cell-level shared-dropout indicator
$$
A_i = \mathbf{1}\{Y_{i1}=0,\dots,Y_{iG}=0\}.
$$

Under CZINB with shared zero probability $\psi_i$ and independent NB latent counts across genes:
$$
P(\mathbf{Y}_i \mid x_i)=
\begin{cases}
\psi_i + (1-\psi_i)\prod_{j=1}^G f_{NB}(0;\mu_{ij},\alpha_j), & A_i=1,\\
(1-\psi_i)\prod_{j=1}^G f_{NB}(y_{ij};\mu_{ij},\alpha_j), & A_i=0.
\end{cases}
$$

Equivalent log-likelihood contribution for cell $i$:
$$
\ell_i=
\begin{cases}
\log\left[\psi_i + (1-\psi_i)\exp\left(\sum_{j=1}^G \log f_{NB}(0;\mu_{ij},\alpha_j)\right)\right], & A_i=1,\\
\log(1-\psi_i) + \sum_{j=1}^G \log f_{NB}(y_{ij};\mu_{ij},\alpha_j), & A_i=0.
\end{cases}
$$

Total log-likelihood:
$$
\ell = \sum_{i=1}^n \ell_i.
$$

 



## 4. MZINB (Marginalized ZINB)

### A. Model Structure

$$
\begin{align*}
Y_i \sim &\begin{cases} 0 \text{ with probability }\psi_i \\ g(y_i) \text{ with probability } 1 - \psi_i \end{cases}

\\ \\
P(y_i=0 \mid x_i) & = \psi_i + (1-\psi_i) g(0;\mu_i) \\
P(y_i \mid x_i ) & = (1-\psi_i) g(y_i ; \mu_i) , \quad y_i > 0

\\ \\
logit(\psi_i) & = x_i^t \gamma \quad (\psi_i = \sigma(x_i^t\gamma)) \\
\nu_{ij} &= g^{-1}(x_i^t \beta_j) = (1-\psi_i)\mu_i = E(Y_{ij} \mid X_{ij}) \\ \\
\because E(Y_{ij} \mid X_{ij}) &=  E(E((1-S_{ij}) Z_{ij} \mid X_{ij})) \\ &= E((1-\psi_i)Z_{ij} \mid X_{ij}) = (1-\psi_i)\mu_i

\\ \\
Y_{ij} &= (1-S_{ij}) Z_{ij} \\
S_{ij} &\sim Bernoulli(\psi_{ij}) \\
Z_{ij} &\sim NB(\mu_{ij},\alpha_j)
\end{align*}
$$




## 5. Test for CZINB

### A. LRT ; dropout design without POI

### B. Profile LRT ; Fix gamma and separately train the NB model

### C. Direct Wald Test 

- Hessian : Pseudo-Blockwise matrix. Blockwise Inversion + Sherman-Morrison



### 
