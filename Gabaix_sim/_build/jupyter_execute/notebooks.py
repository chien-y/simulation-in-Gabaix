# Investigation of the Granular Hypothesis

Prepared by Chien Yeh as research assistance for John Stachurski on power laws and the granular hypothesis

There is a debate about whether firm sizes follow Pareto distribution or log-normal distribution.
This research simulates the variation of growth of aggregate output under the assumptions of {cite}`gabaix2011granular` with these two distributions of firm sizes. The results imply that the firm sizes are Pareto distributed.

By accounting, the aggregate output is the sum of firms' sales, that is,
\begin{equation}
    Y_t = \sum_{i=1}^N {S_{it}}
\end{equation}
where $S_{it}$ denotes the sale for firm $i$. {cite}`gabaix2011granular` assumes that all firms have the same volatility on growth, $\sigma$, so that the volatility of GDP, $\sigma_{GDP}$, contributed from idiosyncratic shocks of firms is
\begin{equation}
    \sigma_{GDP} = \sigma \sum_{i=1}^N \left(\frac{S_{it}}{Y_t}\right)^2. 
\end{equation}

{cite}`gabaix2011granular` proves that if the variance of firms' sales or sizes is finite, then $\sigma_{GDP}$ decays out as the rate $1/\sqrt{N}.$ Then, the idiosyncratic shocks of firms are averaged out and only have little impact on macroeconomic fluctuation.
However, if the sizes of firms follows a power law, 
which have finite variance if the Pareto exponent is less than or equal to $2$, {cite}`gabaix2011granular` shows that the aggregate volatility decays according to $1/\ln N$ under Zipf's law. Therefore, the GDP fluctuation is substantial even if $N$ is large. The fat tail makes the central limit theorem breaks down and the idiosyncratic shocks remains on aggregate output.

This research uses numerical method to compare the aggregate fluctuations when firm sizes follow Pareto distribution and log-normal distribution.
The simulation is under the assumption of {cite}`gabaix2011granular` that all firms have the identical variance of growth and. Specifically, let  $\sigma = 12\%$ following  {cite}`gabaix2011granular`. 
Moreover, the parameter of Pareto distribution follows {cite}`axtell2001zipf` with Pareto exponent $\zeta=1.059.$ The parameters of log-normal distribution is determined by matching the mean and median with Pareto distribution.

When there are $1$ million firms $N=10^6$, the simulated $\sigma_{GDP}$ of equation (2) is $0.94\%$ for Pareto distributed firm sizes and is $0.1\%$ for log-normal distribution. Since the variance of log-normal distribution is finite, this result implies that the propositions in {cite}`gabaix2011granular` is reasonable. Moreover, since the empirical study in {cite}`gabaix2011granular` supports that the idiosyncratic shocks affect aggregate fluctuations and the empirical $\sigma_{GDP}$ is around $1\%$, the Pareto distribution for firm sizes is more reasonable than log-normal distribution. This concludes that the size of firms is Pareto distributed and then shocks among firms can have an impact on aggregate shocks.


import numpy as np
from scipy.stats import levy_stable
from scipy.stats import pareto
from scipy.stats import lognorm
from numba import jit, njit, prange, jitclass, float64
import matplotlib.pyplot as plt
%matplotlib inline

# matching_data = [
#     ('ζ', float64),
#     ('xm', float64),
#     ('h_lognormal', float64),
#     ('h_pareto', float64),
#     ('μ', float64),
#     ('σ', float64),
#     ('h_array', float64[:])
# ]

# @jitclass(matching_data)


class herfindahl():
    '''
    Simulate herfindahl on Pareto and log-normal distributions

    ζ: tail index
    xm: minumum for Pareto distribution
    nSim: number of times of simulation
    N: number of firms
    μ, σ are the shape parameters for log-normal
    '''

    def __init__(self, ζ=1.0, xm=1.0, ):
        self.ζ = ζ
        self.xm = xm
        self.h_lognormal = None
        self.h_pareto = None
        self.μ, self.σ = None, None

    def median_h(self, method='pareto', nSim=1000, N=1000000):
        if method == 'pareto':
            self.h_pareto = sim_pareto(self.ζ, self.xm, nSim=nSim, N=N)
            return self.h_pareto

        elif method == 'lognormal':
            self.solve()
            self.h_lognormal = sim_lognormal(self.μ, self.σ, nSim=nSim, N=N)
            return self.h_lognormal
        else:
            print('The distribution should be either pareto or lognormal.')

    def solve(self, lower=0.2, upper=10, plot=False, title=None):
        '''Solve μ and σ such that the lognormal matches the Pareto distribution'''
        ζ = self.ζ
        if ζ == 1:  # prevent 1/0 in the formula
            ζ += np.finfo(float).eps
        xm = self.xm
        μ = np.log(xm) + np.log(2) / ζ
        σ = np.sqrt(2 * (np.log(ζ * xm / (ζ - 1)) - μ))

        if plot:
            def f(x): return pareto.pdf(x, b=self.ζ, loc=0, scale=xm)
            def g(x): return lognorm.pdf(x, s=σ, scale=np.exp(μ))

            xd = np.linspace(1, upper, 100000)
            xd2 = np.linspace(lower, upper, 100000)
            plt.plot(xd, f(xd), label='Pareto')
            plt.plot(xd2, g(xd2), label='lognormal')
            plt.xlabel('x')
            plt.ylabel('probability')
            if title:
                plt.title(title)
            plt.legend()
            plt.show()
            print('μ=%1.5f, σ=%1.5f' % (μ, σ))
        self.μ, self.σ = μ, σ
        return μ, σ

    def s_GDP(self, s_π=0.12, method='pareto'):
        """Calculate the variance of GDP growth for both distribution"""
        if method == 'pareto':
            if self.h_pareto == None:
                h = self.median_h()
            h = self.h_pareto
        elif method == 'lognormal':
            if self.h_lognormal == None:
                h = self.median_h(method=method)
            h = self.h_lognormal
        else:
            raise TypeError(
                'The distribution should be either pareto or lognormal.')
        return h * s_π

    
@njit(parallel=True)
def sim_lognormal(μ, σ, nSim, N):
    """Simulators. Use numpy package to generate random variables."""
    h_array = np.empty(nSim)
    for i in range(nSim):
        #            size = LogNormal.rand(s=σ, scale=np.exp(μ), size=N)
        size = np.random.lognormal(μ, σ, N)
        h_array[i] = np.sqrt(np.sum(np.square(size / np.sum(size))))
    return np.median(h_array)


@njit(parallel=True)
def sim_pareto(ζ, xm, nSim, N):
    h_array = np.empty(nSim)
    for i in range(nSim):
        #            size = pareto.rvs(b=self.ζ, loc=0, scale=self.xm, size=N)
        U = np.random.uniform(a=0.0, b=1.0, size=N) # uniform[0, 1)
        size = xm / (1 - U)**(1 / ζ)
        h_array[i] = np.sqrt(np.sum(np.square(size / np.sum(size))))
    return np.median(h_array)

#### The Setup
To begin with, simulate the herfindhl for Pareto distribution. The formulas are from {cite}`gabaix2011granular`.

$$h = \big[\sum_{i=1}^{N} (\frac{S_{it}}{Y_t})^2\big]^{0.5} $$
$$Y_t = \sum_{i=1}^{N} S_{it} $$

Except for $\zeta=1$, also assume that $\zeta$ is 1.059 following {cite}`axtell2001zipf`.

Moreover, this experiment generate Pareto distribution by inverse transform sampling. Suppose that the random variable $X$ is Pareto distributed and its cumulative distribution function is 

\begin{equation}
    F_X(x)  =
    \begin{cases}
    {(\frac{x_m}{x})}^{\zeta}, & \text{if } x \geq x_m \\
    0, & x < x_m 
    \end{cases}
\end{equation}

If we let $F_X(x)=u$, then random variable $U=u$ is uniform distributed, $U \sim \mathcal{U}[0, 1)$. Then, by the relationship

\begin{equation}P(X \leq x) = P(F_X^{-1}(U) \leq x) = P(U \leq F_X(x)) = F_U(F_X(x)) = F_X(x),
\end{equation}

we can obtain the Pareto random variable $X$ by taking the inverse transform of uniform distributed $U$. More specifically, $x = F_X^{-1}(u) = x_m (1-u)^{-1/\zeta}$.


H = herfindahl()

for zeta in [1, 1.059, 1.5]:
    H.ζ = zeta
    print('If ζ is %1.3f' % zeta, end=',  ')
    print('then h is %1.2f%%' % (H.median_h()*100), end='')
    print(' and the corresponding σ_GDP is %1.2f%%.' % (H.s_GDP()*100))

Then, if $\zeta$ is one, the herfindahl h is $11.88\%$. This is close to the simulation result, $12\%$ in {cite}`gabaix2011granular`.

While if $\zeta$ is $1.059\%$, the herfindahl is $7.84\%$ and the standard deviation of growth rate is $0.94\%$. 

Gabaix also indicates that the empirically measured aggregate standard deviation of growth rate is around $1\%$.

Both the herfindahl and variance of growth rate are decreasing in $\zeta$. If $\zeta=1.5$, the standard deviation for GDP growth is less than $0.1\%$.

Generally, the result is compatible with {cite}`gabaix2011granular`.

#### Match the mean and median.
Given the Pareto distribution with tail index $\zeta > 1$, the density function is
\begin{equation}
f_P(x)=\frac{\zeta {\bar{x}}^{\zeta}}{x^{\zeta+1}}, \text{  if}\ x\geq \bar{x}. 
\end{equation}
  
Then, the corresponding mean and median are $\frac{\zeta \bar{x}}{\zeta-1}$ and $\bar{x} 2^{1/\zeta}$, respectively. The mean can be obtained by $\mathbb{E}(x)= \int x f_P(x) dx$ and the median is the root for $\frac{1}{2}=F_P(x)$ where $F_P(x)= 1-({\frac{x}{\bar{x}}})^{-\zeta}$ is the cumulative density function for $x \geq \bar{x}$.

If the log-normal distribution is $\ln(x) \sim \mathcal{N}(\mu, \,\sigma)$, then we can use the same approaches to find that its mean and median. The distribution for log-normal is

\begin{equation}
g(x) = \frac{1}{\sigma x \sqrt{2\pi}} \exp\{ {-\frac{(\ln{x-\mu})^2}{2\sigma^2}} \}.
\end{equation}

Thus, from $\mathbb{E}(x)=\int g(x)dx$, the mean is $\exp(\mu + \frac{\sigma^2}{2})$. In addition, the CDF is $G(x) = \Phi(\frac{\ln(x)-\mu}{\sigma})$ where $\Phi(.)$ is the CDF for standard normrl. Then, using $\frac{1}{2} = G(x)$, we know that $\frac{\ln(x)-\mu}{\sigma}=0$ and so the median is $\exp(\mu)$.

Now, we can try to match the log-normal with Pareto distribution under Zipf's law. Since the first moment is infinite when $\zeta=1$ for Pareto distribution, let $\zeta$ be very close to one but not one. Following the assumption for simulation in {cite}`gabaix2011granular`, assume $\bar{x}=1$. Then, fixing the mean and median of Pareto distribution (or fixing $\zeta$ and $\bar{x}$), if the mean and median for log-normal distribution are equal to those of Pareto distribution, we have the mean

\begin{equation}
\text{mean: } \quad \frac{\zeta \bar{x}}{\zeta-1} = \exp(\mu + \frac{\sigma^2}{2})
\end{equation}
\begin{equation}
\text{median: } \quad  \bar{x} 2^{1/\zeta} = \exp(\mu) 
\end{equation}

Solve the equation of median given $\zeta$ and $\bar{x}=1$, we have $\mu = \ln(2^{1/\zeta})$. Plug this into the mean equation, we can get $\sigma = (2 \ln(\frac{\zeta}{\zeta-1}) - \frac{2}{\zeta}\ln2)^{1/2}$.

##### Use the formula to sove $\mu$ and $\sigma$ numerically and plot the distribution.

H = herfindahl()
for zeta in [1+np.finfo(float).eps, 1.059, 1.5]:
    H.ζ = zeta
    H.solve(lower=0, plot=True, title='$\zeta$ is %1.3f'%zeta)

The matched log-normal has the parameters $\mu=0.65453$ and $\sigma=2.11330$ given $\zeta=1059$. 

Now, simulate the herfindahl for log-normal with these parameters under $\zeta=1.059$ and $\zeta=1.5$.

for zeta in [1.059, 1.5]:
    H.ζ = zeta
    print('If ζ is %1.3f' %zeta, end=',  ')
    print('then h is %1.3f%% for matched log-normal' %(H.median_h(method='lognormal')*100), end='')
    print(' and the corresponding σ_GDP is %1.3f%%.' %(H.s_GDP(method='lognormal')*100))

The scales for herfindahl and $\sigma_{GDP}$ under log-normal are way smaller than those under Pareto distribution.

 ##### Conclusion
 
This research simulates the the standard deviations for GDP growth for both Pareto distribution and log-normal distribution for firm sizes under Zipf's law. The choice of Pareto exponent $\zeta$ and the formulas for variance follow cite}`gabaix2011granular`. In addition, the simulation method is Monte Carlo, the Pareto random variables are generated by the inverse transform sampling, and the parameters for log-normal are determined such that the corresponding mean and median are equal those of Pareto distribution. 
 
The simulated volatilities for GDP growth are too small and not in the same magnitude for both $\zeta=1.059$ and $\zeta=1.5$. 
When firm's growth rate is $12\%$, the deviation of growth rate is around $0.1\%$ for log-normal which is much lower than $0.94\%$ of Pareto given $\zeta=1.059$. Since the empirically measured macroeconomic fluctuation is around $1\%$, the log-normal distribution is not reasonable under this matching method and simulation. 
This also confirms the first proposition of {cite}`gabaix2011granular`, which shows that the GDP volatility decays in the rate of $\frac{1}{N^{1/2}}$ when the firm sizes have finite variance. To conclude, the underlying distribution for firm sizes should be Pareto distribution rather than log-normal distribution in this experiment.  


```{bibliography} references.bib
```

