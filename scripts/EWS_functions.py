import numpy as np
import statsmodels.api as sm
import scipy.stats as st
from scipy.optimize import curve_fit


def runmean(x, w):
    ## running mean of timeseries x with window size w
    n = x.shape[0]
    xs = np.zeros_like(x)
    for i in range(w // 2):
        xs[i] = np.nanmean(x[: i + w // 2 + 1])
    for i in range(n - w // 2, n):
        xs[i] = np.nanmean(x[i - w // 2 + 1:])

    for i in range(w // 2, n - w // 2):
        xs[i] = np.nanmean(x[i - w // 2 : i + w // 2 + 1])
    return xs

## EWS functions for timeseries without missing values ##

def runstd(x, w): 
    ## calculate standard deviation of timeseries x in running windows of length w
    n = x.shape[0]
    xs = np.zeros_like(x)
    for i in range(w // 2): # for beginning edge bit, the window centre starts at w/2
        xw = x[: i + w // 2 + 1] # window always starts at 0 and more and more covers into range
        xw = xw - xw.mean() # make into variations around zero
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]
            xw = xw - p0 * np.arange(xw.shape[0]) - p1 # remove linear trend

            xs[i] = np.std(xw) # calculate standard deviation
        else:
            xs[i] = np.nan
    for i in range(n - w // 2, n):
        xw = x[i - w // 2 + 1:] # for end bit window always ends at 0 and covers less and less
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]

            xw = xw - p0 * np.arange(xw.shape[0]) - p1
            xs[i] = np.std(xw)
        else:
            xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1] # standard window
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]
            xw = xw - p0 * np.arange(xw.shape[0]) - p1

            xs[i] = np.std(xw)
        else:
            xs[i] = np.nan

    return xs

def runac2(x, w):
    ## calcualte the autocorrelation of timeseries x in running window size w
    n = x.shape[0]
    xs = np.zeros_like(x)
    for i in range(w // 2):
        xw = x[: i + w // 2 + 1]
        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]
            xw = xw - p0 * np.arange(xw.shape[0]) - p1 # remove linear trend
            xs[i] = sm.tsa.acf(xw)[1]
        else:
            xs[i] = np.nan

    for i in range(n - w // 2, n):
        xw = x[i - w // 2 + 1:]

        xw = xw - xw.mean()
        if np.std(xw) > 0:
            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]

            xw = xw - p0 * np.arange(xw.shape[0]) - p1

            xs[i] = sm.tsa.acf(xw)[1]
        else:
            xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]

        xw = xw - xw.mean()
        if np.std(xw) > 0:

            lg = st.linregress(np.arange(xw.shape[0]), xw)[:]
            p0 = lg[0]
            p1 = lg[1]

            xw = xw - p0 * np.arange(xw.shape[0]) - p1

            xs[i] = sm.tsa.acf(xw)[1]
        else:
            xs[i] = np.nan

    return xs

def run_fit_a_ar1(x, w):
    ## calculate the restoring rate of timeseries x in running windows w
    n = x.shape[0]
    xs = np.zeros_like(x)

    for i in range(w // 2):
        xs[i] = np.nan

    for i in range(n - w // 2, n):
        xs[i] = np.nan

    for i in range(w // 2, n - w // 2):
        xw = x[i - w // 2 : i + w // 2 + 1]
        xw = xw - xw.mean() # variations in the window

        p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)

        xw = xw - p0 * np.arange(xw.shape[0]) - p1 # remove linear trend


        dxw = xw[1:] - xw[:-1]

        xw = sm.add_constant(xw)
        model = sm.GLSAR(dxw, xw[:-1], rho=1)
        results = model.iterative_fit(maxiter=10)

        a = results.params[1]

        xs[i] = a
    return xs

def fourrier_surrogates(ts, ns):
    ## generate ns Fourier-phase surrogates of timeseries ts
    ## surrogates preserve the power spectrum but randomise the phase, destroying temporal structure
    ts_fourier = np.fft.rfft(ts)
    random_phases = np.exp(np.random.uniform(0, 2 * np.pi, (ns, ts.shape[0] // 2 + 1)) * 1.0j)
    ts_fourier_new = ts_fourier * random_phases
    new_ts = np.real(np.fft.irfft(ts_fourier_new))
    return new_ts