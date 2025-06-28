# Taken from curvarbase
qmin=1e-2
dlogq=0.1
baseline = 30

# df ~ qmin / baseline
df = qmin / baseline
fmin = 2. / baseline
fmax = 2.

nf = int(np.ceil((fmax - fmin) / df))
freqs = fmin + df * np.arange(nf)
periods = 1/freqs[::-1]
print ((periods[periods>5][1] - periods[periods>5][0])*720)
print ((periods[periods>14][1] - periods[periods>14][0])*720)
