import matplotlib.pyplot as plt
plt.style.use('style.mplstyle')

rise = [0.2, 0.5, 1, 1.4, 2, 4, 10, 15]
fwhm_squared = [8.93**2, 4.78**2, 3.51**2, 3.33**2, 3.34**2, 4.03**2, 6.28**2, 7.74**2]

plt.scatter(rise, fwhm_squared, color='black', s=10)

plt.style.use('style.mplstyle')
plt.semilogx()
plt.semilogy()
plt.xlabel('Trap Filter Rise Time (microseconds)', ha='right', x=1.0)
plt.ylabel('FWHM^2 (ADC^2)', ha='right', y=1.0)
plt.tight_layout()
plt.show()
