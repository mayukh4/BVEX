import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd 
from scipy import signal
from matplotlib.colors import LogNorm

# load data

uprism_imudata = pd.read_csv("GQ7_IMU_20220822_mod.csv")
uprism_imudata.columns = uprism_imudata.columns.str.replace(' ', '_')
t = uprism_imudata.UTC_Timestamp

# extract data

ax = uprism_imudata.Scaled_Accel_X * 9.8 #m/s2
ay = uprism_imudata.Scaled_Accel_Y * 9.8 #m/s2
az = uprism_imudata.Scaled_Accel_Z * 9.8 #m/s2

gx = uprism_imudata.Scaled_Gyro_X  # deg/sec
gy = uprism_imudata.Scaled_Gyro_Y  # deg/sec
gz = uprism_imudata.Scaled_Gyro_Z  # deg/sec

# plot data

fig, axs = plt.subplots(1,3,sharex=True,figsize=(20,6))
fig.suptitle('U-Prism IMU data: Acceleration values',size=18)

axs[0].plot(ax)
axs[0].set_title("Ax")

axs[1].plot(ay)
axs[1].set_title("Ay")

axs[2].plot(az)
axs[2].set_title("Az")

axs[1].set_xlabel('Samples',size=14)
axs[0].set_ylabel('Acceleration ($ m/s^2) $',size=14)

plt.savefig("imuaccl.png",dpi="figure")

plt.show()

fig, axs = plt.subplots(1,3,sharex=True,figsize=(20,6))
fig.suptitle('U-Prism IMU data: Gyroscope values',size=18)

axs[0].plot(gx)
axs[0].set_title("Gx")

axs[1].plot(gy)
axs[1].set_title("Gy")

axs[2].plot(gz)
axs[2].set_title("Gz")

axs[1].set_xlabel('Samples',size=14)
axs[0].set_ylabel('Rotation ($ deg/s) $',size=14)

plt.savefig("imugyro.png",dpi="figure")

plt.show()

fs= 1 # Hz
win = 2048

fx,psdax = signal.welch(ax,fs,nperseg=win)
fy,psday = signal.welch(ay,fs,nperseg=win)
fz,psdaz = signal.welch(az,fs,nperseg=win)


fig, axs = plt.subplots(1,3,sharex=True,figsize=(24,6))
fig.suptitle('U-Prism IMU data: PSD of the Accelerometer',size=18)

axs[0].plot(fx, psdax,color = 'blue')
axs[0].set_title("Ax")

axs[1].plot(fy, psday,color = 'black')
axs[1].set_title("Ay")

axs[2].plot(fz, psdaz,color = 'red')
axs[2].set_title("Az")

axs[1].set_xlabel('Frequency (Hz)',size=14)
axs[0].set_ylabel(r' ASD ($g^2/Hz$) ',size=14)



plt.xscale('log')

plt.savefig("imuasd.png",dpi="figure")
plt.show()

fx,psdgx = signal.welch(gx,fs,nperseg=win)
fy,psdgy = signal.welch(gy,fs,nperseg=win)
fz,psdgz = signal.welch(gz,fs,nperseg=win)


fig, axs = plt.subplots(1,3,sharex=True,figsize=(24,6))
fig.suptitle('U-Prism IMU data: PSD of the Gyroscope',size=18)

axs[0].plot(fx, psdgx,color = 'blue')
axs[0].set_title("Gx")

axs[1].plot(fy, psdgy,color = 'black')
axs[1].set_title("Gy")

axs[2].plot(fz, psdgz,color = 'red')
axs[2].set_title("Gz")

axs[1].set_xlabel('Frequency (Hz)',size=14)
axs[0].set_ylabel(r' PSD ($dps^2/Hz$) ',size=14)



plt.xscale('log')

plt.savefig("imupsd.png",dpi="figure")
plt.show()

# Plotting spectrogram

fx, t, Sxx = signal.spectrogram(ax, fs)
fy, t, Syy = signal.spectrogram(ay, fs)
fz, t, Szz = signal.spectrogram(az, fs)

fgx, t, Sgx = signal.spectrogram(gx, fs)
fgy, t, Sgy = signal.spectrogram(gy, fs)
fgz, t, Sgz = signal.spectrogram(gz, fs)

fig, axs = plt.subplots(1,3,sharex=True,figsize=(24,6))
fig.suptitle('U-Prism IMU data: Spectrogram of Accelerometer',size=18)


imgax= axs[0].pcolormesh(t, fx, Sxx,norm=LogNorm(),cmap='magma')
axs[0].set_title("Ax")
plt.colorbar(imgax,label=r"ASD[$g^2/Hz$]")

imgay= axs[1].pcolormesh(t, fy, Syy,norm=LogNorm(),cmap='magma')
axs[1].set_title("Ay")
axs[1].set_ylabel('Frequency [Hz]')
plt.colorbar(imgay,label=r"ASD[$g^2/Hz$]")

imgaz= axs[2].pcolormesh(t, fx, Szz,norm=LogNorm(),cmap='magma')
axs[2].set_title("Az")
plt.colorbar(imgaz,label=r"ASD[$g^2/Hz$]")

axs[0].set_ylabel('Frequency [Hz]',size=18)
axs[1].set_xlabel('Time [sec]',size=18)

plt.savefig("imusad.png",dpi="figure")

plt.show()

fig, axs = plt.subplots(1,3,sharex=True,figsize=(24,6))
fig.suptitle('U-Prism IMU data: Spectrogram of Gyroscope',size=18)


imggx= axs[0].pcolormesh(t, fgx, Sgx,norm=LogNorm(),cmap='magma')
axs[0].set_title("Gx")
plt.colorbar(imggx,label=r"PSD[$dps^2/Hz$]")

imggy= axs[1].pcolormesh(t, fgy, Sgy,norm=LogNorm(),cmap='magma')
axs[1].set_title("Gy")
plt.colorbar(imggy,label=r"PSD[$dps^2/Hz$]")

imggz= axs[2].pcolormesh(t, fgz, Sgz,norm=LogNorm(),cmap='magma')
axs[2].set_title("Gz")
plt.colorbar(imggz,label=r"PSD[$dps^2/Hz$]")

axs[0].set_ylabel('Frequency [Hz]',size=18)
axs[1].set_xlabel('Time [sec]',size=18)

plt.savefig("imusgd.png",dpi="figure")

plt.show()