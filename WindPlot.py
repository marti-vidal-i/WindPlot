# WIND PLOTTER FOR FULL-POLARIZATION IMAGES
# Copyright (c) Ivan Marti-Vidal - Universitat de Valencia (2019). 
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>,
# or write to the Free Software Foundation, Inc., 
# 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# a. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# b. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
# c. Neither the name of the author nor the names of contributors may 
#    be used to endorse or promote products derived from this software 
#    without specific prior written permission.
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
## Version history:
##
## v1.0b - 15/08/2019 - Bonn. First version for M87* (2017 campaign).
##
## 2019-2021 - Intermediate versions (EHTC) not available 
##             (just parameter fine-tunning).
##
## v2.0b - 04/04/2024 - Valencia. pCoffeeBreak. 
##                      Turned the code into a Python function.
##                      Added example plot file.
##

from astropy.io import fits as pf
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import sys
import pickle as pk
import gc



class WindPlot(object):
  """ Function that generates a 'wind plot' from images in Stokes I, Q and U."""



  def __init__(self,fits='', Ipow=1.0, Ppow = 1.0, nParticles = 10000, seed = 42, 
                 Icut = 0.001, Pcut = 0.025, zPadding = 8, 
                 zoom = 1.0):

## Lines for testing:
#if True:
#  fits = ['V_STK_0_POLSOLVE_STK_%s.fits'%stk for stk in ['I','Q','U','V']]
#  Ipow = 0.5; nParticles = 10000; nInteg = 200; zPadding = 8;
#  Ppow = 1.0; seed = 42; Icut = 0.001; Pcut = 0.05; windBeam = 2.0; 
#  zoom = 6.0; figTitle = 'WindPlot v2.0b'; useLatex = True
#  outFile = 'WindPlot.png'; angUnit = 'muas'

    """ Initializer method.

      Parameters
      ----------
      fits : str
        Name of full-polarization fits file. It can also be a list of fits 
        files. In such a case, it is assumed that each list element is a 
        fits file representing Stokes I, Q, and U (in that order) and all files
        share the same dimensions.
      Ipow : float
        The Stokes I color intensity will be scaled as I^(Ipow).
      Ppow : float
        The windlines contrast will be scaled as P^(Ppow).
      nParticles : int
        Number of wind particles (high values -> more crowded winds).
      seed : int
        Seed of the random number generator
      nInteg : int 
        Number of 'integration times' for the wind particles. 
        It affects the maximum length of the lines.
      Icut : float
        Cutoff factor for Stokes I, in units of the brightness peak.
      Pcut : float
        Cutoff factor for linear polarization, in units of polarization peak.
      zPadding : int
        The windplot needs a high pixel resolution. If the images have large
        pixels, zPadding will regrid the image with pixels zPadding times smaller,
        by using zero-padding interpolation in Fourier space.
        WARNING: Too large values may generate 'square-like' convolution artifacts. 
      zoom : float
        The zoom-in factor for the plot (i.e., the inner fraction of the image 
        that will be shown in the plot). If a list of floats is given, these will be
        taken as the image edges of the plot [RAmin,RAmax,Decmin,Decmax], in degrees.

    """

## UNITS:
    units = {'deg':1.0, 'rad':np.pi/180., 'arcsec':3600., 'as':3600., 'mas':3.6e6, 'muas':3.6e9}


####################################
# MAKE COPIES OF THE FUNCTION PARAMETERS, 
# FORCING THEIR TYPES:

# NUMBER OF PARTICLES:
    self.N = int(nParticles)

# PARAMETER TO SET THE CONTRAST:
    P_POW = float(Ppow)

# PARAMETER TO GAMMA-CORRECT STOKES I:
    I_POW = float(Ipow)

# SEED OF RANDOM-NUMBER GENERATOR:
    SEED = int(seed)


# BRIGHTNESS CUTOFF (IMAGE INTENSITY UNITS): 
    Cut = float(Icut)

# POLARIZED BRIGHTNESS CUTOFF (JY/BEAM):
    CutPol = float(Pcut)


# ZERO-PADDING INTERPOLATION FACTOR:
    zPad = int(zPadding)


# ZOOM-IN FACTOR TOWARD IMAGE CENTER:

    if type(float) is list:
      Zoom = [float(zi) for zi in zoom]
    else:
      Zoom = float(zoom)




# FITS IMAGES (ONE IMAGE PER STOKES PARAMETER; IN "I, Q, U" ORDER):
    if type(fits) is str:
      IM = str(fits) 
    elif type(fits) is list:
      IM = [str(fi) for fi in fits]
    else:
      raise Exception('Wrong FITS argument (should be string or list of strings).')




##################
# FUNCTION STARTS: #
##################


    np.random.seed(SEED)


#########################
# READ STOKES IQU:
    try:
      if type(fits) is str:
        aux = pf.open(IM)
      else:
        aux = pf.open(IM[0])
    except:
      raise Exception('Bad FITS file!')

## GET IMAGE DIMENSIONS/AXES:
    headerKeys = np.unique([k for k in aux[0].header.keys()])
    STOKES = -1; RA = -1; DEC= -1; FREQ = -1;
    for key in headerKeys:
      item = aux[0].header[key]
      if 'CTYPE' in key and type(item) is str:
        if item.startswith('STOKES'):
          STOKES = int(key[-1])
        if item.startswith('RA'):
          RA = int(key[-1])
        if item.startswith('DEC'):
          DEC = int(key[-1])
        if item.startswith('FREQ'):
          FREQ = int(key[-1])

# Read the image:
    IMGORIG = np.copy(aux[0].data)
    NDIM = len(np.shape(IMGORIG))
## We get the first freq channel:  
    if FREQ>0:
      IMG = IMGORIG.take(axis=NDIM-FREQ,indices=0)
## We get first stokes:
    if STOKES>0:
      if STOKES>FREQ:
        STid = NDIM-STOKES
      else:
        STid = NDIM-STOKES-1
      if RA<DEC:
        I = np.copy(IMG.take(axis=STid,indices=0))
      else:
        I = np.copy(IMG.take(axis=STid,indices=0).transpose())



# CELL SIZE IN DESIRED UNITS:
    cUnit = aux[0].header['CUNIT%i'%RA]

# REFERENCE PIXELS AND VALUES:
    self.RAp0 = int(aux[0].header['CRPIX%i'%RA])
    self.DECp0 = int(aux[0].header['CRPIX%i'%DEC])
    self.RA0 = float(aux[0].header['CRVAL%i'%RA])/units[cUnit]*units['deg']
    self.DEC0 = float(aux[0].header['CRVAL%i'%DEC])/units[cUnit]*units['deg']
    self.dRA = float(aux[0].header['CDELT%i'%RA])/units[cUnit]*units['deg']
    self.dDEC = float(aux[0].header['CDELT%i'%DEC])/units[cUnit]*units['deg']
    aux.close()


## GET IMAGES IN THE OTHER STOKES PARAMETERS:
    if type(fits) is str:
      if STOKES <0 or np.shape(IMGORIG)[NDIM-STOKES]==1:
        raise Exception('ERROR! Fits file does not have polarization info!')
      else:
        if RA<DEC:
          QU = [np.copy(IMG.take(axis=STid,indices=1)),np.copy(IMG.take(axis=STid,indices=2))]
        else:
          QU = [np.copy(IMG.take(axis=STid,indices=1).transpose()),np.copy(IMG.take(axis=STid,indices=2).transpose())]
    else:
      QU = []
      for stk in [1,2]:
        aux = pf.open(IM[stk])
# Read the image:
        IMGORIG = np.copy(aux[0].data)
## We get the first freq channel:  
        if FREQ>0:
          IMG = IMGORIG.take(axis=NDIM-FREQ,indices=0)
## We get the other stokes from the other images:
        if STOKES>0:
          if STOKES>FREQ:
            STid = NDIM-STOKES
          else:
            STid = NDIM-STOKES-1
          if RA<DEC:
            QU.append(np.copy(IMG.take(axis=STid,indices=0)))
          else:
            QU.append(np.copy(IMG.take(axis=STid,indices=0).transpose()))
        aux.close()


###################
# Image arrays:
    Q,U = QU

# The number of rows and columns should be even:
    Nx,Ny = np.shape(I)
    ChangeIt = False
    if Nx%2: 
      Nx += 1; ChangeIt = True
    if Ny%2: 
      Ny += 1; ChangeIt = True
    Iarr = np.zeros((Nx,Ny),dtype=np.float32)
    Qarr = np.zeros((Nx,Ny),dtype=np.float32)
    Uarr = np.zeros((Nx,Ny),dtype=np.float32)
    if ChangeIt:
      Iarr[:Nx-1,:Ny-1] = I
      Qarr[:Nx-1,:Ny-1] = Q
      Uarr[:Nx-1,:Ny-1] = U
    else:
      Iarr[:,:] = I
      Qarr[:,:] = Q
      Uarr[:,:] = U
#####################



    BAD_I = np.isnan(Iarr)
    BAD_POL = np.logical_or(np.isnan(Qarr),np.isnan(Uarr),np.isnan(Iarr))
    Iarr[BAD_I] = 0.0
    Qarr[BAD_POL] = 0.0
    Uarr[BAD_POL] = 0.0  


    print('\nApplying zero-padding interpolation\n')

#################################
# Zero-padding interpolation:

    NxL = int(Nx*zPad); NyL = int(Ny*zPad)

# Set z-padded Fourier transform:
    ILFou = np.zeros((NxL,NyL),dtype=np.complex64)


# Stokes I interpolation:
    Ifou = np.fft.fft2(Iarr)
    ILFou[0:Nx//2, 0:Ny//2] = Ifou[0:Nx//2, 0:Ny//2]
    ILFou[NxL-Nx//2:NxL, NyL-Ny//2:NyL] = Ifou[Nx//2:Nx, Ny//2:Ny]
    ILFou[0:Nx//2, NyL-Ny//2:NyL] = Ifou[0:Nx//2, Ny//2:Ny]
    ILFou[NxL-Nx//2:NxL, 0:Ny//2] = Ifou[Nx//2:Nx, 0:Ny//2]
    I = np.fft.ifft2(ILFou).real

# Stokes Q interpolation:
    Ifou[:] = np.fft.fft2(Qarr)
    ILFou[:] = 0.0
    ILFou[0:Nx//2, 0:Ny//2] = Ifou[0:Nx//2, 0:Ny//2]
    ILFou[NxL-Nx//2:NxL, NyL-Ny//2:NyL] = Ifou[Nx//2:Nx, Ny//2:Ny]
    ILFou[0:Nx//2, NyL-Ny//2:NyL] = Ifou[0:Nx//2, Ny//2:Ny]
    ILFou[NxL-Nx//2:NxL, 0:Ny//2] = Ifou[Nx//2:Nx, 0:Ny//2]
    Q = np.fft.ifft2(ILFou).real

# Stokes U interpolation:
    Ifou = np.fft.fft2(Uarr)
    ILFou[:] = 0.0
    ILFou[0:Nx//2, 0:Ny//2] = Ifou[0:Nx//2, 0:Ny//2]
    ILFou[NxL-Nx//2:NxL, NyL-Ny//2:NyL] = Ifou[Nx//2:Nx, Ny//2:Ny]
    ILFou[0:Nx//2, NyL-Ny//2:NyL] = Ifou[0:Nx//2, Ny//2:Ny]
    ILFou[NxL-Nx//2:NxL, 0:Ny//2] = Ifou[Nx//2:Nx, 0:Ny//2]
    U = np.fft.ifft2(ILFou).real


    del Ifou, ILFou, Iarr, Qarr, Uarr, IMG, IMGORIG, BAD_I, BAD_POL
    gc.collect()

# Set image window to process:
    if type(Zoom) is list:
      self.imin = ((Zoom[0]-self.RA0)/self.dRA + self.RAp0)*zPad
      self.imax = ((Zoom[1]-self.RA0)/self.dRA + self.RAp0)*zPad
      self.jmin = ((Zoom[2]-self.DEC0)/self.dDEC + self.DECp0)*zPad
      self.jmax = ((Zoom[3]-self.DEC0)/self.dDEC + self.DECp0)*zPad
    else:
      st = int(NxL/Zoom)
      self.imin = (NxL-st)//2
      self.imax = (NxL+st)//2
      self.jmin = (NyL-st)//2
      self.jmax = (NyL+st)//2

    if self.imin>self.imax:
      self.imin,self.imax = self.imax,self.imin
    if self.jmin>self.jmax:
      self.jmin,self.jmax = self.jmax,self.jmin

# Ensure that the plot is well within the image's coverage:
    if self.jmin<0:
      self.jmin = 0
    if self.jmax>=NyL:
      self.jmax = NyL-1

    if self.imin<0:
      self.imin = 0
    if self.imax>=NxL:
      self.imax = NxL-1

      
## Compute I mask and apply gamma correction:
    I /= np.max(I)
    self.Mask = (I > Cut)
    self.IPlot = np.copy(I)
    self.IPlot -= Cut
    self.IPlot[self.IPlot<0.0] = 0.0
    self.IPlot[:] = np.power(self.IPlot,I_POW)


# (NORMALIZED) POLARIZED BRIGHTNESS:
    self.P = np.sqrt(Q**2. + U**2.)
    Pmax = np.max(self.P)

# CUTOFF (wind contrast will be set to zero below CutPol):
    self.P -= CutPol*Pmax
    self.P[self.P<0.0] = 0.0
    self.P /= np.max(self.P)
    self.P[:] = np.power(self.P,P_POW)


#########
# EVPA FIELD (FORCE POSITIVE):
    EVPA = np.arctan2(U,Q)/2.
    EVPA[EVPA<0.0] += np.pi

# CONNECT EVPAS FROM NEIGHBORING PIXELS:

    print('\nConnecting EVPAs\n')
    for i in range(self.imin,self.imax):
      for j in range(self.jmin,self.jmax):
        if self.Mask[i,j] and self.Mask[i+1,j]: 
          if (EVPA[i,j]-EVPA[i+1,j])>np.pi/2.:
            EVPA[i+1:,j] += np.pi
          elif (EVPA[i,j]-EVPA[i+1,j])<-np.pi/2.:
            EVPA[i+1:,j] -= np.pi



# PROJECTION ON X AND Y AXES:
    self.Cs = np.cos(EVPA)
    self.Ss = np.sin(EVPA)
#########

    del I, Q, U, QU
    gc.collect()




 
# INITIAL COORDINATES FOR PARTICLES:

# AS INTEGERS:
    self.X0 = np.floor(np.random.random(self.N)*(self.imax-self.imin)+self.imin).astype(np.int32)
    self.Y0 = np.floor(np.random.random(self.N)*(self.jmax-self.jmin)+self.jmin).astype(np.int32)

# AS FLOATS (FOR PRECISE INTEGRATION):
    self.X1 = np.copy(self.X0).astype(np.float64)
    self.Y1 = np.copy(self.Y0).astype(np.float64)

    self.X0m = np.copy(self.X0)
    self.Y0m = np.copy(self.Y0)

    self.X1m = np.copy(self.X0).astype(np.float64)
    self.Y1m = np.copy(self.Y0).astype(np.float64)


####################################
# INITIAL VELOCITIES FOR EACH PARTICLE.
# TAKEN FROM THE EVPAs (IN BOTH SENSES):

    self.Vxa = np.zeros(self.N)
    self.Vya = np.zeros(self.N)

    self.Vxam = np.zeros(self.N)
    self.Vyam = np.zeros(self.N)
    for i in range(self.N):
      self.Vxa[i] = self.Ss[self.X0[i],self.Y0[i]]
      self.Vya[i] = self.Cs[self.X0[i],self.Y0[i]]
      self.Vxam[i] = -self.Ss[self.X0[i],self.Y0[i]]
      self.Vyam[i] = -self.Cs[self.X0[i],self.Y0[i]]
####################################


# IMAGE WHERE THE WIND LINES WILL BE DRAWN:
    self.Wind = np.zeros((NxL,NyL),dtype=np.float32)
    self.changed = np.zeros(self.N,dtype=bool)
    self.changedm = np.zeros(self.N,dtype=bool)
    self.Vfa = np.zeros(self.N)
    self.Vfam = np.zeros(self.N)

    self.NxL = NxL
    self.NyL = NyL
    self.zPad = zPad

    self.units = units


  def makeWindLines(self,nTimes):
    """ Method to integrate the wind lines.

        Parameters
        ----------
        nTimes : int
        Number of integration steps. Larger values make longer lines.

        Returns : bool
    """

    nT = int(nTimes)

    print('\n Integrating wind lines.\n')
    for k in range(nT):
      sys.stdout.write('\r Step %04i of %04i'%(k,nT))
      sys.stdout.flush()
      for i in range(self.N):
        self._applyStep(i)
      self.Wind[self.X0[self.changed],self.Y0[self.changed]] += self.Vfa[self.changed]
      self.Wind[self.X0m[self.changedm],self.Y0m[self.changedm]] += self.Vfam[self.changedm]


    return True





######################
## MAKE FIGURE!

  def plotImage(self,windBeam = 2.0, figTitle = 'WindPlot v2.0b', 
         useLatex = True, outFile = 'WindPlot.png', angUnit = 'arcsec'):

    """ Method to generate the figure.

        Parameters
        ----------
        windBeam : float
           Size of convolving Gaussian (in pixel units) for the wind lines.
        figTitle : str
           Title to be written at the top of the windplot figure.
        useLatex : bool
           If LaTeX fonts are available to Python, use them (make a publication-ready plot).
        outFile : str
           Name (with extension) of the output figure file.
        angUnit : str
           Units for the sky coordinates (can be 'rad','deg','arcsec','as','mas', or 'muas').


        Returns : matplotlib subplot instance.

    """


# WIDTH OF CONVOLVING GAUSSIAN FOR TRAJECTORY MASK:
    GG = float(windBeam)

# FIGURE TITLE:
    Title = str(figTitle)

# USE LATEX FONTS?
    NICE_FONTS = bool(useLatex)

# UNITS FOR IMAGE AXES:
    unit = str(angUnit)

# FILENAME FOR FIGURE:
    outname = outFile


# CONVOLVE THE WIND MASK WITH A GAUSSIAN (i.e., BLUR THE WIND):
    print('\n\nConvolving wind with blurring Gaussian.\n')
    XX = np.linspace(-self.NxL/2.,self.NxL/2.,self.NxL)**2.
    YY = np.ones(self.NyL)
    RR = np.outer(XX,YY) + np.outer(YY,XX)

    Gx = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.exp(-RR/(2.*GG**2.)))*np.fft.fft2(self.Wind)).real)

    del XX,YY,RR
    gc.collect()


    Gx /= np.max(Gx)



# EXTENT OF IMAGE:

    RAs = [(-self.RAp0*self.dRA)/self.units['deg']*self.units[unit], ((self.NxL/self.zPad-self.RAp0)*self.dRA)/self.units['deg']*self.units[unit]]
    DECs = [(-self.DECp0*self.dDEC)/self.units['deg']*self.units[unit], ((self.NyL/self.zPad-self.DECp0)*self.dDEC)/self.units['deg']*self.units[unit]]

# PLOT IMAGE FIGURE:
    fig = pl.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    fig.subplots_adjust(right=0.95,top=0.95)

    Extent1 = (RAs[0]+self.imin/self.NxL*(RAs[1]-RAs[0]), RAs[0]+self.imax/self.NxL*(RAs[1]-RAs[0]))
    Extent2 = (DECs[0]+self.jmin/self.NyL*(DECs[1]-DECs[0]), DECs[0]+self.jmax/self.NyL*(DECs[1]-DECs[0]))

#    pli = sub.imshow(((Gx+1.0)*self.IPlot[self.imin:self.imax,self.jmin:self.jmax]),origin='lower',cmap='afmhot', extent=(Extent1[0],Extent1[1],Extent2[0],Extent2[1]))

    pli = sub.imshow(((Gx+1.0)*self.IPlot),origin='lower',cmap='afmhot', extent=(RAs[0],RAs[1],DECs[0],DECs[1]))

    if unit=='muas':
      unitLabel = r'$\mu$as'
    else:
      unitLabel = unit

    pl.xlabel(r'RA Offset (%s)'%unitLabel)
    pl.ylabel(r'Dec Offset (%s)'%unitLabel)

    sub.set_xlim((Extent1[0],Extent1[1]))
    sub.set_ylim((Extent2[0],Extent2[1]))


# FINE-TUNE THE FIGURE:
    sub.spines['bottom'].set_color('w')
    sub.spines['top'].set_color('w') 
    sub.spines['right'].set_color('w')
    sub.spines['left'].set_color('w')
    sub.tick_params(axis='x', colors='w')
    sub.tick_params(axis='y', colors='w')
    sub.yaxis.label.set_color('w')
    sub.xaxis.label.set_color('w')
    fig.patch.set_facecolor('k')

    sub.set_title(Title)
    sub.title.set_color('w')

# SAVE IT!
    plt.savefig(outname,facecolor='black')

    return sub





  def _applyStep(self,j):

      """ Method that applies one integration step over line j."""

   # ONLY INTEGRATE PARTICLES THAT ARE STILL WITHIN THE MASK:
      if self.X0[j] > self.imin and self.Y0[j]>self.jmin and self.X0[j]<self.imax and self.Y0[j]<self.jmax and self.Mask[self.X0[j],self.Y0[j]]:

     # SET VELOCITY IN X AND Y AXES:
        Vx = self.Cs[self.X0[j],self.Y0[j]]
        Vy = self.Ss[self.X0[j],self.Y0[j]]

     # Modulus will be normalized to 1 pixel:
        dt = 1./np.sqrt(Vx*Vx+Vy*Vy)

     # LINE CONTRAST WILL DEPEND ON POLARIZED BRIGHTNESS:
        Vf = self.P[self.X0[j],self.Y0[j]]

     # CORRECT FOR THE 180 DEG. AMBIGUITY IN EVPA:
        if np.abs(Vx-self.Vxa[j])>0.5:
          Vx *= -1.0; Vy *= -1.0
     # CORRECT AGAIN (SILLY, AIN'T?)
        if np.abs(Vy-self.Vya[j])>0.5:
          Vy *= -1.0; Vx *= -1.0

     # UPDATE THE VELOCITY:
        self.Vxa[j] = Vx
        self.Vya[j] = Vy

     # INTEGRATE THE TRAJECTORY:
        dX = Vx*dt ; dY = Vy*dt
        self.X1[j] -= dX
        self.Y1[j] += dY

     # FIGURE OUT CURRENT PIXEL:
        X0a = int(self.X0[j]); Y0a = int(self.Y0[j])
        self.X0[j] = int(round(self.X1[j]))
        self.Y0[j] = int(round(self.Y1[j]))

     # UPDATE WIND MASK:
        self.changed[j] = (self.Y0[j] != Y0a) or (self.X0[j] != X0a)
        self.Vfa[j] = Vf
 
   # REPEAT THE SAME STEPS, BUT TOWARD THE OPPOSITE DIRECTION:
      if self.X0m[j] > self.imin and self.Y0m[j]>self.jmin and self.X0m[j]<self.imax and self.Y0m[j]<self.jmax and self.Mask[self.X0m[j],self.Y0m[j]]:
        Vx = -self.Cs[self.X0m[j],self.Y0m[j]]
        Vy = -self.Ss[self.X0m[j],self.Y0m[j]]

        Vf = self.P[self.X0m[j],self.Y0m[j]]

        dt = 1./np.sqrt(Vx*Vx+Vy*Vy)

        if np.abs(Vx-self.Vxam[j])>0.5:
          Vx *= -1.0; Vy *= -1.0
        if np.abs(Vy-self.Vyam[j])>0.5:
          Vy *= -1.0; Vx *= -1.0

        self.Vxam[j] = Vx
        self.Vyam[j] = Vy
        dX = Vx*dt ; dY = Vy*dt
        self.X1m[j] -= dX
        self.Y1m[j] += dY
        X0a = int(self.X0m[j]); Y0a = int(self.Y0m[j])
        self.X0m[j] = int(round(self.X1m[j]))
        self.Y0m[j] = int(round(self.Y1m[j]))
        self.changedm[j] = (self.Y0m[j] != Y0a) or (self.X0m[j] != X0a)
        self.Vfam[j] = Vf
















