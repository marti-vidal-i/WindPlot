import WindPlot as WP
import multiprocessing
import sys, os
import pylab as pl
if __name__=='__main__':


  STK_FILES = ['./TEST_DATA/%s_test.fits'%stk for stk in ['I','Q','U','V']]
  NINTEG = 200

  MyWP = WP.WindPlot(fits=STK_FILES, zoom = 1.0, Ppow=1.0)

  MyWP.makeWindLines(NINTEG)
  subPlot = MyWP.plotImage(angUnit='muas')

