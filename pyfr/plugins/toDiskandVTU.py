# -*- coding: utf-8 -*-

#from configparser import NoOptionError, NoSectionError
from ctypes import (POINTER, Structure, cast, c_float, c_int, c_int32, c_uint8, c_void_p)
import numpy as np
import pycuda.driver as cuda

#from pyfr.ctypesutil import load_librself.solHost
#from pyfr.plugins.base import BasePlugin
#from pyfr.shapes import BaseShape
#from pyfr.util import proxylist, subclass_where

class MeshInfo(Structure):
    _fields_ = [
        ('neles', c_int),
        ('nnodesperele', c_int),
        ('ncells', c_int),
        ('vertices', c_void_p),
        ('con', c_void_p),
        ('off', c_void_p),
        ('type', c_void_p)
    ]


class SolnInfo(Structure):
    _fields_ = [
        ('k', c_int),
        ('ldim', c_int),
        ('soln', c_void_p)
    ]

class vtuComp():
  def __init__(self):
    self.neles        = None
    self.nnodesperele = None
    self.ncells       = None
    self.vertices     = None
    self.con          = None
    self.off          = None
    self.type         = None
    
    self.k            = None
    self.ldim         = None
    self.soln         = None
    self.solHost      = None
    self.dtype        = None
    return
  
  def storeMesh(self, nEleTypes, minfo, sinfo, cfg):
    if (nEleTypes > 1):
      print("CAUTION: write vtu only works for hexes so far")
      return
    
    # copy mesh into numpy arrays
    self.neles        = minfo[0].neles
    self.nnodesperele = minfo[0].nnodesperele
    self.ncells       = minfo[0].ncells
    self.vertices     = np.fromiter(cast(minfo[0].vertices, POINTER(c_float)), dtype=np.float32, count=minfo[0].neles*minfo[0].nnodesperele)
    self.con          = np.fromiter(cast(minfo[0].con,      POINTER(c_int32)), dtype=np.int32,   count=minfo[0].ncells*8)
    self.off          = np.fromiter(cast(minfo[0].off,      POINTER(c_int32)), dtype=np.int32,   count=minfo[0].ncells)
    self.type         = np.fromiter(cast(minfo[0].type,     POINTER(c_uint8)), dtype=np.uint8,   count=minfo[0].ncells)
    
    # store shape variables of and pointer to solution
    self.k            = sinfo[0].k
    self.ldim         = sinfo[0].ldim
    self.soln         = sinfo[0].soln
    
    # allocate host memory for solution
    if (cfg.get('backend', 'precision', 'double') == 'single'):
      self.dtype      = np.float32
    else:
      self.dtype      = np.float64
    self.solHost      = np.empty((self.nnodesperele, self.ldim), dtype=self.dtype)
    
    return
    
  
  def storeVTU(self, time, nacptsteps):
    if (nacptsteps > 1):
      return
    
    # bring host buffer in initial shape
    self.solHost = self.solHost.reshape([self.nnodesperele, self.ldim])
    
    # copy solution from device to host
    cuda.memcpy_dtoh(self.solHost, self.soln)
    
    # unpack AoSoA to SoA
    nparr        = self.neles - self.neles % -self.k
    self.solHost = self.solHost.reshape([self.nnodesperele, nparr // self.k, 5, self.k])
    
    
    
    #self.solHost = self.solHost.reshape(self.datashape)
    #self.solHost = self.solHost.swapaxes(-2, -3)
    #self.solHost = self.solHost.reshape(self.ioshape[:-1] + (-1,))
    #self.solHost = self.solHost[..., :self.ioshape[-1]]
    
    # convert conservative to primitive
    
    
    return
