# -*- coding: utf-8 -*-

#from configparser import NoOptionError, NoSectionError
from ctypes import (POINTER, Structure, cast, c_float, c_int, c_int32, c_uint8, c_void_p)
import numpy as np
import pycuda.driver as cuda
import vtk


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
    self.gamred       = None
    return
  
  def storeMesh(self, nEleTypes, minfo, sinfo, cfg):
    if (cfg.get('soln-plugin-vis', 'writeOrigVTU', 'false') == 'false'):
      return
    
    if (nEleTypes > 1):
      print("CAUTION: write vtu only works for hexes so far")
      return
    
    # copy mesh into numpy arrays
    self.neles        = minfo[0].neles
    self.nnodesperele = minfo[0].nnodesperele
    self.ncells       = minfo[0].ncells
    self.vertices     = np.fromiter(cast(minfo[0].vertices, POINTER(c_float)), dtype=np.float32, count=minfo[0].neles*minfo[0].nnodesperele*3)
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
    
    self.gamred = cfg.getfloat('constants', 'gamma') - 1.
    return
    
  
  def storeVTU(self, time, nacptsteps, cfg):
    if (nacptsteps > 1):
      return
    
    if (cfg.get('soln-plugin-vis', 'writeOrigVTU', 'false') == 'false'):
      return
    
    # bring host buffer in initial shape
    self.solHost = self.solHost.reshape([self.nnodesperele, self.ldim])
    
    # copy solution from device to host
    cuda.memcpy_dtoh(self.solHost, self.soln)
    
    # unpack AoSoA to SoA
    nparr        = self.neles - self.neles % -self.k
    self.solHost = self.solHost.reshape([self.nnodesperele, nparr // self.k, 5, self.k])
    self.solHost = self.solHost.swapaxes(-2, -3)
    self.solHost = self.solHost.reshape([self.nnodesperele, 5, -1])
    self.solHost = self.solHost[..., :self.neles]
    
    # separate vars
    self.solHost = self.solHost.swapaxes(-2, -3)
    self.solHost = self.solHost.reshape([5, self.neles*self.nnodesperele])
    
    # convert conservative to primitive
    # velocities
    self.solHost[1] = np.divide(self.solHost[1], self.solHost[0])
    self.solHost[2] = np.divide(self.solHost[2], self.solHost[0])
    self.solHost[3] = np.divide(self.solHost[3], self.solHost[0])
    
    # pressure (p = (gamma-1)*(e-.5*rho*v**2))
    ekin = np.add(np.multiply(self.solHost[1], self.solHost[1]), np.multiply(self.solHost[2], self.solHost[2]))
    ekin = np.add(np.multiply(self.solHost[3], self.solHost[3]), ekin)
    ekin = np.multiply(self.solHost[0], .5*ekin)
    self.solHost[4] = self.gamred*np.subtract(self.solHost[4], ekin)
    
    #for ivert in range(0, self.neles*self.nnodesperele):
      #xx = self.vertices[3*ivert]
      #yy = self.vertices[3*ivert+1]
      #zz = self.vertices[3*ivert+2]
      #rhotest = (6.28318530717958647693+xx) + 3.*(6.28318530717958647693+yy) + 5*(6.28318530717958647693+zz)
      #if (not np.isclose(self.solHost[0][ivert], rhotest, 1.E-5, 1.E-5)):
        #print("Noooo! " + repr(ivert) + "   rho: "+ repr(self.solHost[0][ivert]) + "  rhotest: " + repr(rhotest))
    
    
    # prepare data for vtk output
    
    pts = vtk.vtkPoints()
    pts.SetNumberOfPoints(self.neles*self.nnodesperele)
    rho = vtk.vtkFloatArray()
    rho.SetName("Density")
    rho.SetNumberOfComponents(1)
    rho.SetNumberOfTuples(self.neles*self.nnodesperele)
    vel = vtk.vtkFloatArray()
    vel.SetName("Velocity")
    vel.SetNumberOfComponents(3)
    vel.SetNumberOfTuples(self.neles*self.nnodesperele)
    prs = vtk.vtkFloatArray()
    prs.SetName("Pressure")
    prs.SetNumberOfComponents(1)
    prs.SetNumberOfTuples(self.neles*self.nnodesperele)
    for ivert in range(self.neles*self.nnodesperele):
      pts.InsertPoint(ivert,                    self.vertices[3*ivert], 
                      self.vertices[3*ivert+1], self.vertices[3*ivert+2])
      rho.SetTuple1(ivert, self.solHost[0, ivert])
      vel.SetTuple3(ivert, self.solHost[1, ivert], self.solHost[2, ivert], self.solHost[3, ivert])
      prs.SetTuple1(ivert, self.solHost[4, ivert])
  
    
    
# build cells
    aHexahedron     = vtk.vtkHexahedron()
    aHexahedronGrid = vtk.vtkUnstructuredGrid()
    aHexahedronGrid.Allocate(1, 1)
    for icell in range(self.ncells):
      for ivert in range(8):
        aHexahedron.GetPointIds().SetId(ivert, self.con[8*icell+ivert])
      aHexahedronGrid.InsertNextCell(aHexahedron.GetCellType(),
 aHexahedron.GetPointIds())
    
    # setpoints'n'pointdata
    aHexahedronGrid.SetPoints(pts)
    aHexahedronGrid.GetPointData().AddArray(rho)
    aHexahedronGrid.GetPointData().AddArray(vel)
    aHexahedronGrid.GetPointData().AddArray(prs)
    
    # write out file
    fn = 'aaaaaaaaa.vtu'
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fn)
    writer.SetInputData(aHexahedronGrid)
    writer.Write()
    
    
    #print("Verts 0-17:")
    #for ivert in range(0,17):
      #print("  " + repr(self.vertices[3*ivert]) + "  " + repr(self.vertices[3*ivert+1]) + "  " + repr(self.vertices[3*ivert+2]))
    
    #print("\nVerts 4096-4113:")
    #for ivert in range(4096,4113):
      #print("  " + repr(self.vertices[3*ivert]) + "  " + repr(self.vertices[3*ivert+1]) + "  " + repr(self.vertices[3*ivert+2]))
    
    #print("\nVerts 8192-8209:")
    #for ivert in range(8192,8209):
      #print("  " + repr(self.vertices[3*ivert]) + "  " + repr(self.vertices[3*ivert+1]) + "  " + repr(self.vertices[3*ivert+2]))
    
    #print("\nVerts 12288-12305:")
    #for ivert in range(12288,12305):
      #print("  " + repr(self.vertices[3*ivert]) + "  " + repr(self.vertices[3*ivert+1]) + "  " + repr(self.vertices[3*ivert+2]))
      
    #print("\nVerts 36864-36881:")
    #for ivert in range(36864,36881):
      #print("  " + repr(self.vertices[3*ivert]) + "  " + repr(self.vertices[3*ivert+1]) + "  " + repr(self.vertices[3*ivert+2]))
      
    
    # convert conservative to primitive
    
    
    return
