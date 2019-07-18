# -*- coding: utf-8 -*-

from configparser import NoOptionError, NoSectionError
from ctypes import (CDLL, CFUNCTYPE, POINTER, Structure, cast, c_char, c_char_p,
                    c_double, c_int, c_size_t, c_void_p, memmove, memset)

import numpy as np

from pyfr.ctypesutil import load_library
from pyfr.plugins.base import BasePlugin
from pyfr.shapes import BaseShape
from pyfr.util import proxylist, subclass_where

from pyfr.plugins.toDiskandVTU import vtuComp

def _cfg_wrapper(meth, type):
    @CFUNCTYPE(c_int, c_char_p, c_char_p, POINTER(type))
    def fn(section, option, out):
        try:
            out[0] = meth(section.decode(), option.decode())

            return 0
        except NoOptionError:
            return -1
        except NoSectionError:
            return -2

    return fn


def _cfg_str_wrapper(meth):
    @CFUNCTYPE(c_int, c_char_p, c_char_p, POINTER(c_char), c_size_t)
    def fn(section, option, out, nout):
        try:
            r = meth(section.decode(), option.decode()).encode()

            if out and nout:
                memset(out, 0, nout)
                memmove(out, r, min(len(r), nout - 1))

            return len(r)
        except NoOptionError:
            return -1
        except NoSectionError:
            return -2

    return fn


class CfgGetters(Structure):
    _fields_ = [
        ('get', c_void_p),
        ('getpath', c_void_p),
        ('getbool', c_void_p),
        ('getint', c_void_p),
        ('getfloat', c_void_p)
    ]


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


class VisPlugin(BasePlugin):
    name = 'vis'
    formulations = ['std', 'dual']
    systems = ['euler', 'navier-stokes']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        self.backend = backend = intg.backend
        self.mesh = intg.system.mesh

        # Amount of subdivision to perform
        self.divisor = self.cfg.getint(cfgsect, 'divisor', 3)

        # Allocate a queue on the backend
        self._queue = backend.queue()

        # Solution arrays
        self.eles_scal_upts_inb = inb = intg.system.eles_scal_upts_inb

        # Load the vis library
        self._load_vislib(self.cfg.getpath(cfgsect, 'libvis', abs=False))

        # Prepare the VTU structures and interpolation kernels
        minfo, sinfo, kerns = [], [], []
        for etype, smat in zip(intg.system.ele_types, inb):
            mi, sop = self._prepare_vtu(etype, intg.rallocs.prank)

            # Allocate on the backend
            vmat = backend.matrix((mi.nnodesperele, self.nvars, mi.neles),
                                  tags={'align'})
            sop = backend.const_matrix(sop)
            backend.commit()

            # Prepare the solution info structure
            si = SolnInfo(k=backend.soasz, ldim=vmat.leaddim, soln=vmat.data)

            # Prepare the matrix multiplication kernel
            kk = backend.kernel('mul', sop, smat, out=vmat)

            # Append
            minfo.append(mi)
            sinfo.append(si)
            kerns.append(kk)

        # Wrap the kernels in a proxy list
        self._interpolate_upts = proxylist(kerns)

        # Copy the mesh and solution structures into a C array
        minfo_arr = (MeshInfo*len(minfo))(*minfo)
        sinfo_arr = (SolnInfo*len(sinfo))(*sinfo)

        # Generate the cfg file wrappers
        wcfg, wcfgsect = self._wrap_cfg(self.cfg, cfgsect)

        # Finally, initialise the vis library
        self._vptr = self._lib.vis_init(
            wcfgsect, wcfg, self.nvars, len(minfo_arr), minfo_arr, sinfo_arr)
            
        # store copy of mesh
        self.vtuOut = vtuComp()
        self.vtuOut.storeMesh(len(minfo_arr), minfo_arr, sinfo_arr, self.cfg)
        

    def _prepare_vtu(self, etype, part):
        from pyfr.writers.vtk import BaseShapeSubDiv

        mesh = self.mesh['spt_{0}_p{1}'.format(etype, part)]

        # Get the shape and sub division classes
        shapecls = subclass_where(BaseShape, name=etype)
        subdvcls = subclass_where(BaseShapeSubDiv, name=etype)

        # Dimensions
        nspts, neles = mesh.shape[:2]

        # Sub divison points inside of a standard element
        svpts = shapecls.std_ele(self.divisor)
        nsvpts = len(svpts)

        # Shape
        soln_b = shapecls(nspts, self.cfg)

        # Generate the operator matrices
        mesh_vtu_op = soln_b.sbasis.nodal_basis_at(svpts)
        soln_vtu_op = soln_b.ubasis.nodal_basis_at(svpts)

        # Calculate node locations of vtu elements
        vpts = np.dot(mesh_vtu_op, mesh.reshape(nspts, -1))
        vpts = vpts.reshape(nsvpts, -1, self.ndims)

        # Append dummy z dimension for points in 2D
        if self.ndims == 2:
            vpts = np.pad(vpts, [(0, 0), (0, 0), (0, 1)], 'constant')

        # Reorder and cast
        # vpts = vpts.swapaxes(1, 2).astype(self.backend.fpdtype)
        vpts = vpts.swapaxes(1, 2).astype(np.float32)

        # Perform the sub division
        nodes = subdvcls.subnodes(self.divisor)

        # Prepare vtu cell arrays
        vtu_con = np.tile(nodes, (neles, 1))
        vtu_con += (np.arange(neles)*nsvpts)[:, None]
        vtu_con = vtu_con.astype(np.int32)

        # Generate offset into the connectivity array
        vtu_off = np.tile(subdvcls.subcelloffs(self.divisor), (neles, 1))
        vtu_off += (np.arange(neles)*len(nodes))[:, None]
        vtu_off = vtu_off.astype(np.int32)

        # Tile vtu cell type numbers
        vtu_typ = np.tile(subdvcls.subcelltypes(self.divisor), neles)
        vtu_typ = vtu_typ.astype(np.uint8)

        # Construct the mesh info structure
        mi = MeshInfo(neles=neles, nnodesperele=nsvpts, ncells=len(vtu_typ),
                      vertices=vpts.ctypes.data, con=vtu_con.ctypes.data,
                      off=vtu_off.ctypes.data, type=vtu_typ.ctypes.data)
        # Retain the underlying NumPy objects
        mi._vpts = vpts
        mi._vtu_con = vtu_con
        mi._vtu_off = vtu_off
        mi._vtu_typ = vtu_typ

        return mi, soln_vtu_op

    def _wrap_cfg(self, cfg, cfgsect):
        getters = {
            'get': _cfg_str_wrapper(cfg.get),
            'getpath': _cfg_str_wrapper(cfg.getpath),
            'getbool': _cfg_wrapper(cfg.getbool, c_int),
            'getint': _cfg_wrapper(cfg.getint, c_int),
            'getfloat': _cfg_wrapper(cfg.getfloat, c_double)
        }

        wcfg = CfgGetters()
        wcfg._fns = getters

        for k, v in getters.items():
            setattr(wcfg, k, cast(v, c_void_p))

        return wcfg, c_char_p(cfgsect.encode())

    def _load_vislib(self, lpath):
        self._lib = lib = CDLL(lpath)

        # vis_init
        lib.vis_init.argtypes = [c_char_p, POINTER(CfgGetters), c_int, c_int,
                                 POINTER(MeshInfo), POINTER(SolnInfo)]
        lib.vis_init.restype = c_void_p

        # vis_run
        lib.vis_run.argtypes = [c_void_p, c_double, c_int]
        lib.vis_run.restype = None

        # vis_del
        lib.vis_del.argtypes = [c_void_p]
        lib.vis_del.restype = None

    def __del__(self):
        if hasattr(self, '_vptr'):
            self._lib.vis_del(self._vptr)

    def __call__(self, intg):
        # Configure the input bank
        self.eles_scal_upts_inb.active = intg._idxcurr

        # Interpolate to the vis points
        self._queue % self._interpolate_upts()

        # Call out to the library method
        self._lib.vis_run(self._vptr, intg.tcurr, intg.nacptsteps)
        
        #from pyfr.plugins.toDiskandVTU import vtuComp
        self.vtuOut.storeVTU(intg.tcurr, intg.nacptsteps)
