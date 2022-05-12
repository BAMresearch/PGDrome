'''
    PGD class

    structure of PGD solution: --> PGD class (general infos,  evaluation functions and save/ write functions)
    for each PGD coordinate one PGDMesh class for each solution value in each PGDMesh one PGDAttribute class

    structure is derived from pxdmf file format (visualization from PGD with paraview)
'''

import logging
import os
import xml.etree.ElementTree as et

import dolfin
import h5py
import numpy as np
from numpy import linalg as la
from scipy import interpolate

# set logger
LOGGER = logging.getLogger(__name__)

try:
    # import tensorly if possible only needed in reconstruct_solution_tensor
    import tensorly
    tensorly.set_backend('numpy')
except ModuleNotFoundError as e:
    LOGGER.warning(f'Failed loading tensorly with error: {e}')

try:
    from fenicstools import Probes
except ModuleNotFoundError as e:
    LOGGER.warning(f'Failed loading fenicstools with error: {e}')


def ftool_probe(points, function):
    ''' function from Philipp Diercks to efficiently evaluate fenics functions with fenicstools'''
    probes = Probes(points.flatten(), function.function_space())
    probes(function)
    return probes.array()


class PGD:
    '''
        Stores the whole PGD solution incl. mesh information and
        allows saving, loading and evaluation of it

        file saving: self.write_pxdmf, self.write_hdf5, self.xdmf
        file loading: self.load_pxdmf

        evaluation of PGD solution:
            create_interpolation_fcts, evaluate, evaluate_... (min, min_abs, max, max_abs, max_norm)
            error_computation. reconstruct_tensor
    '''

    def __init__(self, name=None, n_modes=None, fmeshes=[], pgd_modes=None, name_coord=None,
                 modes_info=None, verbose=False, problem=None, *args, **kwargs):
        '''
            Create PGD solution class to save modes and meshes

            :param name:               name of problem
            :param n_modes:            number Modes
            :param name_coord:         name of PGD coordinates [.....] (len(PGD_d))
            :param fmeshes:            fenics mesh fucntion per coord [....] (len(PGD_d))
            :param pgd_modes:          modes given as func from fenics [....] (len(PGD_d))
            :param modes_info:         list with [name, center, type] primarily for writing pxdmf files
        '''
        self.logger = logging.getLogger(__name__)

        self.name = name

        self.folder = ''  # string to folder where the pxdmf file as well as xdmf and h5 files are
        self.numModes = n_modes  # number of PGD modes per variable
        self.used_numModes = n_modes  # copy of numModes used in evaluation to efficent change there the number
        self.mesh = list()  # list of PGDMesh instances

        self.name_coord = name_coord
        self.modes_info = modes_info

        # load mesh data
        for ctr, mesh in enumerate(fmeshes):
            _name = 'PGD' + str(ctr + 1)
            grid = PGDMesh(_name, mesh, self.name_coord[ctr], pgd_modes[ctr], self.numModes, modes_info=self.modes_info)
            self.mesh.append(grid)
            if verbose:
                for att in grid.attributes:
                    att.print_info()
                grid.print_info()
        self.problem = None  # if no pxdmf file exist to solve pgd solution in time load as PGDModel().create_from_problem(PGDProblem)
        self.pos = 0  # if eval_type == 'pos' this gives the point where the fct should be evaluate
        self._eval_fixed_modes = {}  # accessed with self.eval_fixed_modes which is cached

    def __str__(self):
        return 'PGD(name: %s)(meshes: %s)(modes: %s)' % (self.name, len(self.mesh), self.numModes)

    def __repr__(self):
        return f'{str(self)}'

    def eval_fixed_modes(self, sensor_points, fixed_dim, attri):
        _hash = np.sum(sensor_points.flatten())
        if (_hash, fixed_dim, attri) in self._eval_fixed_modes:
            return self._eval_fixed_modes[_hash, fixed_dim, attri]

        probes = Probes(sensor_points.flatten(),
                        self.mesh[fixed_dim].attributes[attri].interpolationfct[0].function_space())
        for k in range(self.numModes):  # chached for ALL numModes in evaluate only used_numModes will be computed!!
            probes(self.mesh[fixed_dim].attributes[attri].interpolationfct[k])

        eval_fixed_mode = probes.array()
        self._eval_fixed_modes[_hash, fixed_dim, attri] = eval_fixed_mode
        return eval_fixed_mode

    @property  # wird dynamisch bestimmt
    def num_pgd_var(self):
        # number of PGD variables == number of meshes
        return len(self.mesh)

    @property  # wird dynamisch bestimmt
    def fenics_meshes(self):
        # put all meshes in one list
        return [m.fenics_mesh for m in self.mesh]

    def _info_str(self):
        ''' String w/ class information '''
        info = 'summary of PGDModel class\n'
        info += '-------------------------------\n'
        info += 'name:                          %s\n' % self.name
        info += 'number of PGD variables:       %s\n' % self.num_pgd_var
        info += 'number of modes for each mesh -- max: %s -- used: %s\n' % (self.numModes, self.used_numModes)
        info += 'number of saved meshes:        %s\n' % len(self.mesh)
        info += 'number of elements per mesh:    '
        for i in range(0, len(self.mesh)):
            info += ' %s, ' % self.mesh[i].numElements
        info += '\nfolder:                        %s' % self.folder
        return info

    def print_info(self):
        print('\n' + self._info_str() + '\n')

    def write_hdf5(self, folder):
        '''
            Write HDF5 file for each mesh with its PGD modes saved in PGDModel class
            as well asl xml file to save the mesh data

            :param folder: directory where to save
        '''
        for coord, mesh in enumerate(self.mesh):
            filepath = os.path.join(folder, mesh.name + '_data.h5')
            file_out = dolfin.HDF5File(dolfin.MPI.comm_world, filepath, 'w')
            file_out.write(self.fenics_meshes[coord], 'mesh')
            for att in mesh.attributes:
                for mode in range(self.numModes):
                    file_out.write(att.interpolationfct[mode], 'MODE_' + str(mode))
                    self.logger.debug('Functionspace degree: %s ', att.interpolationfct[mode].ufl_element().degree())
            file_out.close()
        self.logger.info('Wrote %i HDF files for Mode data', self.num_pgd_var)

    def _write_xdmf(self, folder):
        '''
            Write xdmf file to create data in h5 format for each mesh saved in PGDModel class

            :param folder: directory where to save
        '''
        for coord, mesh in enumerate(self.mesh):
            filepath = os.path.join(folder, mesh.name + '.xdmf')
            file_out = dolfin.XDMFFile(filepath)
            file_out.write(self.fenics_meshes[coord])
            for att in mesh.attributes:
                for mode in range(self.numModes):
                    file_out.write(att.interpolationfct[mode], mode)
            file_out.close()

    def write_pxdmf(self, folder, xdmf_exist=False):
        '''
            Write a PXDMF file by merging xdmf/h5 files for each PGD coordinate (fenics)
            Can be opened with paraview + plugin see https://rom.ec-nantes.fr/?page_id=12

            :param folder: directory where to save
            :param meshes: list of all fenics meshes
            :param xdmf_exist: xdmf files already there?? usually FALSE
            :param pgd: solution saved in class object
            :return:
        '''
        # are there xdmf files ?
        if xdmf_exist is False:
            self._write_xdmf(folder)

        # check the dimensions of the meshes
        dims = np.zeros(self.num_pgd_var)
        kdim = 0
        for cur_mesh in self.mesh:
            self.logger.debug('mesh info %s ', cur_mesh.info)
            dims[kdim] = cur_mesh.info[0]
            kdim += 1
        # all the same dimension than all good
        # one a higher dimension than a other --> need to add zeros to get the dimensions == 3 for Vector attributes!!!
        if dims.max() != dims.min():
            xdmf_exist = True
        else:
            xdmf_exist = False

        # put the xdmf files together in one pxdmf file
        with open(os.path.join(folder, self.name + '.pxdmf'), 'w') as file_out:

            # prefix
            file_out.write(
                '<?xml version="1.0"?><!--pxdmf written by my own code writePXDMF.py based on my forward_models PGD class-->\n')
            file_out.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
            file_out.write('<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
            file_out.write('  <Domain Name="' + self.name + '.pxdmf">\n')

            # each grid with its fPGD modes
            for cur_mesh in self.mesh:

                # name
                file_out.write('    <Grid Name="' + cur_mesh.name + '">\n')

                # informations
                file_out.write('      <Information Name="Dims" Value="' + str(cur_mesh.info[0]) + '" />\n')
                file_out.write('      <Information Name="Dim0" Value="' + cur_mesh.info[1] + '" />\n')
                file_out.write('      <Information Name="Unit0" Value="' + cur_mesh.info[2] + '" />\n')

                # mesh data
                # check dimension for topology and geometry data by reading h5 file
                hf = h5py.File(folder + '/' + cur_mesh.name + '.h5', 'r')
                groupt = hf.get('Mesh/0/mesh/topology')
                nodes_per_element = np.array(groupt).shape[1]
                file_out.write('        <Topology NumberOfElements = "' + str(
                    cur_mesh.numElements) + '" TopologyType = "' + cur_mesh.typElements + '" NodesPerElement = "' + str(
                    nodes_per_element) + '" >\n')
                file_out.write('          <DataItem Dimensions = "' + str(cur_mesh.numElements) + ' ' + str(
                    nodes_per_element) + '" NumberType = "UInt" Format = "HDF">' + cur_mesh.name + '.h5:/Mesh/0/mesh/topology</DataItem>\n')
                file_out.write('        </Topology>\n')

                groupg = hf.get('Mesh/0/mesh/geometry')
                if np.array(groupg).shape[1] == 2:
                    file_out.write('        <Geometry GeometryType = "XY">\n')
                    file_out.write('          <DataItem Dimensions = "' + str(np.array(groupg).shape[0]) + ' ' + str(
                        np.array(groupg).shape[
                            1]) + '" Format = "HDF">' + cur_mesh.name + '.h5:/Mesh/0/mesh/geometry</DataItem>\n')
                    file_out.write('        </Geometry>\n')
                elif np.array(groupg).shape[1] == 3:
                    file_out.write('        <Geometry GeometryType = "XYZ" >\n')
                    file_out.write('          <DataItem Dimensions = "' + str(np.array(groupg).shape[0]) + ' ' + str(
                        np.array(groupg).shape[
                            1]) + '" Format = "HDF">' + cur_mesh.name + '.h5:/Mesh/0/mesh/geometry</DataItem>\n')
                    file_out.write('        </Geometry>\n')

                # attributes
                for cur_attr in cur_mesh.attributes:
                    for count in range(len(cur_attr.data)):
                        if cur_attr.field.lower() == 'vector' and xdmf_exist:  # check if we need to extend attribute data
                            # extend data to 3D PGD2.h5:/VisualisationVector/0
                            data_hdf = np.array(hf.get('/VisualisationVector/' + str(count)))
                            data_extended = np.zeros((data_hdf.shape[0], 3))
                            if cur_mesh.info[0] > 1:
                                # copy into for mesh_x copy ux and uy (uz == 0)
                                data_extended[:, 0:data_hdf.shape[1]] = data_hdf
                            elif cur_mesh.info[0] == 1:
                                for i in range(3):
                                    # copy first line
                                    data_extended[:, i] = data_hdf[:, 0]
                            file_out.write('        <Attribute Name="' + cur_attr.name + '_' + str(
                                count) + '" AttributeType="' + cur_attr.field + '" Center="Node">\n')
                            file_out.write(
                                '          <DataItem Dimensions="' + str(cur_attr.data[count].shape[0]) + ' ' + str(
                                    3) + '" Format="XML" NumberType="float" >\n')
                            for i in range(data_extended.shape[0]):
                                if data_extended.shape[1] == 2:
                                    file_out.write("%.8e %.8e\n" % (data_extended[i, 0], data_extended[i, 1]))
                                if data_extended.shape[1] == 3:
                                    file_out.write("%.8e %.8e %.8e\n" % (
                                        data_extended[i, 0], data_extended[i, 1], data_extended[i, 2]))
                            file_out.write('          </DataItem>\n')
                            file_out.write('        </Attribute>\n')
                        else:
                            # as long flag == False this will work scalar values!!
                            file_out.write('        <Attribute Name="' + cur_attr.name + '_' + str(
                                count) + '" AttributeType="' + cur_attr.field + '" Center="Node">\n')
                            file_out.write(
                                '          <DataItem Dimensions="' + str(cur_attr.data[count].shape[0]) + ' ' + str(
                                    cur_attr.data[count].shape[
                                        1]) + '" Format="HDF">' + cur_mesh.name + '.h5:/VisualisationVector/' + str(
                                    count) + '</DataItem>\n')
                            file_out.write('        </Attribute>\n')

                # end grid
                file_out.write('    </Grid>\n')

            # end
            file_out.write('  </Domain>\n</Xdmf>')

            self.logger.info('Wrote %s ', os.path.join(folder, self.name + '.pxdmf'))

            # close not needed because with open

    def load_pxdmf(self, filepath, verbose=False):
        '''
            Read PXDMF file and save into this PGDModel instance

            Example usage: solution = PGDModel().load_pxdmf('asdlkjalskd')

            :param filepath: filepath to pxdmf file
            :param verbose: if TRUE class info will be printed after loading
            :return: this instance
        '''
        # helper functions
        get_name = lambda fullname: '_'.join(fullname.split('_')[:-1])

        def data_to_array(text, _type):
            ''' Read data in text and save it as list using data type type '''
            temp = list()
            new = text.split('\n')
            new.pop()  # delete first and last value
            new.pop(0)
            for x in new:  # parsing text and appending
                # b=list(filter(None,x.split(' ')))#filter removes whitespace elements overall
                # print(x, b)
                ttemp = list()
                for a in list(filter(None, x.split(' '))):
                    if _type == 'int':
                        ttemp.append(int(a))
                    if _type == 'float':
                        ttemp.append(float(a))
                temp.append(ttemp)
            return temp

        folder = os.path.dirname(os.path.abspath(filepath))
        self.logger.info('Read %s', os.path.abspath(filepath))
        xmltree = et.parse(filepath)
        xmlroot = xmltree.getroot()

        # save pxdmf file in PGDsolution class
        self.folder = folder  # save directory for later!!
        self.name = xmlroot.findall('Domain')[0].attrib.get('Name')  # Name of Domain
        self.mesh = list()

        # pgd_meshes
        for PGDs in xmlroot.iter('Grid'):
            pgd_mesh = PGDMesh(PGDs.get('Name'))

            # if there is a pgd_mesh.name.xml file than load a fenics Mesh and save it
            try:
                # self.logger.debug('try to read Mesh from %s', pgd_mesh.name+'.xml')
                # pgd_mesh.MeshFct = dolfin.Mesh(folder+'/'+pgd_mesh.name+'.xml')
                # print(pgd_mesh.MeshFct.coordinates()[:])
                # from PGDX_data.h5 files
                self.logger.debug('Read Mesh from %s', pgd_mesh.name + '_data.h5')
                hdf = dolfin.HDF5File(dolfin.MPI.comm_world, folder + '/' + pgd_mesh.name + '_data.h5', 'r')
                pgd_mesh.fenics_mesh = dolfin.Mesh()
                hdf.read(pgd_mesh.fenics_mesh, 'mesh', False)
                # print(pgd_mesh.MeshFct.coordinates()[:])
                hdf.close()
            except RuntimeError:
                pgd_mesh.fenics_mesh = None

            pgd_mesh.info = list()
            for elems in PGDs.iter('Information'):
                pgd_mesh.info.append([elems.attrib.get('Name'), elems.attrib.get('Value')])
            pgd_mesh.meshdim = int(pgd_mesh.info[0][1])

            for elems in PGDs.iter('Topology'):
                pgd_mesh.numElements = int(elems.attrib.get('NumberOfElements'))
                pgd_mesh.typElements = elems.attrib.get('TopologyType')

                # save data depends on data type in pxdmf file
                if elems[0].get('Format') == 'XML':
                    self.logger.debug('Data saved as %s', elems[0].get('Format'))
                    topo = data_to_array(elems[0].text, 'int')  # read data and save them into a list integer
                    pgd_mesh.topology = np.array(topo)

                elif elems[0].get('Format') == 'HDF':
                    self.logger.debug('Data saved as %s', elems[0].get('Format'))
                    # print(elems[0].text.split(':'))
                    with h5py.File(folder + '/' + elems[0].text.split(':')[0], 'r') as hf:
                        # hf = h5py.File(folder + '/' + elems[0].text.split(':')[0], 'r')
                        pgd_mesh.topology = np.array(hf.get(elems[0].text.split(':')[1]))
                        self.logger.debug('shape of data %s', pgd_mesh.topology.shape)
                    # print('topo ', pgd_mesh.topology)

            for elems in PGDs.iter('Geometry'):
                if elems[0].get('Format') == 'XML':
                    pgd_mesh.typGeometry = elems.attrib.get('GeometryType')
                    geom = np.array(
                        data_to_array(elems[0].text, 'float'))  # read data and save them into a list integer

                elif elems[0].get('Format') == 'HDF':
                    with h5py.File(folder + '/' + elems[0].text.split(':')[0], 'r') as hf:
                        geom = np.array(hf.get(elems[0].text.split(':')[1]))
                        # print('geom ', geom)

                pgd_mesh.numNodes = geom.shape[0]
                if geom.shape[1] == 3:
                    pgd_mesh.dataX = geom[:, 0]
                    pgd_mesh.dataY = geom[:, 1]
                    pgd_mesh.dataZ = geom[:, 2]
                elif geom.shape[1] == 2:
                    pgd_mesh.dataX = geom[:, 0]
                    pgd_mesh.dataY = geom[:, 1]

            # PGD Attributes (PGD modes)
            pgd_mesh.attributes = list()
            for elems in PGDs.iter('Attribute'):

                name = get_name(elems.attrib.get('Name'))

                # check if there is already an attribute with this name
                check = False
                for i in range(len(pgd_mesh.attributes)):
                    if pgd_mesh.attributes[i].name == name:
                        # print('found name, merge data here')
                        check = True
                        position = i

                if check:
                    # add data to existing attribute
                    if elems[0].get('Format') == 'XML':
                        pgd_mesh.attributes[position].data.append(np.array(data_to_array(elems[0].text, 'float')))
                    elif elems[0].get('Format') == 'HDF':
                        with h5py.File(folder + '/' + elems[0].text.split(':')[0], 'r') as hf:
                            pgd_mesh.attributes[position].data.append(np.array(hf.get(elems[0].text.split(':')[1])))

                else:
                    # create new attribute
                    attr = PGDAttribute()
                    attr.name = name
                    attr._type = elems.attrib.get('Center')
                    attr.field = elems.attrib.get('AttributeType')
                    if elems[0].get('Format') == 'XML':
                        attr.data = [np.array(data_to_array(elems[0].text, 'float'))]
                    elif elems[0].get('Format') == 'HDF':
                        with h5py.File(folder + '/' + elems[0].text.split(':')[0], 'r') as hf:
                            attr.data = [np.array(hf.get(elems[0].text.split(':')[1]))]

                    pgd_mesh.attributes.append(attr)
            self.mesh.append(pgd_mesh)
        self.numModes = len(self.mesh[0].attributes[0].data)  # have do be the same for each mesh and attribute!!
        self.used_numModes = len(self.mesh[0].attributes[0].data)

        # Print summary
        if verbose:
            self.print_info()
            for Mesh in self.mesh:
                Mesh.print_info()
                for Attr in Mesh.attributes:
                    Attr.print_info()

        return self

    def create_from_problem(self, problem=None):
        '''
            create PGDModel by a given PGDProblem
        :param problem: PGDProblem instance
        :return:
        '''
        self.problem = problem
        self.name = problem.name
        self.logger.info('PGDModel created from PGDProblem %s', self.name)

        return self

    def create_interpolation_fcts(self, free_dim, attri, verbose=True):
        '''
            Create interpolation functions for free_dim using 1D-interpolation numpy or fenics functionspaces and save
            it in the self

            Which type is used has to be given in self.mesh[].attributes[].interpolationInfo:
            dict: fenics based: 'name':0  'family':e.g. 'P' or 'CG' (see fenics) 'degree':1 or 2 , '_type' (scalar, vector, tensor) ...
                  scipy: 'name':1 'kind':linear (see scipy)

            :param free_dim: integer array of numbers where the interpolation function should be computed
            :param attri: integer which attribute should be evaluated
        '''
        # check if given coordinates are ok:
        if len(free_dim) > self.num_pgd_var:
            raise ValueError('given number of Dimensions larger then existing Meshes in PGD solution')

        # check if attri is possible
        if attri > len(self.mesh[free_dim[0]].attributes):
            raise ValueError('attribute number not possible')

        for i in range(len(free_dim)):  # loop over free Dimensions
            self.mesh[free_dim[i]].attributes[attri].interpolationfct = list()

            if self.mesh[free_dim[i]].attributes[attri].interpolationInfo["name"] == 0:
                # use 1D interpolation from numpy; only possible if freeDims are one dimensional
                if sum(self.mesh[free_dim[i]].dataY) != 0 and sum(self.mesh[free_dim[i]].dataZ) != 0:
                    raise ValueError('free Dimensions are not 1D, interpolation with INTERP1D not possible')
                else:
                    self.logger.debug('interp1d from scipy will be used for interpolation')
                    string = 'scipy.interpol1d'
                    str_kind = self.mesh[free_dim[i]].attributes[attri].interpolationInfo["kind"]
                    for k in range(self.numModes):  # loop over PGD modes
                        data_X = self.mesh[free_dim[i]].dataX
                        mode = self.mesh[free_dim[i]].attributes[attri].data[k][:, 0]
                        interfct = interpolate.interp1d(data_X, mode, kind=str_kind)
                        self.mesh[free_dim[i]].attributes[attri].interpolationfct.append(interfct)

            elif self.mesh[free_dim[i]].attributes[attri].interpolationInfo["name"] == 1:
                self.logger.debug(
                    'Given FunctionSpaces is used for interpolation, data loaded from _data.h5 files, for free_dim %s',
                    free_dim[i])
                mesh = self.mesh[free_dim[i]].fenics_mesh
                self.logger.debug('Mesh dim: %s ', mesh.topology().dim())
                if verbose: print('test', self.mesh[free_dim[i]].attributes[attri].interpolationInfo)
                str_family = self.mesh[free_dim[i]].attributes[attri].interpolationInfo["family"]
                int_degree = int(self.mesh[free_dim[i]].attributes[attri].interpolationInfo["degree"])
                str_fs_type = self.mesh[free_dim[i]].attributes[attri].interpolationInfo["_type"]

                filepath = os.path.abspath(os.path.join(self.folder, self.mesh[free_dim[i]].name + '_data.h5'))
                hdf = dolfin.HDF5File(dolfin.MPI.comm_world, filepath, 'r')
                self.logger.debug('Finished reading HDF5 file %s', filepath)

                # overwrite fenics mesh because in some case there might be changes
                mesh_hdf = dolfin.Mesh()
                hdf.read(mesh_hdf, 'mesh', False)
                self.logger.debug('mesh loaded from HDF5 file')
                self.mesh[free_dim[i]].fenics_mesh = mesh_hdf

                # if self.mesh[free_dim[i]].attributes[attri].data[0].shape[1] == 1:
                self.logger.debug(
                    'create interpolation functions for dim %s overall field type %s functionspace type %s',
                    free_dim[i], self.mesh[free_dim[i]].attributes[attri].field.lower(), str_fs_type)
                if str_fs_type.lower() == 'scalar':
                    # scalar function 'Scalar' in pxdmf file
                    # Vspace = dolfin.FunctionSpace(mesh, str_family, int_degree)
                    Vspace = dolfin.FunctionSpace(mesh_hdf, str_family, int_degree)
                elif str_fs_type.lower() == 'vector':
                    # Vectorfunction
                    # Vspace = dolfin.VectorFunctionSpace(mesh, str_family, int_degree)
                    Vspace = dolfin.VectorFunctionSpace(mesh_hdf, str_family, int_degree)
                else:
                    raise ValueError('function space type not defined or wrong defined %s' % (str_fs_type))

                for k in range(self.numModes):
                    f_mode = dolfin.Function(Vspace)
                    hdf.read(f_mode, 'MODE_' + str(k))
                    self.mesh[free_dim[i]].attributes[attri].interpolationfct.append(f_mode)
                self.logger.debug('created function with following function space: %s',
                                 f_mode.function_space().ufl_function_space().ufl_element())

                hdf.close()
                # create new data values for modes
                # not in generally useful DG functions ends up in CELL values!!!!
                # self.mesh[free_dim[i]].attributes[attri].fill_data(self.numModes, self.mesh[free_dim[i]], self.mesh[free_dim[i]].attributes[attri].interpolationfct)

            else:
                self.logger.error('interpolation name not defined: %s',
                                  self.mesh[free_dim[i]].attributes[attri].interpolationInfo["name"])

        self.logger.info('Attribute interpolation functions saved')

    def evaluate(self, fixed_dim, free_dim, coord, attri):
        '''
            Reconstruct pgd solution for the fixed variable where all
            other Variables are given

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD class
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :return eval: evaluated solution for the fixed variable NEW as fenics function (old array at vertex values)
        '''

        # check if given coordinates are ok:
        if len(coord) != self.num_pgd_var - 1:
            raise ValueError('given variables are missing or to much, coord=%s <-> num_pgd_var=%s', coord,
                             self.num_pgd_var - 1)

        # only possible if freeDims are one dimensional
        for i in range(len(free_dim)):
            if sum(self.mesh[free_dim[i]].dataY) != 0 and sum(self.mesh[free_dim[i]].dataZ) != 0:
                raise ValueError('free Dimensions are not 1D, interpolation not possible')

        # check if attri is possible
        if attri >= len(self.mesh[fixed_dim].attributes):
            raise ValueError('attribute number not possible')

        # check if interpolation fct exists
        for idx in free_dim:
            if len(self.mesh[idx].attributes[attri].interpolationfct) == 0:
                self.create_interpolation_fcts(free_dim, attri)
                break

        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:  # interp1 function
            # initialize shape of PGD mode
            # same dimension as in data (scalar n x 1 or vector n x 2or3)
            eval = np.zeros(self.mesh[fixed_dim].attributes[attri].data[
                                0].shape)  # nodal or cell values depending on attribute mode_info!!!
            for k in range(self.used_numModes):  # normally used_numModes == numModes - for easy change
                tmp = np.copy(self.mesh[fixed_dim].attributes[attri].data[k])
                tmp_fac = 1.0
                for i in range(len(free_dim)):  # loop over free Dimensions
                    # coord needs right dimension scalar or vector
                    faci = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k](coord[i])
                    tmp *= faci
                    tmp_fac *= faci
                eval += tmp
            return eval

        else:  # fenics fct default!!
            # initialize PGD mode as fenics function same Function space as fixed dim
            # NEW
            eval_fct = dolfin.Function(self.mesh[fixed_dim].attributes[attri].interpolationfct[0].function_space())
            array = np.zeros(len(self.mesh[fixed_dim].attributes[attri].interpolationfct[0].vector()[:]))
            for k in range(self.used_numModes):  # normally used_numModes == numModes - for easy change
                # alternative use numpy array
                array_fixed = np.array(self.mesh[fixed_dim].attributes[attri].interpolationfct[k].vector()[:])
                tmp_fac = 1.0
                for i in range(len(free_dim)):  # loop over free Dimensions
                    # coord needs right dimension scalar or vector
                    faci = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k](coord[i])
                    tmp_fac *= faci
                array += array_fixed * tmp_fac
                eval_fct.vector()[:] = array

            # OLD
            # eval_fct = dolfin.Function(self.mesh[fixed_dim].attributes[attri].interpolationfct[0].function_space())
            # for k in range(self.used_numModes): # normally used_numModes == numModes - for easy change
            #     t_01 = time.time()
            #     tmp = dolfin.Function(self.mesh[fixed_dim].attributes[attri].interpolationfct[k].function_space())
            #     fct_fixed = self.mesh[fixed_dim].attributes[attri].interpolationfct[k]
            #     tmp_fac = 1.0
            #     for i in range(len(free_dim)):  # loop over free Dimensions
            #         # coord needs right dimension scalar or vector
            #         faci = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k](coord[i])
            #         tmp_fac *= faci
            #     tmp.vector().axpy(tmp_fac, fct_fixed.vector())
            #     eval_fct.vector().axpy(1, tmp.vector())
            #     t_02 = time.time()
            #     print('one step in loop',t_02-t_01)

            return eval_fct

    def evaluate_sensor_response(self, fixed_dim, free_dim, coord, attri, sensor_points):
        '''
            Reconstruct pgd solution for the fixed variable where all
            other Variables are given at specific points of the fixed dim!!!!

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD class
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :param sensor_points: coordintes at which point the fixed dim has to be evaluate
            :return eval: evaluated solution for the fixed variable at points AS ARRAY
        '''

        # check if given coordinates are ok:
        if len(coord) != self.num_pgd_var - 1:
            raise ValueError('given variables are missing or to much, coord=%s <-> num_pgd_var=%s', coord,
                             self.num_pgd_var - 1)

        # only possible if freeDims are one dimensional
        for i in range(len(free_dim)):
            if sum(self.mesh[free_dim[i]].dataY) != 0 and sum(self.mesh[free_dim[i]].dataZ) != 0:
                raise ValueError('free Dimensions are not 1D, interpolation not possible')

        # check if attri is possible
        if attri >= len(self.mesh[fixed_dim].attributes):
            raise ValueError('attribute number not possible')

        # check if interpolation fct exists
        for idx in free_dim:
            if len(self.mesh[idx].attributes[attri].interpolationfct) == 0:
                self.create_interpolation_fcts(free_dim, attri)
                break

        # evaluate fixed dim modes at sensor points
        # # old implementation not time efficient only for understanding
        # t1=time.time()
        # dim1 = len(sensor_points)
        # dim2 = len(self.mesh[fixed_dim].attributes[attri].interpolationfct[0](sensor_points[0,:]))
        # eval_array = np.zeros((dim1,dim2))
        # probes = Probes(sensor_points.flatten(), self.mesh[fixed_dim].attributes[attri].interpolationfct[0].function_space())
        # for k in range(self.used_numModes): # normally used_numModes == numModes - for easy change
        #     eval_fct = self.mesh[fixed_dim].attributes[attri].interpolationfct[k]
        #     tmp =ftool_probe(sensor_points, eval_fct) # values at sensor points for mode in x
        #     tmp_fac = 1.0
        #     for i in range(len(free_dim)):  # loop over free Dimensions
        #         # coord needs right dimension scalar or vector
        #         faci = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k](coord[i])
        #         tmp_fac *= faci
        #     eval_array += tmp * tmp_fac
        # t2=time.time()
        # print('eval_array',eval_array,t2-t1)

        # new implementation to speed things up
        # first: fixed modes (usually x) at sensor points:
        # from chache to save time
        eval_fixedmode = self.eval_fixed_modes(sensor_points, fixed_dim, attri)

        self.logger.debug('used number of Modes in evaluation: %s', self.used_numModes)

        # evaluate free_dim modes at coord
        tmp = np.ones(self.used_numModes)
        for i in range(len(free_dim)):
            # probes = Probes(coord[i].flatten(), self.mesh[free_dim[i]].attributes[attri].interpolationfct[0].function_space()) # class creation to expensive!!
            tmp_i = np.zeros(self.used_numModes)
            for k in range(self.used_numModes):  # normally used_numModes == numModes - for easy change
                # probes(self.mesh[free_dim[i]].attributes[attri].interpolationfct[k])
                # tmp_i = probes.array()
                tmp_i[k] = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k](coord[i])
            tmp *= tmp_i

        # multiply and sum up
        eval_array = np.zeros_like(eval_fixedmode)
        # different for dim of function
        self.logger.debug('shape of eval_fixedmode is %s (len %s)', eval_fixedmode.shape, len(eval_fixedmode.shape))
        # (eval_fixedmode[:, :, :] * tmp)[:, :, 19]  # same as eval_fixedmode[:,:,19]*tmp[19]
        # eval_array = np.sum(eval_fixedmode[:,:,:]* tmp,axis=2)
        if self.numModes == 1:
            # special case only one Mode -- changed dimensions
            if len(eval_fixedmode.shape) == 2:  # Vectorfield
                eval_array = eval_fixedmode * tmp[0]
            elif len(eval_fixedmode.shape) == 1:  # Scalarfield
                eval_array = eval_fixedmode * tmp[0]
        else:
            # usually the case
            if len(eval_fixedmode.shape) == 3:  # Vectorfield
                eval_array = np.sum(eval_fixedmode[:, :, 0:self.used_numModes] * tmp, axis=2)
            elif len(eval_fixedmode.shape) == 2:  # Scalarfield
                eval_array = np.sum(eval_fixedmode[:, 0:self.used_numModes] * tmp, axis=1)

        return eval_array

    def evaluate_min(self, fixed_dim, free_dim, coord, attri, *args, **kwargs):
        '''
            Compute Minimum of pgd solution for eval (s. above)
            for the fixed Variable where all other Variables are given

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD cls
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :return eval: evaluated solution for the fixed variable
        '''
        # should be identical to this huge function footprint
        # evaluate_min = lambda *args, **kwargs: evaluate(*args, **kwargs).min()
        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:  # interp1 function
            return self.evaluate(fixed_dim, free_dim, coord, attri).min()
        else:  # fenics fct default!!
            return self.evaluate(fixed_dim, free_dim, coord, attri).vector()[:].min()

    def evaluate_min_abs(self, fixed_dim, free_dim, coord, attri, *args, **kwargs):
        '''
            Compute Minimum of the absolute values of the pgd solution for
            the fixed variable where all other Variables are given

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD cls
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :return eval: evaluated solution for the fixed variable
        '''
        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:  # interp1 function
            return abs(self.evaluate(fixed_dim, free_dim, coord, attri)).min()
        else:  # fenics fct default!!
            return abs(self.evaluate(fixed_dim, free_dim, coord, attri).vector()[:]).min()

    def evaluate_max(self, fixed_dim, free_dim, coord, attri, *args, **kwargs):
        '''
            Compute Maxium of pgd solution for the fixed Variable where all other Variables are given

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD cls
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :return eval: evaluated solution for the fixed variable

        '''
        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:  # interp1 function
            return self.evaluate(fixed_dim, free_dim, coord, attri).max()
        else:  # fenics fct default!!
            return self.evaluate(fixed_dim, free_dim, coord, attri).vector()[:].max()

    def evaluate_max_abs(self, fixed_dim, free_dim, coord, attri, *args, **kwargs):
        '''
            Compute Maxium of absolute values of the pgd solution for the fixed Variable
            where all other Variables are given

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD cls
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :return eval: evaluated solution for the fixed variable
        '''
        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:  # interp1 function
            return abs(self.evaluate(fixed_dim, free_dim, coord, attri)).max()
        else:  # fenics fct default!!
            return abs(self.evaluate(fixed_dim, free_dim, coord, attri).vector()[:]).max()

    def evaluate_max_norm(self, fixed_dim, free_dim, coord, attri, *args, **kwargs):
        '''
            Compute maxium of the norm at each dof point for VectorFunctions of pgd solution for the fixed
            variable where all other Variables are given

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD cls
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :return eval: evaluated solution for the fixed variable
        '''
        max_norm = 0

        # evaluate pgd solution
        new = self.evaluate(fixed_dim, free_dim, coord, attri)

        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:
            # compute norm for each element in New
            normed = np.zeros(len(new))
            for i in range(len(new)):
                normed[i] = np.linalg.norm(new[i, :])
            max_norm = max(normed)

        else:  # default with fenics functions

            if new.function_space().mesh().geometry().dim() == 1:
                raise ValueError('Function is 1D use evaluate_max instead!!')
            elif new.function_space().mesh().geometry().dim() > 1:
                new_xyz = new.split()

                # compute norm for each element in new_xyz
                dof_normed = np.zeros(len(new_xyz))
                for i in range(len(new_xyz)):
                    dof_normed[i] = np.linalg.norm(new_xyz[i, :])
                max_norm = max(dof_normed)

        return max_norm

    def evaluate_abs_value(self, fixed_dim, free_dim, coord, attri, *args, **kwargs):
        '''
            Compute absolute value of u at explicit position

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD cls
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :return eval: evaluated solution for the fixed variable
        '''
        '''compute absolute value of u at explicit position'''

        # evaluate pgd solution
        new = self.evaluate(fixed_dim, free_dim, coord, attri)

        return abs(new(self.pos)).max()

    def create_derivation_fct(self, free_dim, attri):
        '''
            Create derivation from interpolation functions for free_dim using fenics functionspaces and save
            it in the self

            type for derivation alsways DG
            order of derivation order function -1

            :param free_dim: integer array of numbers where the derivative of the interpolation function should be computed
            :param attri: integer which attribute should be evaluated
        '''
        self.logger.debug('create derivation functions for free_dim=%s', free_dim)
        # check if given coordinates are ok:
        if len(free_dim) > self.num_pgd_var:
            raise ValueError('given number of Dimensions larger then existing Meshes in PGD solution')

        # check if attri is possible
        if attri > len(self.mesh[free_dim[0]].attributes):
            raise ValueError('attribute number not possible')

        for i in range(len(free_dim)):  # loop over free Dimensions
            self.mesh[free_dim[i]].attributes[attri].derivationfct = list()

            if self.mesh[free_dim[i]].attributes[attri].interpolationInfo["name"] == 0:
                raise ValueError('derivation for interp1 functions not implemented (only fencis functions)')

            elif self.mesh[free_dim[i]].attributes[attri].interpolationInfo["name"] == 1:
                self.logger.debug(
                    'DG function space and order-1 is used for derivation of interpolation fct for free_dim %s',
                    free_dim[i])
                mesh = self.mesh[free_dim[i]].fenics_mesh
                self.logger.debug('Mesh dim: %s ', mesh.topology().dim())

                # int_degree = int(self.mesh[free_dim[i]].attributes[attri].interpolationInfo["degree"])-1 # one order less then original mode
                int_degree = self.mesh[free_dim[i]].attributes[attri].interpolationfct[
                                 0].function_space().ufl_element().degree() - 1

                # generate function space for derivation dependent to the one for the mode
                check_string = str(self.mesh[free_dim[i]].attributes[attri].interpolationfct[
                                       0].function_space().ufl_function_space().ufl_element())
                self.logger.debug('Functionspace of dim %s is from type %s', free_dim[i], check_string)
                if check_string.split(' ')[0] == '<vector':
                    Vspace = dolfin.TensorFunctionSpace(self.mesh[free_dim[i]].fenics_mesh, 'DG', int_degree)
                    for k in range(self.numModes):
                        f_mode = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k]
                        d_mode = dolfin.project(dolfin.grad(f_mode), Vspace)
                        self.mesh[free_dim[i]].attributes[attri].derivationfct.append(d_mode)
                    self.logger.info('derivation from vector valued function --> tensor --> NOT YET TESTED!!!')
                    self.logger.debug('Functionspace of dim %s is from type %s', free_dim[i], check_string)
                    self.logger.debug('Derivation-Functionspace of dim %s is from type %s', free_dim[i],
                                      d_mode.function_space().ufl_function_space().ufl_element())
                else:
                    Vspace = dolfin.FunctionSpace(self.mesh[free_dim[i]].fenics_mesh, 'DG', int_degree)
                    # compute derivation for each mode and put into list ONLY 1 D ones
                    # print('check', free_dim[i], len(self.mesh[free_dim[i]].attributes[attri].interpolationfct), int_degree)
                    # print('check', self.numModes)
                    for k in range(self.numModes):
                        # print('check k', k)
                        f_mode = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k]
                        d_mode = dolfin.project(f_mode.dx(0), Vspace)
                        self.mesh[free_dim[i]].attributes[attri].derivationfct.append(d_mode)
                    self.logger.debug('Derivation-Functionspace of dim %s is from type %s', free_dim[i],
                                      d_mode.function_space().ufl_function_space().ufl_element())

            else:
                self.logger.error('interpolation name not defined: %s',
                                  self.mesh[free_dim[i]].attributes[attri].interpolationInfo["name"])

        self.logger.info('derivations for dimensions %s are saved in PGD instance' % (free_dim))

    def evaluate_derivative(self, fixed_dim, free_dim, coord, attri, d_dim):
        '''
            Reconstruct derivation of pgd solution against d_dim for the fixed variable where all
            other Variables are given

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD class
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :param d_dim: derivation dimension
            :return eval: evaluated solution for the fixed variable NEW as fenics function (old array at vertex values)
        '''

        # check if given coordinates are ok:
        if len(coord) != self.num_pgd_var - 1:
            raise ValueError('given variables are missing or to much, coord=%s <-> num_pgd_var=%s', coord,
                             self.num_pgd_var - 1)

        # only possible if freeDims are one dimensional
        for i in range(len(free_dim)):
            if sum(self.mesh[free_dim[i]].dataY) != 0 and sum(self.mesh[free_dim[i]].dataZ) != 0:
                raise ValueError('free Dimensions are not 1D, interpolation not possible')

        # check if attri is possible
        if attri >= len(self.mesh[fixed_dim].attributes):
            raise ValueError('attribute number not possible')

        # check if interpolation fct exists
        for idx in free_dim:
            if len(self.mesh[idx].attributes[attri].interpolationfct) == 0:
                self.create_interpolation_fcts(free_dim, attri)
                # break

        # check if fixed_dim == d_dim
        if fixed_dim == d_dim:
            raise ValueError('derivation against fixed dim not possible in the moment')

        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:  # interp1 function
            self.logger.error('derivation for interp1 functions not implemented (only fencis functions)')
            raise ValueError('derivation for interp1 functions not implemented (only fencis functions)')
        else:  # fenics fct default!!
            # initialize PGD mode as fenics function same Function space as fixed dim
            eval_fct = dolfin.Function(self.mesh[fixed_dim].attributes[attri].interpolationfct[0].function_space())
            for k in range(self.used_numModes):  # normally used_numModes == numModes - for easy change
                tmp = dolfin.Function(self.mesh[fixed_dim].attributes[attri].interpolationfct[k].function_space())
                fct_fixed = self.mesh[fixed_dim].attributes[attri].interpolationfct[k]
                tmp_fac = 1.0
                for i in range(len(free_dim)):  # loop over free Dimensions
                    if free_dim[i] == d_dim:
                        # compute derivation and evaluate this
                        self.logger.debug('use derivation from d_dim %s mode number %s', d_dim, k)
                        faci = self.mesh[free_dim[i]].attributes[attri].derivationfct[k](coord[i])
                    else:
                        # coord needs right dimension scalar or vector
                        faci = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k](coord[i])
                    tmp_fac *= faci
                tmp.vector().axpy(tmp_fac, fct_fixed.vector())
                eval_fct.vector().axpy(1, tmp.vector())

            return eval_fct

    def evaluate_derivative_sensor_response(self, fixed_dim, free_dim, coord, attri, d_dim, sensor_points):
        '''
            Reconstruct derivation of pgd solution against d_dim for the fixed variable where all
            other Variables are given at specif sensor data!!

            :param fixed_dim: integer number of fixed variable
            :param free_dim: int array of numbers which are not fixed in the order of the PGD modes safed in the PGD class
            :param coord: array with the explicitly given variables corresponding to free_dim
            :param attri: integer which attribute should be evaluate
            :param d_dim: derivation dimension
            :param sensor_points: coordintes at which point the fixed dim has to be evaluate
            :return eval: evaluated solution for the fixed variable NEW as fenics function (old array at vertex values)
        '''

        # check if given coordinates are ok:
        if len(coord) != self.num_pgd_var - 1:
            raise ValueError('given variables are missing or to much, coord=%s <-> num_pgd_var=%s', coord,
                             self.num_pgd_var - 1)

        # only possible if freeDims are one dimensional
        for i in range(len(free_dim)):
            if sum(self.mesh[free_dim[i]].dataY) != 0 and sum(self.mesh[free_dim[i]].dataZ) != 0:
                raise ValueError('free Dimensions are not 1D, interpolation not possible')

        # check if attri is possible
        if attri >= len(self.mesh[fixed_dim].attributes):
            raise ValueError('attribute number not possible')

        # check if interpolation fct exists
        for idx in free_dim:
            if len(self.mesh[idx].attributes[attri].interpolationfct) == 0:
                self.create_interpolation_fcts(free_dim, attri)
                # break

        # check if fixed_dim == d_dim
        if fixed_dim == d_dim:
            raise ValueError('derivation against fixed dim not possible in the moment')

        if self.mesh[free_dim[0]].attributes[attri].interpolationInfo["name"] == 0:  # interp1 function
            self.logger.error('derivation for interp1 functions not implemented (only fencis functions)')
            raise ValueError('derivation for interp1 functions not implemented (only fencis functions)')
        else:  # fenics fct default!!
            # new implementation to speed things up
            # first: fixed modes (usually x) at sensor points:
            # from chache to save time
            eval_fixedmode = self.eval_fixed_modes(sensor_points, fixed_dim, attri)

            # evaluate free_dim modes at coord
            tmp = np.ones(self.used_numModes)
            for i in range(len(free_dim)):
                tmp_i = np.zeros(self.used_numModes)
                if free_dim[i] == d_dim:
                    for k in range(self.used_numModes):  # normally used_numModes == numModes - for easy change
                        tmp_i[k] = self.mesh[free_dim[i]].attributes[attri].derivationfct[k](coord[i])
                else:
                    for k in range(self.used_numModes):
                        tmp_i[k] = self.mesh[free_dim[i]].attributes[attri].interpolationfct[k](coord[i])
                tmp *= tmp_i

            # multiply and sum up
            # different for dim of function
            self.logger.debug('shape of eval_fixedmode is %s', eval_fixedmode.shape)
            # (eval_fixedmode[:, :, :] * tmp)[:, :, 19]  # same as eval_fixedmode[:,:,19]*tmp[19]
            # eval_array = np.sum(eval_fixedmode[:,:,:]* tmp,axis=2)
            if self.numModes == 1:
                # special case only one Mode -- changed dimensions
                if len(eval_fixedmode.shape) == 2:  # Vectorfield
                    eval_array = eval_fixedmode * tmp[0]
                elif len(eval_fixedmode.shape) == 1:  # Scalarfield
                    eval_array = eval_fixedmode * tmp[0]
            else:
                # usually the case
                if len(eval_fixedmode.shape) == 3:  # Vectorfield
                    eval_array = np.sum(eval_fixedmode[:, :, 0:self.used_numModes] * tmp, axis=2)
                elif len(eval_fixedmode.shape) == 2:  # Scalarfield
                    eval_array = np.sum(eval_fixedmode[:, 0:self.used_numModes] * tmp, axis=1)

            return eval_array

    def error_computation(self, points, analytic, some, param, free_dims=None, fixed_dim=0):
        '''
            compute errors between analytic solution and PGD solution for a given set of PGD_variables
            compute error at dof points!!
            :param points: random values of PGD extra coordinates list of list of len(self.num_pgd_var-1)
            :param analytic: function to compute the analytic displacement Expression with input: one point, param

            :param some: dictionary {'mesh_order':list, 'mesh_type':list, 'attri':value}
            :param param: parameter dictionary for analytic function

            :param free_dims: pgd dimesnions which are free corresponding to the points array (default None -> [1,2,3 .., num_pgd_var])
            :param fixed_dim: pgd dimension over which error will be computed (default 0 --> x space)
            :return: error_L2 and error_max
        '''

        # define interpolation for pgd solution
        if free_dims == None:
            free_dims = np.arange(1, self.num_pgd_var)  # default values all except 0

        if self.mesh[free_dims[0]].attributes[some['attri']].interpolationfct == []:
            self.logger.debug('create interpolation functions for modes')
            for k in range(self.num_pgd_var):
                info = {'name': 1,
                        'family': some['mesh_type'][k],
                        'degree': some['mesh_order'][k]
                        }
                self.mesh[k].attributes[some['attri']].interpolationInfo = info

            # create interpolation functions for all PGD COORD
            self.create_interpolation_fcts(np.arange(0, self.num_pgd_var), some['attri'])

        # evaluate pgd and analytic solution at dof points and compute errors
        # Function space for analytic solution
        mesh_ana = self.evaluate(fixed_dim, free_dims, points[0], some['attri']).function_space().mesh()
        V_ana = dolfin.FunctionSpace(mesh_ana, some['mesh_type'][fixed_dim], some['mesh_order'][fixed_dim])

        error_L2 = list()
        error_max = list()
        # error_fenics = list()
        for pp in points:
            if self.mesh[0].typElements.lower() == 'polyline':
                disp_PGD = self.evaluate(fixed_dim, free_dims, pp, some['attri'])
                # print('in error',disp_PGD.vector()[:])
                ana_exp = analytic(pp, param)
                disp_ana = dolfin.interpolate(ana_exp, V_ana)
                # print('in error',disp_ana.vector()[:])
                # print('? ', len(disp_PGD.vector()[:]), len(disp_ana.vector()[:]))
                diff = disp_PGD.vector()[:] - disp_ana.vector()[:]
                error_L2.append(la.norm(diff) / la.norm(disp_ana.vector()[:]))
                error_max.append(abs(diff).max() / abs(disp_ana.vector()[:]).max())
                # using fenics error function geht nicht!! Value shapes do not match ??
                # error_fenics.append(dolfin.errornorm(disp_PGD,disp_ana,'L2')/dolfin.norm(disp_ana,'L2'))
            else:
                err = 'error computation currently only for 1D'
                self.logger.error(err)
                raise NotImplementedError(err)

        return error_L2, error_max

    def save_modes_latex(self, folder, attri, prefix='_'):
        '''
            save 1D modes in a file which can be used in latex for 1D plotting (dof values!!)
            :param folder: where to save
            :param attri: modes of which attribute
            :param prefix: prefix for name
            :return: save files in folder with name modes_<attri>_<name_coord> including [dof_coord, mode1, mode2, ...]
        '''
        name = 'modes_%s_%i_%s.out'
        for k in range(self.num_pgd_var):
            if self.mesh[k].typElements.lower() == 'polyline':
                self.logger.info('save modes for dimension %s as latex file', k)
                dof_number = len(self.mesh[k].attributes[attri].interpolationfct[0].vector()[:])
                out_k = np.zeros((dof_number, self.numModes + 1))
                dof_coord = self.mesh[k].attributes[attri].interpolationfct[
                                0].function_space().tabulate_dof_coordinates().reshape((-1, 1))[:, 0]
                for m in range(self.numModes):
                    zip_data = zip(dof_coord, self.mesh[k].attributes[attri].interpolationfct[m].vector()[:])
                    tmp = sorted(zip_data, key=lambda x: x[0])
                    dof_coord_sorted, mode_sorted = map(list, zip(*tmp))
                    out_k[:, 0] = dof_coord_sorted
                    out_k[:, m + 1] = mode_sorted
                np.savetxt(os.path.join(folder, name % (prefix, attri, self.mesh[k].info[1][1])), out_k, delimiter=',')

    def reconstruct_solution_tensor(self, attri):
        '''
            Reconstruct the full tensor of all solution of attribute (attri[integer])
            stored in the PGD solution: sum kronecker products of PGD modes

            :param  attribute number
            :return tensor of all solutions
        '''

        factorPGD = list()

        for i in range(self.num_pgd_var):
            if self.mesh[i].attributes[attri].field.lower() == 'scalar':
                # save 1D modes in array
                tmp = np.zeros((self.mesh[i].numNodes, self.numModes))
                for k in range(self.numModes):  # fill with 1D modes
                    tmp[:, k] = self.mesh[i].attributes[attri].data[k][:, 0]

            if self.mesh[i].attributes[attri].field.lower() == 'vector':
                # vector field will be saved as 1D mode reshape
                tmp = np.zeros((self.mesh[i].numNodes * 3, self.numModes))
                for k in range(self.numModes):
                    vect = self.mesh[i].attributes[attri].data[k].reshape(self.mesh[i].numNodes * 3, 1)
                    tmp[:, k] = vect[:, 0]

            factorPGD.append(tmp)

        self.logger.info('PGD factors len %i shape %s', len(factorPGD), [f.shape for f in factorPGD])
        x = tensorly.kruskal_to_tensor(factorPGD)
        return x


class PGDAttribute(object):

    def __init__(self, num_modes=0, mesh=None, pgd_modes=None, modes_info=None):
        '''

            :param modes_info: [name, type, field (Scalar, Vector)] of mode
            :param num_modes: from above instance PGDModel
            :param mesh: PGDMesh instance
            :param pgd_modes: form above instance PGDModel
        '''

        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        if modes_info is not None:
            self.name = modes_info[0]  # Name of attribute e.g. U_x
            self._type = modes_info[1]  # Typ of Attribute: Node or Cell
            self.field = modes_info[2]  # scalar or vector field for overall variable (NOT FOR PGD variables!!)
        self.data = list()  # PGD modes [Number of values (Nodes or Cell) x 1 (Skalarfield) or 3 (Vectorfield)] x PGDsolution.numModes]

        self.interpolationInfo = {'name': 1}  # what kind of interpolation method should be used
        # for interp1d:  {'name': 0, kind='linear'} (kind can also be 'quadratic', 'cubic' ... s. scipy)
        # for FunctionSpace: {'name': 1, 'family': 'P', 'degree': 1, '_type': scalar} possible inputs for FunctionSpaces in fenics/dolfin
        # default fenics functions!!!
        self.interpolationfct = list()
        self.derivationfct = list()
        for ctr in range(num_modes):
            self.interpolationfct.append(pgd_modes[ctr])  # fenics function space functions of PGD modes

        self.fill_data(num_modes, mesh, pgd_modes)

        # for ctr in range(num_modes):
        #     mode = np.zeros((mesh.numNodes, mesh.meshdim))
        #     if mesh.meshdim == 1:
        #         mode[:, 0] = pgd_modes[ctr].compute_vertex_values()[:]
        #     elif mesh.meshdim == 2:
        #         x, y = pgd_modes[ctr].split()
        #         mode[:, 0] = x.compute_vertex_values()[:]
        #         mode[:, 1] = y.compute_vertex_values()[:]
        #     elif mesh.meshdim == 3:
        #         x, y, z = pgd_modes[ctr].split()
        #         mode[:, 0] = x.compute_vertex_values()[:]
        #         mode[:, 1] = y.compute_vertex_values()[:]
        #         mode[:, 2] = y.compute_vertex_values()[:]
        #
        #     self.data.append(mode)
        #     self.interpolationfct.append(pgd_modes[ctr])  # fenics function space functions of PGD modes

    def fill_data(self, num_modes, mesh, pgd_modes):
        # create data at nodes for each mode
        self.logger.debug('fill_data for attribute based on fenics function and mode_info[1] (Node or Cell) and mode_info[2] (scalar or vector)')
        self.data = list()
        for ctr in range(num_modes):
            if self._type.lower() == 'node':
                mode = np.zeros((mesh.numNodes, mesh.meshdim))
            elif self._type.lower() == 'cell':
                mode = np.zeros((mesh.numElements, mesh.meshdim))
            else:
                raise ValueError(' Error in filling attribute data: self._type not known')

            # fill data with nodal values or cell values for different mesh dims
            if self.field.lower() == 'scalar' and self._type.lower() == 'node':
                mode[:, 0] = pgd_modes[ctr].compute_vertex_values()[:]
            elif self.field.lower == 'vector':
                # differentiate between dimensions
                if mesh.meshdim == 1:
                    if self._type.lower() == 'node':
                        mode[:, 0] = pgd_modes[ctr].compute_vertex_values()[:]
                    elif self._type.lower() == 'cell':
                        mode[:, 0] = pgd_modes[ctr].vector()[:]
                elif mesh.meshdim == 2:
                    if self._type.lower() == 'node':
                        x, y = pgd_modes[ctr].split()
                        mode[:, 0] = x.compute_vertex_values()[:]
                        mode[:, 1] = y.compute_vertex_values()[:]
                    elif self._type.lower() == 'cell':
                        self.logger.error('CELL VALUES NOT IMPLEMENTED YET!!')
                        raise ValueError()
                elif mesh.meshdim == 3:
                    if self._type.lower() == 'node':
                        x, y, z = pgd_modes[ctr].split()
                        mode[:, 0] = x.compute_vertex_values()[:]
                        mode[:, 1] = y.compute_vertex_values()[:]
                        mode[:, 2] = y.compute_vertex_values()[:]
                    elif self._type.lower() == 'cell':
                        self.logger.error('CELL VALUES NOT IMPLEMENTED YET!!')
                        raise ValueError()

            self.data.append(mode)

        return self

    def print_info(self):
        print('\n')
        print('summary of PGDAttribute class')
        print('----------------------------')
        print('name:                        ', self.name)
        print('type:                        ', self._type)
        print('field type:                  ', self.field)
        print('len of data:                 ', len(self.data))
        print('interpolationInfo:           ', self.interpolationInfo)
        print('len of interpolation fct     ', len(self.interpolationfct))
        for i in range(len(self.data)):
            print('     shape data ', i, ':   ', self.data[i].shape)
        print('\n')


class PGDMesh(object):
    '''
        A mesh class wrapping the fenics mesh. Mostly used for saving and loading results
    '''

    def __init__(self, name=None, mesh=None, name_coord=None, pgd_modes=None, num_modes=0, modes_info=None):
        '''
            :param name:
            :param mesh: fenics mesh function
            :param name_coord: name of pgd coordinates
            :param pgd_modes: pgd modes list
            :param num_modes: number of pgd modes
            :param modes_info: passed to attribute [name, type, field (Scalar, Vector)] of mode
        '''
        self.logger = logging.getLogger(__name__ + '.' + self.__class__.__name__)

        # usally mesh data
        self.name = name  # name of mesh PGD1 ...
        self.meshdim = mesh.topology().dim() if mesh is not None else 0  # mesh dimension known from fenics mesh!!
        self.info = [self.meshdim, name_coord, '-?-']  # list contains Dimension, Name and Metrics
        self.numElements = mesh.num_cells() if mesh is not None else None
        self.numNodes = mesh.num_vertices() if mesh is not None else 0
        self.topology = mesh.cells() if mesh is not None else None  # Mesh data: element point connection [Number of elements x points per element]
        self.typGeometry = 'XYZ'  # Typ of Geometry: default XYZ
        self.dataX = np.zeros(self.numNodes)  # xyz data of points [Number of points x 1]
        self.dataY = np.zeros(self.numNodes)
        self.dataZ = np.zeros(self.numNodes)
        self.fenics_mesh = mesh

        # Coordinates
        if self.meshdim == 1:
            # 1D mesh
            self.dataX = mesh.coordinates()[:, 0]
            self.typElements = 'Polyline'
        elif self.meshdim == 2:
            # 2D mesh
            xy = mesh.coordinates()[:]
            self.dataX = xy[:, 0]
            self.dataY = xy[:, 1]
            self.typElements = "Triangle"
        elif self.meshdim == 3:
            # s3D mesh
            xyz = mesh.coordinates()
            self.dataX = xyz[:, 0]
            self.dataY = xyz[:, 1]
            self.dataZ = xyz[:, 2]
            self.typElements = "Tetrahedron"

        # specific mesh data for PGD
        self.attributes = list()  # PGD modes for e.g U or sigma ...
        att = PGDAttribute(num_modes, self, pgd_modes, modes_info=modes_info)
        self.attributes.append(att)

    def print_info(self):
        print('\n')
        print('summary of PGDMesh class')
        print('----------------------------')
        print('name:                            ', self.name)
        print('info:                            ', self.info)
        print('number of Elements:              ', self.numElements)
        print('number of Nodes:                 ', self.numNodes)
        print('type of Elements:                ', self.typElements)
        print('shape of topology data:          ', self.topology.shape)
        print('type of geometry:                ', self.typGeometry)
        print('shape of geometry data (X,Y,Z):  ', self.dataX.shape, self.dataY.shape, self.dataZ.shape)
        print('number of saved attributes:      ', len(self.attributes))
        print('fenics Mesh                      ', self.fenics_mesh)
        print('\n')
        
class PGDErrorComputation(object):
    
    def sampling_LHS(self,n_sample, n_var, l_bound, r_bound):
        
        '''Sampling is done using Latin Hypercube sampling method:
            # n_sample: Number of samples.
            # n_var: Number of variables
            # l_bound and r bound: The ranges of the variable'''
        
        sampler = qmc.LatinHypercube(d=n_var, seed = 3452)
        sample = sampler.random(n=n_sample)
        # l_bounds = [input_mesh[1][0][0], input_mesh[2][0][0], input_mesh[3][0][0]] # Minimum boundary
        # r_bounds = [input_mesh[1][0][1], input_mesh[2][0][1], input_mesh[3][0][1]] # Maximum boundary
        data_test = qmc.scale(sample, l_bound, r_bound) # Scale the sample
        data_test = data_test.tolist()
        
        return data_test

    def compute_SampleError(self,u_FOM, u_PGD):
            
        ''' The error between the Full-Order Model (Analytical or FEM)
        and PGD solution is computed selecting different snapshots. The
        error is computed using the norm2:
            # u_FOM: The exact solution.
            # u_PGD: The solution coputed through pgdrome'''
        
        # PGD solution
        #---------------------------------
        # u_pgd = pgd_solution.evaluate(0, [1,2,3], [data_test[i][0],data_test[i][1],data_test[i][2]], 0) # The last zero means to compute displacements
        # uvec_pgd = u_pgd.compute_vertex_values()
        
        # Compare PGD and FOM
        #---------------------------------
        residual = u_PGD-u_FOM
        if type(u_FOM)==float:
            error = residual/u_FOM
        else:
            error = np.linalg.norm(residual,2)/np.linalg.norm(u_FOM,2)
         
        return error
