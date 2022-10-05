'''
    test for PGD Class functions

    using an analytic pgd solution of form:
        u(x,E,L)
        PGD CO: x: space; E: E-modul, L: load factor
        attribute 0: U: u(x,E,L)= 1/2A(lae*x-x**2) 1/E  L*n
        attribute 1: sig: sig(x,E,L)=1/2A(lae-2x)  E/E  L*n

        with modes and meshes given as np.arrays NO fenics dependency!!
'''
import unittest
import numpy as np

from pgdrome.model import PGD, PGDAttribute, PGDMesh, PGDErrorComputation


class u_analytic():
    # reference model u analytic
    # analytic solution of u truss: u = 1/(2*EA) (-x**2+laen*x) *L*n '''
    # with A:crosssection, E:Emodul, lae: Stablength, x:position, L:loadfactor, n:loadamplitude
    def __init__(self, x=[], p={}):
        self.x = x
        self.param = p

    def __call__(self, values):
        E=values[0]
        L=values[1]
        return 1.0 / 2.0 * 1 / (self.param['A'] * E) * (self.param['lae'] * self.x - self.x ** 2) * L * self.param['n']

class sig_analytic():
    # reference model sig analytic
    # Analytic solution of sig truss: E* u'
    # A:crosssection, E:Emodul, lae: Stablength, x:position, L:loadfactor, n:loadamplitude
    def __init__(self, x=[], p={}):
        self.x = x
        self.param = p

    def __call__(self, values):
        E=values[0]
        L=values[1]
        return 1.0 / 2.0 * 1 / (self.param['A']) * (self.param['lae'] - 2 * self.x) * L * self.param['n']



''' separtion of analytic u analytically --> modes'''
def analytic_mode_UX(x, param):
    return 1.0 / (2.0 * param['A']) * (param['lae'] * x - x ** 2)


def analytic_mode_UE(E, param):
    return 1.0 / E


def analytic_mode_UL(L, param):
    return L * param['n']


''' separation of  analytic sigma analytically  --> modes'''
def analytic_mode_SX(x, param):
    return 1.0 / (2.0 * param['A']) * (param['lae'] - 2 * x)


def analytic_mode_SE(E, param):
    # return E/E
    return 1.0


def analytic_mode_SL(L, param):
    return L * param['n']


def create_example_pgd_solution(param):
    '''
        create a special example pgd solution 1D example u(x,E,L)
        PGD CO: x: pysical dim; E: E-modul, L: load factor
        attribute 0: U: u(x,E,L)= 1/2A(lae*x-x**2) 1/E  L*n
        attribute 1: sig: sig(x,E,L)=1/2A(lae-2x)  E/E  L*n
        with param dict= 'A', 'lae' and 'n'
    '''
    pgdtest = PGD()
    pgdtest.name = 'test'
    pgdtest.numModes = 1
    pgdtest.used_numModes = pgdtest.numModes
    Grids = list()

    # load/create num_pgd_var Meshes as PGDMesh-Class with modes as PGDAttribute-Class

    # first forward_models variable: X
    Grid1 = PGDMesh('PGD1')
    Grid1.info = [1, 'X', 'm']
    Grid1.numNodes = 11
    Grid1.numElements = 10

    # create Coordinates
    Grid1.dataX = np.linspace(0, 1, num=Grid1.numNodes)
    Grid1.dataY = np.zeros(Grid1.numNodes)  # default zero
    Grid1.dataZ = np.zeros(Grid1.numNodes)  # default zero

    # create 1D Mesh
    Grid1.typElements = 'Polyline'
    temp = list()
    for i in range(Grid1.numElements):
        temp.append([i, i + 1])
    Grid1.topology = temp

    # load Attributes
    # displacement
    Attributes = list()
    newAttr = PGDAttribute()
    newAttr.name = 'U_x'
    newAttr._type = 'Node'
    newAttr.field = 'Scalar'
    newAttr.data = list()
    newAttr.data.append(np.zeros((Grid1.numNodes, 1)))  # first mode
    for i in range(Grid1.numNodes):
        newAttr.data[0][i, 0] = analytic_mode_UX(Grid1.dataX[i], param)
    Attributes.append(newAttr)
    # stress
    newAttr = PGDAttribute()
    newAttr.name = 'Sig_x'
    newAttr._type = 'Node'
    newAttr.field = 'Scalar'
    newAttr.data = list()
    newAttr.data.append(np.zeros((Grid1.numNodes, 1)))  # first mode
    for i in range(Grid1.numNodes):
        newAttr.data[0][i, 0] = analytic_mode_SX(Grid1.dataX[i], param)
    Attributes.append(newAttr)

    Grid1.attributes = Attributes
    Grids.append(Grid1)

    # second forward_models variable: E
    Grid2 = PGDMesh('PGD2')
    Grid2.info = [1, 'E', 'N/mm2']
    Grid2.numNodes = 61
    Grid2.numElements = 60

    # create Coordinates
    Grid2.dataX = np.linspace(0.5, 1., num=Grid2.numNodes)
    Grid2.dataY = np.zeros(Grid2.numNodes)  # default
    Grid2.dataZ = np.zeros(Grid2.numNodes)

    # create 1D Mesh
    Grid2.typElements = 'Polyline'
    temp = list()
    for i in range(Grid2.numElements):
        temp.append([i, i + 1])
    Grid2.topology = temp

    # load Attributes
    Attributes = list()
    # disp
    newAttr = PGDAttribute()
    newAttr.name = 'U_x'
    newAttr._type = 'Node'
    newAttr.field = 'Scalar'
    newAttr.data = list()
    newAttr.data.append(np.zeros((Grid2.numNodes, 1)))  # first mode
    for i in range(Grid2.numNodes):
        newAttr.data[0][i, 0] = analytic_mode_UE(Grid2.dataX[i],param)
    Attributes.append(newAttr)
    # sigma
    newAttr = PGDAttribute()
    newAttr.name = 'Sig_x'
    newAttr._type = 'Node'
    newAttr.field = 'Scalar'
    newAttr.data = list()
    newAttr.data.append(np.zeros((Grid2.numNodes, 1)))  # first mode
    for i in range(Grid2.numNodes):
        newAttr.data[0][i, 0] = analytic_mode_SE(Grid2.dataX[i],param)
    Attributes.append(newAttr)

    Grid2.attributes = Attributes
    Grids.append(Grid2)

    # third forward_models variable: load factor
    Grid3 = PGDMesh('PGD3')
    Grid3.info = [1, 'L', '-']
    Grid3.numNodes = 11
    Grid3.numElements = 10

    # create Coordinates
    Grid3.dataX = np.linspace(0, 1, num=Grid3.numNodes)
    Grid3.dataY = np.zeros(Grid3.numNodes)  # default
    Grid3.dataZ = np.zeros(Grid3.numNodes)

    # create 1D Mesh
    Grid3.typElements = 'Polyline'
    temp = list()
    for i in range(Grid3.numElements):
        temp.append([i, i + 1])
    Grid3.topology = temp

    # load Attributes
    Attributes = list()
    # disp
    newAttr = PGDAttribute()
    newAttr.name = 'U_x'
    newAttr._type = 'Node'
    newAttr.field = 'Scalar'
    newAttr.data = list()
    newAttr.data.append(np.zeros((Grid3.numNodes, 1)))  # first mode
    for i in range(Grid3.numNodes):
        newAttr.data[0][i, 0] = analytic_mode_UL(Grid3.dataX[i], param)
    Attributes.append(newAttr)
    # sigma
    newAttr = PGDAttribute()
    newAttr.name = 'Sig_x'
    newAttr._type = 'Node'
    newAttr.field = 'Scalar'
    newAttr.data = list()
    newAttr.data.append(np.zeros((Grid3.numNodes, 1)))  # first mode
    for i in range(Grid3.numNodes):
        newAttr.data[0][i, 0] = analytic_mode_UL(Grid3.dataX[i], param)
    Attributes.append(newAttr)

    Grid3.attributes = Attributes
    Grids.append(Grid3)

    # merge to PGDsolution
    pgdtest.mesh = Grids

    return pgdtest


class TestPGD(unittest.TestCase):
    def setUp(self):
        # general parameters
        self.param = {'A':1, 'n':1, 'lae':1}

        self.pgd = create_example_pgd_solution(self.param)
        # self.pgd.print_info()

        # set PGD extra coordinates to fixed values for tests
        self.E = 0.5
        self.L = 0.4

        # create analytic solution for tests
        # analytic solution
        self.u_ana = u_analytic(x=self.pgd.mesh[0].dataX, p=self.param)([self.E,self.L])
        self.sig_ana = sig_analytic(x=self.pgd.mesh[0].dataX, p=self.param)([self.E, self.L])
        # print('u_ana for E=%s and L=%s: %s' % (self.E, self.L, self.u_ana))
        # print('sig_ana for E=%s and L=%s: %s' % (self.E, self.L, self.sig_ana))

    def tearDown(self):
        pass

    def test_evaluate(self):
        # compute PGD solution at given values and check with analytic solution
        # pgd evaluation
        # 1. displacements 2. stresses
        attri = [0, 1]
        for at in attri:

            # define type of interpolation in pgd solution class for the attributes
            self.pgd.mesh[1].attributes[at].interpolationInfo = {'name': 0, 'kind': 'linear'}
            self.pgd.mesh[2].attributes[at].interpolationInfo = {'name': 0, 'kind': 'linear'}
            # self.pgd.mesh[1].attributes[at].print_info()

            self.pgd.create_interpolation_fcts([1, 2], at) # not necessary but to test fct too

            pgd_eval = self.pgd.evaluate(0, [1, 2], [self.E, self.L], at)
            if at == 0:
                # print('Disp for E=%s and l=%s: %s' %(self.E, self.L, pgd_eval.flatten()))
                np.testing.assert_almost_equal(pgd_eval.flatten(), self.u_ana, 5)

            elif at == 1:
                # print('Stress for E=%s and l=%s: %s' % (self.E, self.L, pgd_eval.flatten()))
                np.testing.assert_almost_equal(pgd_eval.flatten(), self.sig_ana, 5)

    def test_evaluate_min(self):
        # check evaluate min max funtion of pgd solution compared with analytic solution
        attri = 0 # displacements
        self.pgd.mesh[1].attributes[0].interpolationInfo = {'name': 0, 'kind': 'linear'}
        self.pgd.mesh[2].attributes[0].interpolationInfo = {'name': 0, 'kind': 'linear'}

        pgd_min = self.pgd.evaluate_min(0, [1, 2], [self.E, self.L], attri)
        pgd_max = self.pgd.evaluate_max(0, [1, 2], [self.E, self.L], attri)

        # check
        # print('min disp pgd %s = %s' %(pgd_min, self.u_ana.min()))
        # print('max disp pgd %s = %s' %(pgd_max, self.u_ana.max()))
        self.assertAlmostEqual(pgd_min, self.u_ana.min(), places=7)
        self.assertAlmostEqual(pgd_max, self.u_ana.max(), places=7)

    def test_check_error(self):
        # check if evaluate fct find out that given value is not in the range of pgd solution
        attri = 0  # displacements
        self.pgd.mesh[1].attributes[0].interpolationInfo = {'name': 0, 'kind': 'linear'}
        self.pgd.mesh[2].attributes[0].interpolationInfo = {'name': 0, 'kind': 'linear'}

        with self.assertRaises(ValueError):
            self.pgd.evaluate_min(0, [1, 2], [0.2, self.L], attri)


if __name__ == "__main__":
    unittest.main()
