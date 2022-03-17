# 基于棱元离散的2D圆柱型隐身衣
# author: 王唯

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from icecream import ic

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC

from fealpy.decorator import cartesian

from opt_einsum import contract as einsum


class pml2D:
    def __init__(self,
                 domain,
                 kappa,
                 pml_delta_x=None,
                 pml_delta_y=None,
                 absortion_constant=None):
        self.domain = domain
        self.kappa = kappa
        self.pml_delta_x = 0.5
        self.pml_delta_y = 0.5
        self.absortion_constant = 200

    def pml_sigma_x(self, x):
        domain = self.domain()
        d_x = self.pml_delta_x
        C = self.absortion_constant

        a1 = domain[0] + d_x
        b1 = domain[1] - d_x

        sigma = np.zeros_like(x)

        idx_1 = (x < a1)
        idx_2 = (x > a1) & (x < b1)
        idx_3 = (x > b1)

        sigma[idx_1] = C * ((x[idx_1] - a1) / d_x) ** 2
        sigma[idx_2] = 0.0
        sigma[idx_3] = C * ((x[idx_3] - b1) / d_x) ** 2
        k = self.kappa
        alpha_x = 1.0 + (sigma / (1.j * k))

        return alpha_x

    def pml_sigma_y(self, y):
        domain = self.domain()

        d_y = self.pml_delta_y
        C = self.absortion_constant

        a2 = domain[2] + d_y
        b2 = domain[3] - d_y

        sigma = np.zeros_like(y)

        idx_1 = (y > domain[2]) & (y < a2)
        idx_2 = (y > a2) & (y < b2)
        idx_3 = (y > b2) & (y < domain[3])

        sigma[idx_1] = C * ((y[idx_1] - a2) / d_y) ** 2
        sigma[idx_2] = 0.0
        sigma[idx_3] = C * ((y[idx_3] - b2) / d_y) ** 2
        k = self.kappa
        alpha_y = 1.0 + (sigma / (1.j * k))
        return alpha_y

    # @cartesian
    # def find_pml(self, p):
    #     x = p[..., 0]
    #     y = p[..., 1]
    #     domain = self.domain()
    #     d_x = self.pml_delta_x
    #     d_y = self.pml_delta_y
    #
    #     a = domain[0]
    #     b = domain[1]
    #     c = domain[2]
    #     d = domain[3]
    #
    #     a1 = domain[0] + d_x
    #     b1 = domain[1] - d_x
    #     c1 = domain[2] + d_y
    #     d1 = domain[3] - d_y
    #
    #     x1 = (x > a) & (x < a1)
    #     x2 = (x > a1) & (x < b1)
    #     x3 = (x > b1) & (x < b)
    #
    #     y1 = (y > c) & (y < c1)
    #     y2 = (y > c1) & (y < d1)
    #     y3 = (y > d1) & (y < d)
    #
    #     idx_c = (x1 & y3) | (x3 & y3) | (x1 & y1) | (x3 & y1)
    #     idx_x = (x2 & y3) | (x2 & y1)
    #     idx_y = (x1 & y2) | (x3 & y2)
    #
    #     return idx_c, idx_x, idx_y

    @cartesian
    def pml_alpha(self, p):
        pml_d_x = self.pml_sigma_x(p[..., 0])
        pml_d_y = self.pml_sigma_y(p[..., 1])


        shape = p.shape[0:2]
        val = np.ones(shape=shape, dtype=np.complex_)

        val = 1.0 / (pml_d_x * pml_d_y)

        return val

    @cartesian
    def pml_beta(self, p):
        pml_d_x = self.pml_sigma_x(p[..., 0])
        pml_d_y = self.pml_sigma_y(p[..., 1])

        shape = p.shape + (2,)
        val = np.zeros(shape=shape, dtype=np.complex_)
        val[..., 0, 0] = val[..., 1, 1] = 1.0

        # idx_c, idx_x, idx_y = self.find_pml(p)

        val[..., 0, 0] = pml_d_y / pml_d_x
        val[..., 1, 1] = pml_d_x / pml_d_y

        # val[idx_x, 0, 0] = pml_d_y[idx_x]
        # val[idx_x, 1, 1] = 1.0 / pml_d_y[idx_x]
        #
        # val[idx_y, 0, 0] = 1.0 / pml_d_x[idx_y]
        # val[idx_y, 1, 1] = pml_d_x[idx_y]

        return val

    def get_pml_matrix(self, p):
        return self.pml_beta(p)

    def get_pml_vector(self, p):
        return self.pml_alpha(p)


class model:
    def __init__(self,
                 kappa=None,
                 great_circle=None,
                 small_circle=None,
                 mesh_n=None):
        self.kappa = 20
        self.great_circle = 0.3
        self.small_circle = 0.15
        self.mesh_n = 0.02

    def domain(self):
        val = np.array([-2, 2, -2, 2])
        return val

    def init_mesh(self):
        from fealpy.geometry import dcircle, drectangle, ddiff
        from fealpy.geometry import DistDomain2d
        from fealpy.geometry import huniform
        from fealpy.mesh import DistMesh2d
        from fealpy.mesh import MeshFactory as mf

        box = self.domain()
        R = self.small_circle

        fd = lambda p: ddiff(drectangle(p, box), dcircle(p, [0, 0], R))
        fh = huniform

        h0 = self.mesh_n

        pfix = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, -1.0), (1.0, 1.0)])
        domain = DistDomain2d(fd, fh, box, pfix)
        distmesh2d = DistMesh2d(domain, h0)

        distmesh2d.run()
        mesh = distmesh2d.mesh

        # fig = plt.figure()
        # axes = fig.gca()
        # distmesh2d.mesh.add_plot(axes)
        # plt.show()
        return mesh

    @cartesian
    def mu(self, p):
        x = p[..., 0]
        y = p[..., 1]
        shape = p.shape[0:2]
        val = np.ones(shape=shape, dtype=np.float64)
        R1 = self.small_circle
        R2 = self.great_circle

        r = np.sqrt(x ** 2 + y ** 2)

        idx = (r < R2)

        # q = R2 / (R2 - R1)
        # r = (r - R1) / q

        r = r[idx]


        det = ((R2 - R1) / R2) ** 2
        det *= r / (r - R1)


        val[idx] = det

        return val

    @cartesian
    def epsilon(self, p):
        x = p[..., 0]
        y = p[..., 1]
        shape = p.shape + (2,)
        val = np.ones(shape=shape, dtype=np.float64)
        val[..., 1, 0] = val[..., 0, 1] = 0

        R1 = self.small_circle
        R2 = self.great_circle

        r = np.sqrt(x ** 2 + y ** 2)
        # q = R2 / (R2 - R1)
        # r = (r - R1) / q

        t = np.zeros_like(r)
        k1 = ((y == 0) & (x > 0))
        t[k1] = 0

        k2 = ((y == 0) & (x < 0))
        t[k2] = np.pi

        k3 = ((x == 0) & (y > 0))
        t[k3] = np.pi / 2

        k4 = ((x == 0) & (y < 0))
        t[k4] = 3 * np.pi / 2

        k5 = ((x > 0) & (y > 0))
        t[k5] = np.arctan(y[k5] / x[k5])

        k6 = ((x < 0) & (y > 0))
        t[k6] = np.pi - np.abs(np.arctan(y[k6] / x[k6]))

        k7 = ((x < 0) & (y < 0))
        t[k7] = np.pi + np.abs(np.arctan(y[k7] / x[k7]))

        k8 = ((x > 0) & (y < 0))
        t[k8] = 2 * np.pi - np.abs(np.arctan(y[k8] / x[k8]))

        idx = (r < R2)

        # q = R2 / (R2 - R1)
        # r = (r - R1) / q
        r = r[idx]
        t = t[idx]

        det = ((R2 - R1) / R2) ** 2
        det *= r / (r - R1)

        a = ((R2 - R1) / R2)
        b = R1 / r

        # e1 = (a ** 2) + (b * (2 * a + b) * np.sin(t) ** 2)
        # e2 = -1 * b * (2 * a + b) * np.sin(t) * np.cos(t)
        # e4 = (a ** 2) + (b * (2 * a + b) * np.cos(t) ** 2)
        e1 = (r ** 2 + R1 * (R1 - 2 * r) * (np.cos(t) ** 2)) / (r * (r - R1))
        e2 = R1 * (R1 - 2 * r) * np.sin(t) * np.cos(t) / (r * (r - R1))
        e4 = (r ** 2 + R1 * (R1 - 2 * r) * (np.sin(t) ** 2)) / (r * (r - R1))

        val[idx, 0, 0] = e1
        val[idx, 0, 1] = e2
        val[idx, 1, 0] = e2
        val[idx, 1, 1] = e4
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]

        idx = (x > -1.5) & (x < -1.45) & (np.abs(y) < 1.25)

        k = self.kappa
        val = np.zeros(shape=p.shape, dtype=np.complex_)
        val[idx, 1] = (k ** 2) * np.exp(1.j * k * x[idx])
        return val

    @cartesian
    def dirichlet(self, p, t):
        val = np.zeros(shape=p.shape)
        val = np.sum(val * t, axis=-1)
        return val


class mysolution:
    def __init__(self,
                 pde=None,
                 pml=None,
                 mesh=None,
                 q=None,
                 space=None):
        self.pde = model()
        self.pml = pml2D(domain=self.pde.domain, kappa=self.pde.kappa)
        self.mesh = self.pde.init_mesh()
        self.q = 6
        self.space = FirstKindNedelecFiniteElementSpace2d(mesh=self.mesh, p=0, q=self.q)

    def get_curl_matrix(self):
        qf = self.mesh.integrator(self.q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.mesh.bc_to_point(bcs)
        mu = (self.pml.get_pml_vector(ps)) * self.pde.mu(ps)
        A = self.space.curl_matrix(c=mu)
        return A

    def get_mass_matrix(self):
        qf = self.mesh.integrator(self.q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.mesh.bc_to_point(bcs)
        eps = self.pml.get_pml_matrix(ps) * self.pde.epsilon(ps)
        B = self.space.mass_matrix(c=eps)
        k = self.pde.kappa
        B = (k ** 2) * B
        return B

    def get_right_hand(self):
        qf = self.mesh.integrator(self.q, etype='cell')

        # 重心坐标, 权重
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 得到自然坐标
        ps = self.mesh.bc_to_point(bcs)  # (积分点个数, 单元个数, 坐标轴个数)
        cellmeasure = self.mesh.entity_measure('cell')

        phi = self.space.basis(bcs)
        val = self.pde.source(ps)
        cell2dof = self.space.cell_to_dof()

        # beta = self.pml.get_pml_matrix(ps)
        alpha = self.pml.get_pml_vector(ps)

        bb = einsum('i, ij, ijm, ijkm, j -> jk', ws, alpha, val, phi, cellmeasure)

        gdof = self.space.number_of_global_dofs()
        F = np.zeros(gdof, dtype=np.complex_)
        np.add.at(F, cell2dof, bb)
        return F

    def algebra_system(self):
        A = self.get_curl_matrix() - self.get_mass_matrix()
        F = self.get_right_hand()
        uh = self.space.function(dtype=np.complex_)

        bc = DirichletBC(self.space, self.pde.dirichlet)
        A, F = bc.apply(A, F, uh)
        uh[:] = spsolve(A, F)
        return uh

    def get_imag(self):
        mesh = self.pde.init_mesh()

        # qf = self.mesh.integrator(self.q, etype='cell')
        # bcs, ws = qf.get_quadrature_points_and_weights()
        # ps = self.mesh.bc_to_point(bcs)

        uh = self.algebra_system()
        aa = np.array([1 / 3, 1 / 3, 1 / 3])
        value = uh(aa)

        # mesh.add_plot(plt, cellcolor=value[..., 0].real, linewidths=0, showaxis=True, showcolorbar=True, cmap='jet')
        # plt.title('real_x')
        mesh.add_plot(plt,
                      cellcolor=value[..., 1].real,
                      linewidths=0,
                      showaxis=True,
                      showcolorbar=True,
                      cmap='jet')

        plt.title('real_y')
        # mesh.add_plot(plt, cellcolor=value[..., 0].imag, linewidths=0, showaxis=True, showcolorbar=True, cmap='jet')
        # plt.title('imag_x')
        mesh.add_plot(plt,
                      cellcolor=value[..., 1].imag,
                      linewidths=0,
                      showaxis=True,
                      showcolorbar=True,
                      cmap='jet')
        plt.title('imag_y')
        # pp = mesh.bc_to_point(aa)
        # fig = plt.figure()
        # xx = pp[..., 0]
        # yy = pp[..., 1]
        # plt.quiver(xx, yy, value[..., 0], value[..., 1])
        # plt.title('vector filed')
        plt.show()

    def get_paraview(self):
        mesh = self.mesh
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        cell = cell.astype('int64')
        uh = self.algebra_system()
        aa = np.array([1 / 3, 1 / 3, 1 / 3])
        value = uh(aa)
        pmesh = TriangleMesh(node, cell)
        # pmesh.celldata['real'] = value[..., 1].real
        pmesh.celldata['cloak'] = value[..., 1].imag
        pmesh.to_vtk(fname='ww.vtu')


clock = mysolution()
clock.get_imag()
# clock.get_paraview()
