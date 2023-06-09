import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

PI = np.pi


class LambWaveSymmetric:
    d = 1  # half thickness of the plate
    C = 1  # arbitrary constant
    xi = 1  # wave number (changing)
    omega = 1  # angular frequency (changing)
    v = 0.33  # Poisson's ratio https://zh.wikipedia.org/zh-cn/%E6%B3%8A%E6%9D%BE%E6%AF%94
    c_P = 1  # speed of P wave

    k = (2*(1-v)/(1-2*v))**(1/2)  # (eq.26, P. 312)
    c_S = c_P/k  # speed of SV wave (eq.26, P. 312)

    @property
    def eta_P(self):
        """(eq.83, P.325)"""
        return np.sqrt(self.omega**2/self.c_P**2-self.xi**2+0j)

    @property
    def eta_S(self):
        """(eq.83, P.325)"""
        return np.sqrt(self.omega**2/self.c_S**2-self.xi**2+0j)

    @property
    def f(self):
        """frequency (before eq.103, P.328)"""
        return self.omega/(2*PI)

    @property
    def xi_(self):
        """(before eq.103, P.328)"""
        return self.xi*self.d

    @property
    def D(self):
        """(eq.101, P.328)"""
        return (self.xi**2-self.eta_S**2)**2*np.cos(self.eta_P*self.d)*np.sin(self.eta_S*self.d)+4*self.xi**2*self.eta_P*self.eta_S*np.sin(self.eta_P*self.d)*np.cos(self.eta_S*self.d)

    @property
    def Omega(self):
        """(before eq.103, P.328)"""
        return self.omega*self.d/self.c_S

    @property
    def eta_P_(self):
        """(eq.105, P.328)"""
        return np.sqrt(self.Omega**2/self.k**2-self.xi_**2+0j)

    @property
    def eta_S_(self):
        """(eq.105, P.328)"""
        return np.sqrt(self.Omega**2-self.xi_**2+0j)

    @property
    def D_(self):
        """(eq.108, P.328)"""
        return (self.xi_**2-self.eta_S_**2)**2*np.cos(self.eta_P_)*np.sin(self.eta_S_)+4*self.xi_**2*self.eta_P_*self.eta_S_*np.sin(self.eta_P_)*np.cos(self.eta_S_)

    def get_scatter(self):
        """
        perform non-linear root search to find a set of 
        (xi, omega) that satisfy D(xi, omega) = 0
        """
        d_xi = .001
        d_omega = .001
        self.xi = np.arange(0, 5, d_xi)
        self.omega = np.arange(0, 5, d_omega)
        self.xi, self.omega = np.meshgrid(self.xi, self.omega)
        indices = np.where(np.abs(self.D_ - 0) < 0.01)
        # indices = (indices[1]*d_xi*self.d, indices[0]*d_omega*self.d/self.c_S)
        xi = indices[1]
        omega = indices[0]
        xi_ = xi*d_xi*self.d
        Omega = omega*d_omega*self.d/self.c_S
        return xi_, Omega

    def c_div_cS(self, Omega, xi_):
        """use the root (Omega, xi_) found to calculate c/c_S"""
        # (eq.103, P.328)
        xi = xi_/self.d
        omega = Omega*self.c_S/self.d
        # (before eq.112, P.330)
        c = omega/xi
        return c/self.c_S

    def fd(self, Omega):
        """use the root (Omega) found to calculate fd"""
        # (before eq.112, P.330)
        omega = Omega*self.c_S/self.d
        fd = omega/(2*PI)*self.d
        return fd

    def get_xi_and_omega_by_fd(self, fd):
        """use fd to calculate omega and xi

        Args:
            fd (float): frequency in Hz, get by dispersion curve
        """
        xi_, Omega = self.get_scatter()
        c_div_cS = self.c_div_cS(Omega, xi_)
        _fd = self.fd(Omega)
        # find the closest fd in the scatter set
        index = np.argmin(np.abs(_fd-fd))
        fd = _fd[index] # scalar
        c_div_cS = c_div_cS[index] # scalar
        omega = fd*2*PI/self.d
        xi = omega/c_div_cS*self.c_S  # (before eq.112, P.330)
        return xi, omega


class LambWaveAntiSymmetric:
    """Different in equation D and D_"""
    d = 1  # half thickness of the plate
    C = 1  # arbitrary constant
    xi = 1  # wave number (changing)
    omega = 1  # angular frequency (changing)
    v = 0.33  # Poisson's ratio https://zh.wikipedia.org/zh-cn/%E6%B3%8A%E6%9D%BE%E6%AF%94
    c_P = 1  # speed of P wave

    k = (2*(1-v)/(1-2*v))**(1/2)  # (eq.26, P. 312)
    c_S = c_P/k  # speed of SV wave (eq.26, P. 312)

    @property
    def eta_P(self):
        """(eq.83, P.325)"""
        return np.sqrt(self.omega**2/self.c_P**2-self.xi**2+0j)

    @property
    def eta_S(self):
        """(eq.83, P.325)"""
        return np.sqrt(self.omega**2/self.c_S**2-self.xi**2+0j)

    @property
    def f(self):
        """frequency (before eq.103, P.328)"""
        return self.omega/(2*PI)

    @property
    def xi_(self):
        """(before eq.103, P.328)"""
        return self.xi*self.d

    @property
    def D(self):
        """(eq.115, P.331)"""
        return (self.xi**2-self.eta_S**2)**2*np.sin(self.eta_P*self.d)*np.cos(self.eta_S*self.d)+4*self.xi**2*self.eta_P*self.eta_S*np.cos(self.eta_P*self.d)*np.sin(self.eta_S*self.d)

    @property
    def Omega(self):
        """(before eq.103, P.328)"""
        return self.omega*self.d/self.c_S

    @property
    def eta_P_(self):
        """(eq.105, P.328)"""
        return np.sqrt(self.Omega**2/self.k**2-self.xi_**2+0j)

    @property
    def eta_S_(self):
        """(eq.105, P.328)"""
        return np.sqrt(self.Omega**2-self.xi_**2+0j)

    @property
    def D_(self):
        """(eq.117, P.331)"""
        return (self.xi_**2-self.eta_S_**2)**2*np.sin(self.eta_P_)*np.cos(self.eta_S_)+4*self.xi_**2*self.eta_P_*self.eta_S_*np.cos(self.eta_P_)*np.sin(self.eta_S_)

    def get_scatter(self):
        """
        for loop xi_(xi) value and Omega(omega) value to find the combination that makes D_ = 0
        """
        d_xi = .001
        d_omega = .001
        self.xi = np.arange(0, 5, d_xi)
        self.omega = np.arange(0, 5, d_omega)
        self.xi, self.omega = np.meshgrid(self.xi, self.omega)
        indices = np.where(np.abs(self.D_ - 0) < 0.01)
        xi = indices[1]
        omega = indices[0]
        xi_ = xi*d_xi*self.d
        Omega = omega*d_omega*self.d/self.c_S
        return xi_, Omega

    def c_div_cS(self, Omega, xi_):
        # (eq.103, P.328)
        xi = xi_/self.d
        omega = Omega*self.c_S/self.d
        # (before eq.112, P.330)
        c = omega/xi
        return c/self.c_S

    def fd(self, Omega):
        # (before eq.112, P.330)
        omega = Omega*self.c_S/self.d
        fd = omega/(2*PI)*self.d
        return fd

    def get_xi_and_omega_by_fd(self, fd):
        """use fd to calculate omega and xi

        Args:
            fd (float): frequency in Hz, get by dispersion curve
        """
        xi_, Omega = self.get_scatter()
        c_div_cS = self.c_div_cS(Omega, xi_)
        _fd = self.fd(Omega)
        # find the closest fd in the scatter set
        index = np.argmin(np.abs(_fd-fd))
        fd = _fd[index]  # scalar
        c_div_cS = c_div_cS[index]  # scalar
        omega = fd*2*PI/self.d
        xi = omega/c_div_cS*self.c_S  # (before eq.112, P.330)
        return xi, omega


def nonlinear_root_search():
    lw = LambWaveSymmetric()
    # lw = LambWaveAntiSymmetric()
    xi_, Omega = lw.get_scatter()
    fig, ax = plt.subplots()
    ax.scatter(xi_, Omega, s=1)
    # ax.invert_xaxis()
    ax.set_xlabel(r'$\operatorname{Im}\bar{\xi}$')
    ax.set_ylabel(r'$\Omega$')
    plt.show()


def dispersion_curve():
    # lw = LambWaveSymmetric()
    lw = LambWaveAntiSymmetric()
    xi_, Omega = lw.get_scatter()
    c_cS = lw.c_div_cS(Omega, xi_)
    fd = lw.fd(Omega)
    fig, ax = plt.subplots()
    ax.scatter(fd, c_cS, s=1)
    ax.set_ylim(0, 3)
    ax.set_xlabel(r'$fd$')
    ax.set_ylabel(r'$c/c_S$')
    plt.show()


def get_sample():
    lw = LambWaveSymmetric()
    # lw = LambWaveAntiSymmetric()
    xi, omega = lw.get_xi_and_omega_by_fd(0.2)
    print(f"xi={xi}",f"omega={omega}")

if __name__ == '__main__':
    # nonlinear_root_search()
    dispersion_curve()
    # get_sample()
