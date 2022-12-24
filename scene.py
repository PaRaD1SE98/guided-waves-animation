# doc: https://docs.manim.community/en/stable/index.html
# run: manim -pqh scene.py <ClassName>
import math

import sympy
from manim import *


class WavePropInBar(Scene):
    c = 1

    def func(self, x, t):
        X = x - self.c*t
        # bump function
        if X <= -1:
            return 0
        elif -1 < X < 1:
            return math.exp(-1/(1-X**2))
        else:
            return 0

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 10, 1],
            y_range=[0, .4, .1],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-.01, 10.01, 2),
                # 大尺度标记
                "numbers_with_elongated_ticks": np.arange(-.01, 10.01, 2),
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, .41, .2),
                "numbers_with_elongated_ticks": np.arange(0, .41, .2),
            },
            tips=False,  # 坐标箭头
        )
        axes_labels = axes.get_axis_labels()

        # 设置一个变量t，用于控制动画
        t = ValueTracker(0)

        sin_graph = axes.plot(lambda x: self.func(
            x, t.get_value()), color=BLUE)
        sin_graph.add_updater(lambda mobj: mobj.become(
            axes.plot(lambda x: self.func(x, t.get_value()), color=BLUE)))
        # equations = MathTex(
        #     r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}").next_to(label, UP)
        equations = MathTex(r"y = f(x - ct)")
        c_label = MathTex(f"c = {self.c}").next_to(equations, RIGHT)
        t_label = MathTex("t = ").next_to(c_label, RIGHT)
        t_number = DecimalNumber(t.get_value(), num_decimal_places=2)
        t_number.add_updater(lambda mobj: mobj.set_value(
            t.get_value()).next_to(t_label, RIGHT))
        text_group = VGroup(equations, c_label, t_label, t_number).next_to(
            axes, UP)
        plot = VGroup(axes, sin_graph, t_number,
                      axes_labels, t_label, c_label, equations)
        self.add(plot, text_group)
        self.play(t.animate.set_value(10), rate_func=linear)


class EulerFormula(Scene):
    def construct(self):
        # axes
        x_axis_pos_y = 2.5
        x_axis_range = (-5, 5)
        x_start = np.array([x_axis_range[0], x_axis_pos_y, 0])
        x_end = np.array([x_axis_range[1], x_axis_pos_y, 0])
        im_start = np.array([x_axis_range[0], x_axis_pos_y-1, 0])
        im_end = np.array([x_axis_range[0], x_axis_pos_y+1, 0])

        y_axis_pos_x = -3.5
        y_axis_range = (4, -3)
        y_start = np.array([y_axis_pos_x, y_axis_range[0], 0])
        y_end = np.array([y_axis_pos_x, y_axis_range[1], 0])
        re_start = np.array([y_axis_pos_x-1, y_axis_range[0], 0])
        re_end = np.array([y_axis_pos_x+1, y_axis_range[0], 0])

        self.x_rate, self.y_rate = .5, .3

        self.x_axis = Arrow(x_start, x_end, stroke_width=2,
                            max_tip_length_to_length_ratio=.02, color=GREEN)
        self.y_axis = Arrow(y_start, y_end, stroke_width=2,
                            max_tip_length_to_length_ratio=.025, color=GREEN)

        self.re_axis = Arrow(re_start, re_end, stroke_width=2,
                             max_tip_length_to_length_ratio=.1, color=GREEN)
        self.re_axis.shift(.26*DOWN)
        self.im_axis = Arrow(im_start, im_end, stroke_width=2,
                             max_tip_length_to_length_ratio=.1, color=GREEN)
        self.im_axis.shift(.26*RIGHT)

        self.re_label = MathTex("Re").next_to(
            self.re_axis, .2*RIGHT).rotate(-PI/2).scale(.8)
        self.im_label = MathTex("Im").next_to(self.im_axis, .2*UP).scale(.8)

        axis_label_x = MathTex(
            r"\theta").next_to(self.x_axis, RIGHT).scale(.8)
        axis_label_y = MathTex(
            r"\theta").next_to(self.y_axis, DOWN).scale(.8)

        # circle
        self.circle_radius = 1
        self.origin = np.array([y_axis_pos_x, x_axis_pos_y, 0])
        self.circle = Circle(radius=self.circle_radius, color=BLUE)
        self.circle.move_to(self.origin)

        # orbit dot
        self.orbit_dot = Dot(radius=.05, color=YELLOW)
        self.orbit_dot.move_to(self.circle.point_from_proportion(0))

        # main arrow
        self.arrow = Arrow(self.origin,
                           self.orbit_dot.get_center(),
                           color=YELLOW_A,
                           max_tip_length_to_length_ratio=0.15,
                           max_stroke_width_to_length_ratio=2,
                           buff=0)

        self.t = 0

        def go_around_circle(mobj, dt):
            self.t += dt
            mobj.move_to(self.circle.point_from_proportion((self.t / TAU) % 1))
            return mobj

        self.orbit_dot.add_updater(go_around_circle)

        def update_arrow(mobj):
            mobj.put_start_and_end_on(self.origin, self.orbit_dot.get_center())
            return mobj

        self.arrow.add_updater(update_arrow)

        # x curve dot
        self.x_curve_start = np.array(
            [self.origin[0], self.orbit_dot.get_center()[1], 0])

        def get_x_curve_dot():
            x = self.x_curve_start[0] + self.t * self.x_rate
            y = self.orbit_dot.get_center()[1]
            return Dot(np.array([x, y, 0]), color=RED, radius=.05)

        # x curve
        self.x_curve = VGroup()

        def get_x_curve():
            if len(self.x_curve) == 0:
                self.x_curve.add(Line(self.x_curve_start,
                                      self.x_curve_start, color=RED))
            last_line = self.x_curve[-1]
            x = self.x_curve_start[0] + self.t * self.x_rate
            y = self.orbit_dot.get_center()[1]
            new_line = Line(last_line.get_end(), np.array(
                [x, y, 0]), color=RED)
            self.x_curve.add(new_line)
            return self.x_curve

        self.x_curve_dot = always_redraw(get_x_curve_dot)
        self.x_curve_line = always_redraw(get_x_curve)

        # y curve dot
        self.y_curve_start = np.array(
            [self.orbit_dot.get_center()[0], self.origin[1], 0])

        def get_y_curve_dot():
            x = self.orbit_dot.get_center()[0]
            y = self.y_curve_start[1] - self.t * self.y_rate
            return Dot(np.array([x, y, 0]), color=YELLOW, radius=.05)

        # y curve
        self.y_curve = VGroup()

        def get_y_curve():
            if len(self.y_curve) == 0:
                self.y_curve.add(Line(self.y_curve_start,
                                      self.y_curve_start, color=YELLOW))
            last_line = self.y_curve[-1]
            x = self.orbit_dot.get_center()[0]
            y = self.y_curve_start[1] - self.t * self.y_rate
            new_line = Line(last_line.get_end(), np.array(
                [x, y, 0]), color=YELLOW)
            self.y_curve.add(new_line)
            return self.y_curve

        self.y_curve_dot = always_redraw(get_y_curve_dot)
        self.y_curve_line = always_redraw(get_y_curve)

        # orbit_dot to x axis line
        def get_orbit_dot_to_x_axis_line():
            pos_line_start = self.orbit_dot.get_center()
            pos_line_end = np.array(
                [self.orbit_dot.get_center()[0], x_axis_pos_y, 0])
            return Line(pos_line_start, pos_line_end, color=PURPLE, stroke_width=2)

        self.orbit_dot_to_x_axis_line = always_redraw(
            get_orbit_dot_to_x_axis_line)

        # orbit_dot to y axis line
        def get_orbit_dot_to_y_axis_line():
            pos_line_start = self.orbit_dot.get_center()
            pos_line_end = np.array(
                [y_axis_pos_x, self.orbit_dot.get_center()[1], 0])
            return Line(pos_line_start, pos_line_end, color=MAROON, stroke_width=2)

        self.orbit_dot_to_y_axis_line = always_redraw(
            get_orbit_dot_to_y_axis_line)

        # origin to x axis arrow
        def get_origin_to_x_axis_arrow():
            pos_end = np.array(
                [self.orbit_dot.get_center()[0], x_axis_pos_y, 0])
            return Arrow(self.origin, pos_end, color=WHITE, max_tip_length_to_length_ratio=0.15,
                         max_stroke_width_to_length_ratio=2, buff=0)

        self.origin_to_x_axis_arrow = always_redraw(get_origin_to_x_axis_arrow)

        # origin to y axis arrow
        def get_origin_to_y_axis_arrow():
            pos_end = np.array(
                [y_axis_pos_x, self.orbit_dot.get_center()[1], 0])
            return Arrow(self.origin, pos_end, color=WHITE, max_tip_length_to_length_ratio=0.15,
                         max_stroke_width_to_length_ratio=2, buff=0)

        self.origin_to_y_axis_arrow = always_redraw(get_origin_to_y_axis_arrow)

        # orbit_dot to y curve line
        def get_orbit_dot_to_y_curve_line():
            pos_line_start = self.orbit_dot.get_center()
            pos_line_end = np.array(
                [self.orbit_dot.get_center()[0], self.y_curve_dot.get_center()[1], 0])
            return Line(pos_line_start, pos_line_end, color=PURPLE, stroke_width=2)

        self.orbit_dot_to_y_curve_line = always_redraw(
            get_orbit_dot_to_y_curve_line)

        # orbit_dot to y axis line
        def get_orbit_dot_to_x_curve_line():
            pos_line_start = self.orbit_dot.get_center()
            pos_line_end = np.array(
                [self.x_curve_dot.get_center()[0], self.orbit_dot.get_center()[1], 0])
            return Line(pos_line_start, pos_line_end, color=MAROON, stroke_width=2)

        self.orbit_dot_to_x_curve_line = always_redraw(
            get_orbit_dot_to_x_curve_line)

        self.equation = MathTex(
            r"e^{i \theta} = \cos(\theta) + i \sin(\theta)").move_to([0, 0, 0]).shift(RIGHT)

        # ticks
        x_labels = [
            MathTex("\pi"), MathTex("2 \pi"),
            MathTex("3 \pi"), MathTex("4 \pi"), MathTex("5 \pi")
        ]
        y_labels = [
            MathTex("\pi"), MathTex("2 \pi"),
            MathTex("3 \pi"), MathTex("4 \pi"), MathTex("5 \pi")
        ]

        x_label_group = VGroup()
        for i, label in enumerate(x_labels):
            label.move_to(self.origin).shift(.25*DOWN)
            label.shift(self.x_rate*PI * (i+1) * RIGHT)
            x_label_group.add(label)
        y_label_group = VGroup()
        for i, label in enumerate(y_labels):
            label.move_to(self.origin).shift(.25*LEFT).rotate(-PI/2)
            label.shift(self.y_rate*PI * (i+1) * DOWN)
            y_label_group.add(label)

        # theta value
        self.theta_label = MathTex(r"\theta = ").next_to(
            self.equation, DOWN, aligned_edge=LEFT)

        def get_theta():
            return DecimalNumber(self.t, num_decimal_places=2).next_to(self.theta_label, RIGHT)

        self.theta = always_redraw(get_theta)

        axis_group = VGroup(self.x_axis, self.y_axis,
                            self.im_axis, self.re_axis)
        axis_label_group = VGroup(
            axis_label_x, axis_label_y, self.re_label, self.im_label)
        axis_value_group = VGroup(x_label_group, y_label_group)

        group1 = VGroup(axis_group,
                        axis_label_group,
                        axis_value_group,
                        self.circle)

        group2 = VGroup(self.orbit_dot,
                        self.arrow,
                        self.x_curve_dot,
                        self.x_curve_line,
                        self.y_curve_dot,
                        self.y_curve_line)
        group3 = VGroup(self.orbit_dot_to_x_axis_line,
                        self.orbit_dot_to_y_axis_line,
                        self.origin_to_x_axis_arrow,
                        self.origin_to_y_axis_arrow,
                        self.orbit_dot_to_y_curve_line,
                        self.orbit_dot_to_x_curve_line,
                        self.equation,
                        self.theta_label,
                        self.theta)
        group = VGroup(group1, group2, group3)
        self.add(group)
        self.wait(2*PI * 2.5)


class StandingWave(Scene):
    gamma_f = 1  # wave number forward
    gamma_b = 1  # wave number backward
    omega_f = 1  # angular frequency forward
    omega_b = 1  # angular frequency backward
    gamma_verify = 1
    omega_verify = 1

    def forward_wave(self, x, t):
        alpha = .5
        return alpha * np.sin(self.gamma_f*x - self.omega_f*t)

    def backward_wave(self, x, t):
        alpha = .5
        return alpha * np.sin(self.gamma_b*x + self.omega_b*t)

    def standing_wave(self, x, t):
        return self.forward_wave(x, t) + self.backward_wave(x, t)

    def standing_wave_verify(self, x, t):
        return np.cos(self.gamma_verify*x)*np.exp(-1j*self.omega_verify*t)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1, 1, .5],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-2*PI, 2.01*PI, PI),
                "decimal_number_config": {"num_decimal_places": 2},
                # 大尺度标记
                # "numbers_with_elongated_ticks": np.arange(-2*PI, 2*PI, PI),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-1, 1.01, .5),
                # "numbers_with_elongated_ticks": np.arange(-1, 1, .5),
            },
            tips=False,  # 坐标箭头
        ).shift(1.5*UP)
        axes_labels = axes.get_axis_labels()

        equation_forward = MathTex(r"\sin(\gamma_f x - \omega_f t)", color=BLUE).next_to(
            axes, UP).shift(3*RIGHT)
        equation_backward = MathTex(r"\sin(\gamma_b x + \omega_b t)", color=YELLOW).next_to(
            axes, UP).shift(3*LEFT)

        self.add(axes, axes_labels, equation_backward, equation_forward)

        t = ValueTracker(0)

        def update_t(mob, dt):
            t.increment_value(dt)
        t.add_updater(update_t)

        forward_wave = axes.plot(lambda x: self.forward_wave(
            x, t.get_value()), color=BLUE)

        def update_forward_wave(mobj):
            mobj.become(axes.plot(lambda x: self.forward_wave(
                x, t.get_value()), color=BLUE))
        forward_wave.add_updater(update_forward_wave)

        backward_wave = axes.plot(lambda x: self.backward_wave(
            x, t.get_value()), color=YELLOW)

        def update_backward_wave(mobj):
            mobj.become(axes.plot(lambda x: self.backward_wave(
                x, t.get_value()), color=YELLOW))
        backward_wave.add_updater(update_backward_wave)

        axes2 = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1, 1, .5],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-2*PI, 2.01*PI, PI),
                "decimal_number_config": {"num_decimal_places": 2},
                # 大尺度标记
                # "numbers_with_elongated_ticks": np.arange(-2*PI, 2*PI, PI),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-1, 1.01, .5),
                # "numbers_with_elongated_ticks": np.arange(-1, 1, .5),
            },
            tips=False,  # 坐标箭头
        ).shift(1.5*DOWN)
        axes_labels2 = axes.get_axis_labels()
        self.add(axes2, axes_labels2)

        standing_wave = axes2.plot(lambda x: self.standing_wave_verify(
            x, t.get_value()).imag, color=PURPLE)

        def update_standing_wave(mobj):
            mobj.become(axes2.plot(lambda x: self.standing_wave_verify(
                x, t.get_value()).imag, color=PURPLE))
        standing_wave.add_updater(update_standing_wave)

        gamma_label_b = MathTex(f"\gamma_b = {self.gamma_b}", color=YELLOW).next_to(
            axes, DOWN).shift(3.5*LEFT)
        omega_label_b = MathTex(f"\omega_b = {self.omega_b}", color=YELLOW).next_to(
            gamma_label_b, RIGHT)
        gamma_label_f = MathTex(f"\gamma_f = {self.gamma_f}", color=BLUE).next_to(
            omega_label_b, RIGHT)
        omega_label_f = MathTex(f"\omega_f = {self.omega_f}", color=BLUE).next_to(
            gamma_label_f, RIGHT)

        t_label = MathTex("t = ").next_to(omega_label_f, RIGHT)
        t_number = DecimalNumber(t.get_value(), num_decimal_places=2).next_to(
            t_label, RIGHT)
        t_number.add_updater(lambda m: m.set_value(t.get_value()))

        self.add(t,
                 gamma_label_f,
                 gamma_label_b,
                 omega_label_f,
                 omega_label_b,
                 t_label,
                 t_number)
        self.add(forward_wave,
                 backward_wave,
                 standing_wave)
        self.wait(2*PI)


class StandingWaveVerify(Scene):
    gamma = 1
    omega = 1

    def standing_wave(self, x, t):
        return np.cos(self.gamma*x)*np.exp(-1j*self.omega*t)

    def construct(self):
        equation = MathTex(r"u(x, t) = \cos(\gamma x) e^{-i\omega t}")
        text_gamma = MathTex(f"\gamma = {self.gamma}").next_to(equation, RIGHT)
        text_omega = MathTex(f"\omega = {self.omega}").next_to(
            text_gamma, RIGHT)
        t = ValueTracker(0)
        t_label = MathTex("t = ").next_to(text_omega, RIGHT)
        t_number = DecimalNumber(t.get_value(), num_decimal_places=2).next_to(
            t_label, RIGHT)
        t_number.add_updater(lambda m: m.set_value(t.get_value()))
        eqgroup = VGroup(equation,
                         text_gamma,
                         text_omega,
                         t_label,
                         t_number).move_to(ORIGIN).to_edge(UP)

        axes_re = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1, 1, .5],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-2*PI, 2.01*PI, PI),
                "decimal_number_config": {"num_decimal_places": 2},
                # 大尺度标记
                # "numbers_with_elongated_ticks": np.arange(-2*PI, 2*PI, PI),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-1, 1.01, .5),
                # "numbers_with_elongated_ticks": np.arange(-1, 1, .5),
            },
            tips=False,  # 坐标箭头
        ).next_to(eqgroup, 2*DOWN)
        axes_im = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1, 1, .5],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-2*PI, 2.01*PI, PI),
                "decimal_number_config": {"num_decimal_places": 2},
                # 大尺度标记
                # "numbers_with_elongated_ticks": np.arange(-2*PI, 2*PI, PI),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-1, 1.01, .5),
                # "numbers_with_elongated_ticks": np.arange(-1, 1, .5),
            },
            tips=False,  # 坐标箭头
        ).next_to(axes_re, 2*DOWN)
        axes_labels_re = axes_re.get_axis_labels(y_label="Re")
        axes_labels_im = axes_im.get_axis_labels(y_label="Im")
        self.add(axes_re, axes_labels_re, axes_im, axes_labels_im)
        t.add_updater(lambda m, dt: m.increment_value(dt))
        standing_wave_re = axes_re.plot(lambda x: self.standing_wave(
            x, t.get_value()).real, color=BLUE)

        def update_standing_wave_re(mobj):
            mobj.become(axes_re.plot(lambda x: self.standing_wave(
                x, t.get_value()).real, color=BLUE))
        standing_wave_re.add_updater(update_standing_wave_re)

        standing_wave_im = axes_im.plot(lambda x: self.standing_wave(
            x, t.get_value()).imag, color=PURPLE)

        def update_standing_wave_im(mobj):
            mobj.become(axes_im.plot(lambda x: self.standing_wave(
                x, t.get_value()).imag, color=PURPLE))
        standing_wave_im.add_updater(update_standing_wave_im)

        self.add(eqgroup)
        self.add(t)
        self.add(standing_wave_re, standing_wave_im)

        self.wait(2*PI)


class PowerAndEnergy(Scene):
    rho = 1  # density (assumed constant)
    E = 1  # elastic modulus (assumed constant)
    A = 1  # cross section area (assumed constant)
    omega = 1  # angular frequency (assumed constant) (eq.50, P.225)
    m = rho*A  # mass of the infinitesimal element dx (above eq.10, P.218)
    c = (E/rho)**(1/2)  # wave speed (eq.10, P.218)
    gamma = omega / c  # wave number (eq.53, P.225)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.construct_derivatives()

    def theta(self, x, t):
        """The single variable"""
        return x + self.c*t

    def f(self, x, t):
        """Propagating wave function (any)
        
        Note: Need to use sympy functions for derivatives
        """
        return sympy.cos(self.gamma*self.theta(x, t))

    def construct_derivatives(self):
        """Construct derivatives of f(x, t)"""
        X, T = sympy.symbols('x t')
        _df_dx = sympy.diff(self.f(X, T), X)
        self._df_dx_func = sympy.lambdify((X, T), _df_dx)
        _df_dt = sympy.diff(self.f(X, T), T)
        self._df_dt_func = sympy.lambdify((X, T), _df_dt)

    def df_dx(self, x, t):
        """Propagating velocity df/dx"""
        return self._df_dx_func(x, t)

    def df_dt(self, x, t):
        """Propagating velocity df/dt"""
        return self._df_dt_func(x, t)

    def k(self, x, t):
        """Kinetic energy (eq.73, P.229)

        Note:
            k should be = v
        """
        return 1/2*self.m*self.df_dt(x, t)**2

    def v(self, x, t):
        """Elastic energy (eq.74, P.229)

        Note:
            v should be = k
        """
        return 1/2*self.E*self.A*self.df_dx(x, t)**2

    def e(self, x, t):
        """Total energy (eq.80, P.230)"""
        return self.k(x, t) + self.v(x, t)

    def P(self, x, t):
        """Power (eq.81, P.230)"""
        return -self.m*self.c**2*self.df_dx(x, t)*self.df_dt(x, t)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1, 1, .5],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-2*PI, 2.01*PI, PI),
                "decimal_number_config": {"num_decimal_places": 2},
                # 大尺度标记
                # "numbers_with_elongated_ticks": np.arange(-2*PI, 2*PI, PI),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-1, 1.01, .5),
                # "numbers_with_elongated_ticks": np.arange(-1, 1, .5),
            },
            tips=False,  # 坐标箭头
        ).shift(2*UP)
        axes_labels = axes.get_axis_labels(y_label="")
        axes2 = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1, 1, .5],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-2*PI, 2.01*PI, PI),
                "decimal_number_config": {"num_decimal_places": 2},
                # 大尺度标记
                # "numbers_with_elongated_ticks": np.arange(-2*PI, 2*PI, PI),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-1, 1.01, .5),
                # "numbers_with_elongated_ticks": np.arange(-1, 1, .5),
            },
            tips=False,  # 坐标箭头
        ).next_to(axes, DOWN)
        axes_labels2 = axes2.get_axis_labels(y_label="")

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        # Propagating wave
        f_plot = axes.plot(lambda x: self.f(x, 0), color=WHITE)

        def update_f_plot(mobj):
            mobj.become(axes.plot(lambda x: self.f(
                x, t.get_value()), color=WHITE))
        f_plot.add_updater(update_f_plot)
        label_f = MathTex("u", color=WHITE)

        # df/dx
        dfdx_plot = axes.plot(lambda x: self.df_dx(x, 0), color=BLUE)

        def update_dfdx_plot(mobj):
            mobj.become(axes.plot(lambda x: self.df_dx(
                x, t.get_value()), color=BLUE))
        dfdx_plot.add_updater(update_dfdx_plot)
        label_dfdx = MathTex(r"u\prime", color=BLUE)

        # df/dt
        dfdt_plot = axes.plot(lambda x: self.df_dt(x, 0), color=GREEN)

        def update_dfdt_plot(mobj):
            mobj.become(axes.plot(lambda x: self.df_dt(
                x, t.get_value()), color=GREEN))
        dfdt_plot.add_updater(update_dfdt_plot)
        label_dfdt = MathTex(r"\dot{u}", color=GREEN)

        # Kinetic energy
        k_plot = axes.plot(lambda x: self.k(x, 0), color=RED)

        def update_k_plot(mobj):
            mobj.become(axes.plot(lambda x: self.k(
                x, t.get_value()), color=RED))
        k_plot.add_updater(update_k_plot)
        label_k = MathTex("k", color=RED)

        # Elastic energy
        v_plot = axes2.plot(lambda x: self.v(x, 0), color=PURPLE)

        def update_v_plot(mobj):
            mobj.become(axes2.plot(lambda x: self.v(
                x, t.get_value()), color=PURPLE))
        v_plot.add_updater(update_v_plot)
        label_v = MathTex("v", color=PURPLE)

        # Total energy
        e_plot = axes2.plot(lambda x: self.e(x, 0), color=ORANGE)

        def update_e_plot(mobj):
            mobj.become(axes2.plot(lambda x: self.e(
                x, t.get_value()), color=ORANGE))
        e_plot.add_updater(update_e_plot)
        label_e = MathTex("e", color=ORANGE)

        # Power
        P_plot = axes.plot(lambda x: self.P(x, 0), color=YELLOW)

        def update_P_plot(mobj):
            mobj.become(axes.plot(lambda x: self.P(
                x, t.get_value()), color=YELLOW))
        P_plot.add_updater(update_P_plot)
        label_P = MathTex("P", color=YELLOW)

        label_f.to_corner(UL)
        label_dfdx.next_to(label_f, RIGHT)
        label_dfdt.next_to(label_dfdx, RIGHT)
        label_k.next_to(label_dfdt, RIGHT)
        label_v.next_to(label_k, RIGHT)
        label_e.next_to(label_v, RIGHT)
        label_P.next_to(label_e, RIGHT)

        function_labels = VGroup(label_f,
                                 label_dfdx,
                                 label_dfdt,
                                 label_k,
                                 label_v,
                                 label_e,
                                 label_P).next_to(axes2, DOWN)

        self.add(axes, axes_labels)
        self.add(axes2, axes_labels2)

        self.add(t, f_plot, dfdt_plot, dfdx_plot,
                 k_plot, v_plot, e_plot, P_plot)

        self.add(function_labels)
        self.wait(2*PI)
