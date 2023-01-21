# doc: https://docs.manim.community/en/stable/index.html
# run: manim -pqh scene.py <ClassName>
import math

import sympy
from manim import *
from tqdm import tqdm


class WavePropInBar(Scene):
    c = 1

    def theta_forward(self, x, t):
        return x - self.c*t

    def theta_backward(self, x, t):
        return x + self.c*t

    def f_forward(self, x, t):
        # bump function
        X = self.theta_forward(x, t)
        if X <= -1:
            return 0
        elif -1 < X < 1:
            return math.exp(-1/(1-X**2))
        else:
            return 0

    def f_backward(self, x, t):
        # bump function
        X = self.theta_backward(x, t)
        if X <= -1:
            return 0
        elif -1 < X < 1:
            return math.exp(-1/(1-X**2))
        else:
            return 0

    def construct(self):
        axes_forward = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 5, 1],
            y_range=[0, .4, .1],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-.01, 5.01, 2),
                # 大尺度标记
                "numbers_with_elongated_ticks": np.arange(-.01, 5.01, 2),
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, .41, .2),
                "numbers_with_elongated_ticks": np.arange(0, .41, .2),
            },
            tips=False,  # 坐标箭头
        ).to_edge(UP)
        axes_labels_forward = axes_forward.get_axis_labels(y_label="")
        axes_backward = Axes(
            # 坐标轴数值范围和步长
            x_range=[-5, 0, 1],
            y_range=[0, .4, .1],
            # 坐标轴长度（比例）
            x_length=10,
            y_length=2,
            axis_config={"color": GREEN},
            x_axis_config={
                # 尺度标出数值
                "numbers_to_include": np.arange(-5.01, .01, 2),
                # 大尺度标记
                "numbers_with_elongated_ticks": np.arange(-5.01, .01, 2),
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, .41, .2),
                "numbers_with_elongated_ticks": np.arange(0, .41, .2),
            },
            tips=False,  # 坐标箭头
        ).next_to(axes_forward, DOWN)
        axes_labels_backward = axes_backward.get_axis_labels(y_label="")
        # 设置一个变量t，用于控制动画
        t = ValueTracker(0)
        t.add_updater(lambda mobj, dt: mobj.increment_value(dt))

        forward_plot = axes_forward.plot(lambda x: self.f_forward(
            x, t.get_value()), color=BLUE)
        forward_plot.add_updater(lambda mobj: mobj.become(
            axes_forward.plot(lambda x: self.f_forward(x, t.get_value()), color=BLUE)))
        backward_plot = axes_backward.plot(lambda x: self.f_backward(
            x, t.get_value()), color=YELLOW)
        backward_plot.add_updater(lambda mobj: mobj.become(
            axes_backward.plot(lambda x: self.f_backward(x, t.get_value()), color=YELLOW)))
        # equations = MathTex(
        #     r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}").next_to(label, UP)
        equation_f = MathTex(r"y_f = f(x - ct)", color=BLUE)
        equation_b = MathTex(r"y_b = f(x + ct)",
                             color=YELLOW).next_to(equation_f, RIGHT)
        c_label = MathTex(f"c = {self.c}").next_to(equation_b, RIGHT)
        t_label = MathTex("t = ").next_to(c_label, RIGHT)
        t_number = DecimalNumber(t.get_value(), num_decimal_places=2)
        t_number.add_updater(lambda mobj: mobj.set_value(
            t.get_value()).next_to(t_label, RIGHT))
        text_group = VGroup(equation_f, equation_b, c_label, t_label, t_number).next_to(
            axes_backward, DOWN)
        plot = VGroup(axes_forward,
                      axes_labels_forward,
                      axes_backward,
                      axes_labels_backward,
                      forward_plot,
                      backward_plot,
                      t_number,
                      t_label,
                      c_label)
        self.add(t)
        self.add(plot, text_group)
        self.wait(5)


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

    def forward_wave(self, x, t):
        alpha = .5
        return alpha * np.sin(self.gamma_f*x - self.omega_f*t)

    def backward_wave(self, x, t):
        alpha = .5
        return alpha * np.sin(self.gamma_b*x + self.omega_b*t)

    def standing_wave(self, x, t):
        return self.forward_wave(x, t) + self.backward_wave(x, t)

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
        """The single variable

        Toggle the sign of the 't' term to change the direction of the wave
        """
        return x - self.c*t

    def u(self, x, t):
        """Propagating wave function (any)

        Note: Need to use sympy functions for derivatives
        """
        return sympy.cos(self.gamma*self.theta(x, t))

    def construct_derivatives(self):
        """Construct derivatives of f(x, t)"""
        X, T = sympy.symbols('x t')
        _du_dx = sympy.diff(self.u(X, T), X)
        self._du_dx_func = sympy.lambdify((X, T), _du_dx)
        _du_dt = sympy.diff(self.u(X, T), T)
        self._du_dt_func = sympy.lambdify((X, T), _du_dt)

    def du_dx(self, x, t):
        """Strain (du/dx)

        Note:
            (eq.4, P.217)
            When a particle has a stable valocity, the strain is zero.
            Only when the particle volocity is changing, (i.e. the
            particle has an acceleration), then the strain is non-zero.
        """
        return self._du_dx_func(x, t)

    def du_dt(self, x, t):
        """Particle velocity (du/dt)

        Note:
            df_dt should be = -c*df_dx for a forward wave (eq.38, P.223)
            the sign is inverted for a backward wave
        """
        return self._du_dt_func(x, t)

    def k(self, x, t):
        """Kinetic energy (eq.73, P.229)

        Note:
            k should be = v
        """
        return 1/2*self.m*self.du_dt(x, t)**2

    def v(self, x, t):
        """Elastic energy (eq.74, P.229)

        Note:
            v should be = k
        """
        return 1/2*self.E*self.A*self.du_dx(x, t)**2

    def e(self, x, t):
        """Total energy (eq.80, P.230)"""
        return self.k(x, t) + self.v(x, t)

    def P(self, x, t):
        """Power (eq.81, P.230)"""
        return -self.m*self.c**2*self.du_dx(x, t)*self.du_dt(x, t)

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
        ).to_edge(UP)
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
        ).next_to(axes, 3*DOWN)
        axes_labels2 = axes2.get_axis_labels(y_label="")

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        # Propagating wave
        f_plot = axes.plot(lambda x: self.u(x, 0), color=WHITE)

        def update_f_plot(mobj):
            mobj.become(axes.plot(lambda x: self.u(
                x, t.get_value()), color=WHITE))
        f_plot.add_updater(update_f_plot)
        label_f = MathTex("u", color=WHITE)
        verbose_f = Tex("Particle Displacement", color=WHITE)

        # df/dx
        dfdx_plot = axes.plot(lambda x: self.du_dx(x, 0), color=BLUE)

        def update_dfdx_plot(mobj):
            mobj.become(axes.plot(lambda x: self.du_dx(
                x, t.get_value()), color=BLUE))
        dfdx_plot.add_updater(update_dfdx_plot)
        label_dfdx = MathTex(r"u\prime", color=BLUE)
        verbose_dfdx = Tex("Strain", color=BLUE)

        # df/dt
        dfdt_plot = axes2.plot(lambda x: self.du_dt(x, 0), color=GREEN)

        def update_dfdt_plot(mobj):
            mobj.become(axes2.plot(lambda x: self.du_dt(
                x, t.get_value()), color=GREEN))
        dfdt_plot.add_updater(update_dfdt_plot)
        label_dfdt = MathTex(r"\dot{u}", color=GREEN)
        verbose_dfdt = Tex("Particle Velocity", color=GREEN)

        # Kinetic energy
        k_plot = axes.plot(lambda x: self.k(x, 0), color=RED)

        def update_k_plot(mobj):
            mobj.become(axes.plot(lambda x: self.k(
                x, t.get_value()), color=RED))
        k_plot.add_updater(update_k_plot)
        label_k = MathTex("k", color=RED)
        verbose_k = Tex("Kinetic Energy", color=RED)

        # Elastic energy
        v_plot = axes2.plot(lambda x: self.v(x, 0), color=PURPLE)

        def update_v_plot(mobj):
            mobj.become(axes2.plot(lambda x: self.v(
                x, t.get_value()), color=PURPLE))
        v_plot.add_updater(update_v_plot)
        label_v = MathTex("v", color=PURPLE)
        verbose_v = Tex("Elastic Energy", color=PURPLE)

        # Total energy
        e_plot = axes2.plot(lambda x: self.e(x, 0), color=ORANGE)

        def update_e_plot(mobj):
            mobj.become(axes2.plot(lambda x: self.e(
                x, t.get_value()), color=ORANGE))
        e_plot.add_updater(update_e_plot)
        label_e = MathTex("e", color=ORANGE)
        verbose_e = Tex("Total Energy", color=ORANGE)

        # Power
        P_plot = axes.plot(lambda x: self.P(x, 0), color=YELLOW)

        def update_P_plot(mobj):
            mobj.become(axes.plot(lambda x: self.P(
                x, t.get_value()), color=YELLOW))
        P_plot.add_updater(update_P_plot)
        label_P = MathTex("P", color=YELLOW)
        verbose_P = Tex("Power", color=YELLOW)

        label_f.to_corner(UL)
        label_dfdx.next_to(label_f, RIGHT)
        label_dfdt.next_to(label_dfdx, RIGHT)
        label_k.next_to(label_dfdt, RIGHT)
        label_v.next_to(label_k, RIGHT)
        label_e.next_to(label_v, RIGHT)
        label_P.next_to(label_e, RIGHT)

        verbose_f.to_corner(UL)
        verbose_dfdx.next_to(verbose_f, RIGHT)
        verbose_dfdt.next_to(verbose_dfdx, RIGHT)
        verbose_k.next_to(verbose_dfdt, RIGHT)
        verbose_v.next_to(verbose_k, RIGHT)
        verbose_e.next_to(verbose_v, RIGHT)
        verbose_P.next_to(verbose_e, RIGHT)

        function_labels = VGroup(label_f,
                                 label_dfdx,
                                 label_dfdt,
                                 label_k,
                                 label_v,
                                 label_e,
                                 label_P).next_to(axes2, DOWN)

        verbose_labels = VGroup(verbose_f,
                                verbose_dfdx,
                                verbose_dfdt,
                                verbose_k,
                                verbose_v,
                                verbose_e,
                                verbose_P).next_to(function_labels, DOWN).scale(0.5)
        self.add(axes, axes_labels)
        self.add(axes2, axes_labels2)

        self.add(t, f_plot, dfdt_plot, dfdx_plot,
                 k_plot, v_plot, e_plot, P_plot)

        self.add(function_labels)
        self.add(verbose_labels)
        self.wait(2*PI)


class InterfaceCondition(Scene):
    """
    omega is the same across the interface
    u_i and u_r has the same gamma (gamma1)
    u_t has gamma2
    """
    rho1 = 1  # density (assumed constant)
    rho2 = 1  # density (assumed constant)
    E1 = 1  # elastic modulus (assumed constant)
    E2 = 1  # elastic modulus (assumed constant)
    A1 = 1  # cross section area (assumed constant)
    A2 = 1  # cross section area (assumed constant)
    omega = 1  # angular frequency (assumed constant) (eq.50, P.225)
    m1 = rho1*A1  # mass of the infinitesimal element dx (above eq.10, P.218)
    m2 = rho2*A2  # mass of the infinitesimal element dx (above eq.10, P.218)
    c1 = (E1/rho1)**(1/2)  # wave speed (eq.10, P.218)
    c2 = (E2/rho2)**(1/2)  # wave speed (eq.10, P.218)
    gamma1 = omega / c1  # wave number (eq.53, P.225)
    gamma2 = omega / c2  # wave number (eq.53, P.225)
    Z1 = rho1*c1  # acoustic impedance (eq.46, P.224)
    Z2 = rho2*c2  # acoustic impedance (eq.46, P.224)

    def theta_forward(self, x, t):
        """The single variable for forward wave

        Note: (eq.158, P.241)
        """
        return x - self.c1*t

    def theta_backward(self, x, t):
        """The single variable for backward wave

        Note: (eq.158, P.241)
        """
        return x + self.c1*t

    def f_forward(self, x, t):
        """The wave function"""
        return sympy.cos(self.gamma1*self.theta_forward(x, t))

    def f_backward(self, x, t):
        """The wave function"""
        return sympy.cos(self.gamma1*self.theta_backward(x, t))

    def u_i(self, x, t):
        """Incident wave

        Note: Can use cos() for short or use e^i*theta
            to only retain the real part
        """
        return self.f_forward(x, t)

    def u_r(self, x, t):
        """Reflected wave

        Note: (eq.128, P.236)
            Need to use the backward wave function f_backward(),
            because the u_hat_r and u_hat_t has opposite
            propagating direction when converting to scaler.

            !!!Adding a negative sign on u_i() will not work.
        """
        return ((self.Z1*self.A1-self.Z2*self.A2)
                / (self.Z1*self.A1+self.Z2*self.A2)) * self.f_backward(x, t)

    def u_t(self, x, t):
        """Transmitted wave

        Note: (eq.128, P.236)
        """
        return ((2*self.Z1*self.A1)
                / (self.Z1*self.A1+self.Z2*self.A2))*self.u_i(x, t)

    def u1(self, x, t):
        """Total wave in material 1"""
        return self.u_i(x, t) + self.u_r(x, t)

    def u2(self, x, t):
        """Total wave in material 2"""
        return self.u_t(x, t)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1.5, 1.5, .5],
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
        ).to_edge(UP)
        axes_labels = axes.get_axis_labels(y_label="")
        axes2 = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1.5, 1.5, .5],
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
        step_plot = 0.2
        ui = axes.plot(lambda x: self.u_i(x, 0),
                       color=BLUE, x_range=[-2*PI, 0, step_plot])

        def update_ui(mobj):
            mobj.become(axes.plot(lambda x: self.u_i(
                x, t.get_value()), color=BLUE, x_range=[-2*PI, 0, step_plot]))
        ui.add_updater(update_ui)

        ur = axes.plot(lambda x: self.u_r(x, 0),
                       color=RED, x_range=[-2*PI, 0, step_plot])

        def update_ur(mobj):
            mobj.become(axes.plot(lambda x: self.u_r(
                x, t.get_value()), color=RED, x_range=[-2*PI, 0, step_plot]))
        ur.add_updater(update_ur)

        ut = axes.plot(lambda x: self.u_t(x, 0), color=GREEN,
                       x_range=[0, 2*PI, step_plot])

        def update_ut(mobj):
            mobj.become(axes.plot(lambda x: self.u_t(
                x, t.get_value()), color=GREEN, x_range=[0, 2*PI, step_plot]))
        ut.add_updater(update_ut)

        u1 = axes2.plot(lambda x: self.u1(x, 0),
                        color=ORANGE, x_range=[-2*PI, 0, step_plot])

        def update_u1(mobj):
            mobj.become(axes2.plot(lambda x: self.u1(
                x, t.get_value()), color=ORANGE, x_range=[-2*PI, 0, step_plot]))
        u1.add_updater(update_u1)

        u2 = axes2.plot(lambda x: self.u2(x, 0), color=PURPLE,
                        x_range=[0, 2*PI, step_plot])

        def update_u2(mobj):
            mobj.become(axes2.plot(lambda x: self.u2(
                x, t.get_value()), color=PURPLE, x_range=[0, 2*PI, step_plot]))
        u2.add_updater(update_u2)

        rho1 = MathTex(r"\rho_1" + f"={self.rho1}", color=ORANGE)
        rho2 = MathTex(r"\rho_2" + f"={self.rho2}",
                       color=PURPLE).next_to(rho1, DOWN)
        rho = VGroup(rho1, rho2).next_to(axes2, DOWN)

        E1 = MathTex(r"E_1" + f"={self.E1}", color=ORANGE)
        E2 = MathTex(r"E_2" + f"={self.E2}", color=PURPLE).next_to(E1, DOWN)
        E = VGroup(E1, E2).next_to(rho, RIGHT)

        A1 = MathTex(r"A_1" + f"={self.A1}", color=ORANGE)
        A2 = MathTex(r"A_2" + f"={self.A2}", color=PURPLE).next_to(A1, DOWN)
        A = VGroup(A1, A2).next_to(E, RIGHT)

        texts = VGroup(rho, E, A).scale(0.8).next_to(axes2, DOWN)

        axis_group = VGroup(axes, axes2, axes_labels, axes_labels2)
        u_group = VGroup(ui, ur, ut, u1, u2)
        all = VGroup(axis_group, u_group, texts).move_to(ORIGIN)

        self.add(t)
        self.add(all)
        self.wait(2*PI)


class FlexuralWavesInABeam(Scene):
    rho = 1  # density (assumed constant)
    E = 1  # elastic modulus (assumed constant)
    omega = 1  # angular frequency (assumed constant) (eq.50, P.225)
    h = 1  # cross section height (assumed constant) (above eq.196, P.246)
    b = 1  # cross section width (assumed constant) (above eq.196, P.246)
    I = 1/12*h*b**3  # moment of inertia (eq.196, P.246)
    A = h*b  # cross section area
    m = rho*A  # mass of the infinitesimal element dx (above eq.10, P.218)
    # c = (E/rho)**(1/2)  # wave speed (eq.10, P.218)
    # gamma = omega / c  # wave number (eq.53, P.225)
    a = ((E*h**2)/(12*rho))**(1/4)  # characteristic length (eq.196, P.246)
    gamma = omega**(1/2) / a  # wave number (eq.196, P.246)
    c_F = omega / gamma  # flexural wave speed (eq.209, P.247)

    def w(self, x, t):
        """Vertical displacement"""
        A1 = 1
        A2 = 1
        propagating = A1*np.exp(1j*(self.gamma*x-self.omega*t))
        evanescent = A2*np.exp(-self.gamma*x)*np.exp(-1j*self.omega*t)
        return propagating+evanescent

    def u(self, x, z, t):
        """In-plane displacement"""
        return -1j*self.gamma*z*self.w(x, t)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[-2*PI, 2*PI, PI],
            y_range=[-1, 1, 1],
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
            },
            tips=False,  # 坐标箭头
        )
        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))
        forward_plot = axes.plot(lambda x: self.w(x, 0).real, color=BLUE,
                                 x_range=[-2*PI, 2*PI, PI/4])
        axes_labels = axes.get_axis_labels(
            x_label=Tex(r"x"), y_label=Tex(r"w(x,0)"))

        def update_forward_plot(mobj):
            mobj.become(axes.plot(lambda x: self.w(x, t.get_value()).real, color=BLUE,
                                  x_range=[-2*PI, 2*PI, PI/4]))
        forward_plot.add_updater(update_forward_plot)
        self.add(t)
        self.add(axes, axes_labels, forward_plot)
        self.wait(2*PI)


class FlexuralWavesVField(Scene):
    rho = 1  # density (assumed constant)
    E = 1  # elastic modulus (assumed constant)
    omega = 1  # angular frequency (assumed constant) (eq.50, P.225)
    h = 20  # cross section height (assumed constant) (above eq.196, P.246)
    b = 1  # cross section width (assumed constant) (above eq.196, P.246)
    I = b*h**3/12  # moment of inertia (eq.196, P.246)
    A = h*b  # cross section area
    m = rho*A  # mass of the infinitesimal element dx (above eq.10, P.218)
    a = ((E*h**2)/(12*rho))**(1/4)  # characteristic length (eq.196, P.246)
    gamma = omega**(1/2) / a  # wave number (eq.200, P.246)
    c_F = omega / gamma  # flexural wave speed (eq.209, P.247)

    def w(self, x, t):
        """vertical displacement

        Note: (eq.211, P.247)
        """
        A2 = 1
        A4 = 1
        # propagating = A2*np.cos(self.gamma*x-self.omega*t)
        propagating = A2*np.exp(1j*(self.gamma*x-self.omega*t))
        # evanescent = A4*np.exp(-self.gamma*x)*np.exp(-1j*self.omega*t)
        return propagating

    def u(self, x, z, t):
        """in-plane displacement

        Note: according to eq.188, P.245, u(x,z,t) = -z*w'(x,t),
        but u(x,z,t) needs to be -(-z*w'(x,t)) to match the graph the author provided.
        """
        return 1j*self.gamma*z*self.w(x, t)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 60, 10],
            y_range=[-10, 10, 5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 60.01, 10),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-10.01, 10.01, 5),
            },
            tips=False,  # 坐标箭头
        )

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        self.add(axes)

        def func(x, z):
            vertical = self.w(x, t.get_value()).real
            horizontal = self.u(x, z, t.get_value()).real
            return horizontal, vertical

        # adjust the rate of x and y axis to make the vector field look good
        rate_x = .2
        rate_y = 1

        def Field():
            vgroup = VGroup()
            for x in np.arange(0, 60.1, 1):
                for z in np.arange(-10, 10.1, 1):
                    result = func(x, z)
                    vector = Arrow(start=axes.coords_to_point(x, z),
                                   end=axes.coords_to_point(
                                       x, z+rate_y*result[1]),
                                   buff=0)
                    vgroup.add(vector)
            return vgroup

        field = Field()

        def update_field(mobj):
            vgroup = Field()
            mobj.become(vgroup)

        field.add_updater(update_field)
        self.add(field)
        self.add(t)
        self.wait(2*PI)


class FlexuralWaveDispersionCurve(Scene):
    """c_F change with omega"""
    rho = 200  # density (assumed constant)
    E = 1  # elastic modulus (assumed constant)
    h = 100  # cross section height (assumed constant) (above eq.196, P.246)
    b = 1  # cross section width (assumed constant) (above eq.196, P.246)
    I = b*h**3/12  # moment of inertia (eq.196, P.246)
    A = h*b  # cross section area
    m = rho*A  # mass of the infinitesimal element dx (above eq.10, P.218)
    a = (E*I/m)**(1/4)  # characteristic length (eq.196, P.246)

    def c_F(self, omega):
        """flexural wave speed (eq.209, P.247)

        Args:
            omega (float): angular frequency
        """
        return self.a*omega**(1/2)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 4, 1],
            y_range=[0, 3, 1],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 4.01, 1),
            },
            y_axis_config={
                "numbers_to_include": np.arange(0, 3.01, 1),
            },
            tips=False,  # 坐标箭头
        )
        label = axes.get_axis_labels(x_label="\omega", y_label="c_F")
        self.add(axes, label)
        dispersion_curve = axes.plot(lambda omega: self.c_F(omega),
                                     x_range=[0, 4, .01],
                                     color=RED)
        self.add(dispersion_curve)


class FlexuralWaveDispersion(Scene):
    """c_F change with omega"""
    rho = 200  # density (assumed constant)
    E = 1  # elastic modulus (assumed constant)
    h = 100  # cross section height (assumed constant) (above eq.196, P.246)
    b = 1  # cross section width (assumed constant) (above eq.196, P.246)
    I = b*h**3/12  # moment of inertia (eq.196, P.246)
    A = h*b  # cross section area
    m = rho*A  # mass of the infinitesimal element dx (above eq.10, P.218)
    a = (E*I/m)**(1/4)  # characteristic length (eq.196, P.246)

    # angular frequency (assumed decreasing with time) (eq.50, P.225)
    omega = 100

    @property
    def gamma(self):
        """wave number (eq.196, P.246)"""
        return self.omega**(1/2) / self.a

    @property
    def c_F(self):
        """flexural wave speed (eq.209, P.247)"""
        return self.a*self.omega**(1/2)

    def w(self, x, t):
        """flexural wave vertical displacement"""
        return np.exp(1j*(self.gamma*x-self.omega*t))

    def gaussian(self, x, t, tau=None):
        """gaussian distribution with changing center and width

        Args:
            sigma (float, optional): width. Defaults bind to time.

        https://en.wikipedia.org/wiki/Normal_distribution#Alternative_parameterizations
        """
        # mu defining the center of the distribution
        mu = self.gamma*x-self.omega*t

        #  precision tau defining the width of the distribution
        # higher precision means narrower distribution
        # tau = 1/sigma**2
        # Note that the peak decreases as width increases
        tau = np.exp(-t) if tau is None else tau

        return 5*np.sqrt(tau/(2*PI))*np.exp(-tau*(x-mu)**2/2)

    def u(self, x, t):
        """
        gaussian fixed width but propagating with time times wave
        """
        return self.gaussian(x, t, tau=.05)*self.w(x, t)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 20*PI, PI],
            y_range=[-1, 1, 0.5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
        )
        self.add(axes)
        t = ValueTracker(0)

        def update_t(m, dt):
            m.increment_value(dt)
            # make omega decrease with time
            # output with high quality need /4 (because of the 4x fps)
            self.omega -= .025*self.omega/4
        t.add_updater(update_t)

        def get_plot():
            return axes.plot(lambda x: self.u(x, t.get_value()), color=BLUE)
        plot = always_redraw(get_plot)
        omega_Label = MathTex(r"\omega =").next_to(axes, DOWN)
        c_F_Label = MathTex(r"c_F =").next_to(omega_Label, DOWN)

        def get_omega():
            return DecimalNumber(self.omega, num_decimal_places=2).next_to(omega_Label, RIGHT)
        omega = always_redraw(get_omega)

        def get_c_F():
            return DecimalNumber(self.c_F, num_decimal_places=2).next_to(c_F_Label, RIGHT)
        c_F = always_redraw(get_c_F)

        self.add(omega_Label, c_F_Label)
        self.add(omega)
        self.add(c_F)

        self.add(t)
        self.add(plot)
        self.wait(2*PI)


class GaussianToneBurst(Scene):
    c = 1  # wave speed
    k = 5  # wave number

    def f(self, x, t):
        """a forward propagating wave"""
        return np.exp(1j*self.k*(x-self.c*t))

    def gaussian(self, x, t, tau=None):
        """gaussian distribution with changing center and width

        Args:
            sigma (float, optional): width. Defaults bind to time.

        https://en.wikipedia.org/wiki/Normal_distribution#Alternative_parameterizations
        """
        # mu defining the center of the distribution
        mu = self.k*(x-self.c*t)

        #  precision tau defining the width of the distribution
        # higher precision means narrower distribution
        # tau = 1/sigma**2
        # Note that the peak decreases as width increases
        tau = np.exp(-t**1) if tau is None else tau

        return 2*np.sqrt(tau/(2*PI))*np.exp(-tau*(x-mu)**2/2)

    def u(self, x, t):
        """without dispersion

        an unchanged gaussian distribution * wave 
        becomes a tone burst without dispersion
        """
        return self.gaussian(x, t, sigma=5)*self.f(x, t)

    def u_d(self, x, t):
        """with dispersion

        # https://en.wikipedia.org/wiki/Wave_packet#Dispersive
        # https://en.wikipedia.org/wiki/Normal_distribution#Alternative_parameterizations
        a changing gaussian distribution * wave
        becomes a tone burst with dispersion
        """
        # return ((2/PI)**(1/4)/(1+2j*t)**(1/2))*np.exp(-1/4*self.k**2)*np.exp(-1/(1+2j*t)*(x-1j*self.k/2)**2)
        return self.gaussian(x, t)*self.f(x, t)

    def fft(self, u):
        """fast fourier transform

        Args:
            u (array): wave function

        Returns:
            array: frequency spectrum
        """
        return np.fft.fftshift(np.abs(np.fft.fft(u).real))

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[-PI/4, 4*PI, PI/4],
            y_range=[-1, 1, 0.5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=3,
            axis_config={"color": GREEN},
            tips=False,  # 坐标箭头
        ).to_edge(UP)
        fft_axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 100, 1],
            y_range=[0, 5, 1],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=3,
            axis_config={"color": GREEN, "include_ticks": False},
            tips=False,  # 坐标箭头

        ).next_to(axes, DOWN)
        self.add(axes, fft_axes)
        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        def get_plot():
            return axes.plot(lambda x: self.u_d(x, t.get_value()).real, color=BLUE)
        plot = always_redraw(get_plot)

        def get_fft_plot():
            return fft_axes.plot(lambda x: self.fft(self.u_d(x, t.get_value())), color=BLUE, use_vectorized=True)
        fft_plot = always_redraw(get_fft_plot)
        self.add(t)
        self.add(plot)
        self.add(fft_plot)
        self.wait(4*PI)


class GroupVelocity(Scene):
    gamma1 = 10  # wave speed
    gamma2 = 14
    omega1 = 1  # frequency
    omega2 = 4

    d_gamma = gamma2-gamma1
    d_omega = omega2-omega1

    gamma_avg = (gamma1+gamma2)/2
    omega_avg = (omega1+omega2)/2

    c_g = d_omega/d_gamma  # group velocity (eq.221, P.252)
    c_avg = omega_avg/gamma_avg  # average phase velocity (eq.220, P.251)

    def u(self, x, t):
        """carrier wave

        Note: constructed by adding two waves with different wave speed and frequency
            (eq.215, P.251)
        """
        return 1/2*(np.cos(self.gamma1*x-self.omega1*t) + np.cos(self.gamma2*x-self.omega2*t))

    def g(self, x, t):
        """modulation wave

        Note: (eq.219, P.251) first part of the equation
        """
        return np.cos(self.d_gamma*x/2-self.d_omega*t/2)

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 2*PI, PI/4],
            y_range=[-1, 1, 0.5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
        )
        self.add(axes)

        c_avg_Label = MathTex(
            r"c_{av} = "+f"{self.c_avg:.2f}", color=BLUE).next_to(axes, DOWN)
        c_g_Label = MathTex(
            r"c_g = "+f"{self.c_g:.2f}", color=RED).next_to(c_avg_Label, DOWN)
        self.add(c_avg_Label, c_g_Label)
        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        def get_plot():
            return axes.plot(lambda x: self.u(x, t.get_value()), color=BLUE)
        plot = always_redraw(get_plot)

        def get_gplot():
            return axes.plot(lambda x: self.g(x, t.get_value()), color=RED)
        gplot = always_redraw(get_gplot)

        def get_ngplot():
            return axes.plot(lambda x: -self.g(x, t.get_value()), color=RED)
        ngplot = always_redraw(get_ngplot)
        self.add(t)
        self.add(plot)
        self.add(gplot)
        self.add(ngplot)
        self.wait(2*PI)


class ShearVerticalWaveVField(Scene):
    rho = 1  # density
    G = 1  # shear modulus
    omega = 1  # angular frequency

    c = (G/rho)**(1/2)  # wave speed
    gamma = omega/c  # wave number

    def w(self, x, t):
        """shear wave

        Note: Purely shear wave and no flexure takes place. (P. 268)
        """
        return np.exp(1j*(self.gamma*x-self.omega*t))

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 60, 10],
            y_range=[-10, 10, 5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 60.01, 10),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-10.01, 10.01, 5),
            },
            tips=False,  # 坐标箭头
        )

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        self.add(axes)

        def func(x, z):
            vertical = self.w(x, t.get_value()).real
            return vertical

        # adjust the rate of y axis to make the vector field look good
        rate_y = 1

        def Field():
            vgroup = VGroup()
            for x in np.arange(0, 60.1, 1):
                for z in np.arange(-10, 10.1, 1):
                    result = func(x, z)
                    vector = Arrow(start=axes.coords_to_point(x, z),
                                   end=axes.coords_to_point(
                                       x, z+rate_y*result),
                                   buff=0)
                    vgroup.add(vector)
            return vgroup

        field = Field()

        def update_field(mobj):
            vgroup = Field()
            mobj.become(vgroup)

        field.add_updater(update_field)
        self.add(field)
        self.add(t)
        self.wait(2*PI)


class PressureWaveVField(Scene):
    rho = 1  # density
    E = 40  # elastic modulus
    omega = 2  # angular frequency
    # https://en.wikipedia.org/wiki/Poisson%27s_ratio
    v = 0  # Poisson's ratio ranging between 0.0 and 0.5

    # pressure wave speed (eq.474, p.292)
    c = (((1-v)/((1+v)*(1-2*v)))*(E/rho))**(1/2)
    gamma = omega/c  # wave number

    def u(self, x, t):
        """pressure wave

        Note: The wave propagates in the direction of the oscillation. (eq.485, p.294)
        """
        return np.exp(1j*(self.gamma*x-self.omega*t))

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 60, 10],
            y_range=[-10, 10, 5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 60.01, 10),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-10.01, 10.01, 5),
            },
            tips=False,  # 坐标箭头
        )

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        self.add(axes)

        def func(x, z):
            horizontal = self.u(x, t.get_value()).real
            return horizontal

        # adjust the rate of x axis to make the vector field look good
        rate_x = 1

        def Field():
            vgroup = VGroup()
            for x in np.arange(0, 60.1, 1):
                for z in np.arange(-10, 10.1, 1):
                    result = func(x, z)
                    vector = Arrow(start=axes.coords_to_point(x, z),
                                   end=axes.coords_to_point(
                                       x+rate_x*result, z),
                                   buff=0)
                    vgroup.add(vector)
            return vgroup

        field = Field()

        def update_field(mobj):
            vgroup = Field()
            mobj.become(vgroup)

        field.add_updater(update_field)
        self.add(field)
        self.add(t)
        self.wait(2*PI)


class RayleighWaveVField(Scene):
    A = .1  # amplitude
    omega = 2  # angular frequency
    c_P = 5  # pressure wave speed
    c_S = 5  # shear-vertical wave speed
    v = 0.5  # Poisson's ratio ranging between 0.0 and 0.5

    # a common approximation of the rayleigh wave speed (eq.28, p.312)
    c_R = c_S*((0.87+1.12*v)/(1+v))
    xi = omega/c_R  # wave number (eq.29, p.313)
    alpha = (xi**2*(1-c_R**2/c_P**2))**(1/2)  # (eq.23, p.312)
    beta = (xi**2*(1-c_R**2/c_S**2))**(1/2)  # (eq.23, p.312)

    def u_x(self, x, y, t):
        """pressure wave

        Note: Add a negative y to fit the author provided animation.
        """
        y = -y
        u_x_y = self.A*1j*(self.xi*np.exp(-self.alpha*y)
                           - (self.beta**2+self.xi**2)
                           / (2*self.xi)*np.exp(-self.beta*y))
        return u_x_y*np.exp(1j*(self.xi*x-self.omega*t))

    def u_y(self, x, y, t):
        """shear-vertical wave

        Note: Add a negative y to fit the author provided animation.
        """
        y = -y
        u_y_y = self.A*(- self.alpha*np.exp(-self.alpha*y)
                        + (self.beta**2+self.xi**2)
                        / (2*self.beta)*np.exp(-self.beta*y))
        return u_y_y*np.exp(1j*(self.xi*x-self.omega*t))

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 60, 10],
            y_range=[0, 20, 5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 60.01, 10),
            },
            y_axis_config={
                "numbers_to_include": np.arange(0.01, 20.01, 5),
            },
            tips=False,  # 坐标箭头
        )

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        self.add(axes)

        def func(x, y):
            """P+SV Solution"""
            P = self.u_x(x, y, t.get_value()).real
            SV = self.u_y(x, y, t.get_value()).real
            return P, SV

        # adjust the rate of axis to make the vector field look good
        rate_x = 2
        rate_y = .7

        def Field():
            vgroup = VGroup()
            for x in np.arange(0, 60.1, 1):
                for y in np.arange(0, 20.1, 1):
                    P, SV = func(x, y)
                    vector = Arrow(start=axes.coords_to_point(x, y),
                                   end=axes.coords_to_point(
                                       x+rate_x*P, y+rate_y*SV),
                                   buff=0)
                    vgroup.add(vector)
            return vgroup

        field = Field()

        def update_field(mobj):
            vgroup = Field()
            mobj.become(vgroup)

        field.add_updater(update_field)
        self.add(field)
        self.add(t)
        self.wait(2*PI)


class SHWaveXSymmetricVField(Scene):
    d = 1  # half of the thickness of the plate
    C1 = 1  # amplitude 1 (eq.48, p.316) symmetric when C1=0 and C2!=0
    C2 = 0  # amplitude 2 (eq.48, p.316) anti-symmetric when C2=0 and C1!=0

    # the value of omega, xi, c_S should match the requirement of the mode
    omega = 1  # SH angular frequency
    xi = 1  # SH wave number (eq.34, p.314)
    # c_S = 1  # SH wave speed (eq.36, p.315)

    # angular frequency of the standing wave(eq.40, p.315)
    # eta = ((omega**2/c_S**2)-xi**2)**(1/2)

    n = 0  # mode number
    if C1 == 0 and C2 != 0:  # symmetric
        eta = n*PI/d  # (eq.52, p.316)
    elif C2 == 0 and C1 != 0:  # anti-symmetric
        eta = (2*n+1)*PI/2/d  # (eq.56, p.317)

    # select two to be constants, and let the other one be calculated
    # derived from (eq.40, p.315)
    # omega = ((eta**2+xi**2)*c_S**2)**(1/2)  # SH angular frequency
    # xi = ((omega**2/c_S**2)-eta**2)**(1/2)  # SH wave number (eq.34, p.314)
    c_S = (omega**2/(eta**2+xi**2))**(1/2)  # SH wave speed (eq.36, p.315)

    def h(self, y):
        """standing wave

        Note: symmetric when C1=0 and anti-symmetric when C2=0
        """
        return self.C1*np.sin(self.eta*y)+self.C2*np.cos(self.eta*y)

    def u_z(self, x, y, t):
        """SH waves

        Note: (eq.36, p.314)
        """
        return self.h(y)*np.exp(1j*(self.xi*x-self.omega*t))

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 30, 10],
            y_range=[-self.d, self.d, .5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 30.01, 10),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-self.d-.01, self.d+.01, .5),
            },
            tips=False,  # 坐标箭头
        )

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        omega = MathTex(r"\omega = "+f"{self.omega:.2f}")
        xi = MathTex(r"\xi = "+f"{self.xi:.2f}").next_to(omega, RIGHT)
        c_S = MathTex(r"c_S = "+f"{self.c_S:.2f}").next_to(xi, RIGHT)
        positive = Tex(r"+", color=GREEN).next_to(c_S, RIGHT)
        negative = Tex(r"-", color=BLUE).next_to(positive, RIGHT)
        text = VGroup(omega, xi, c_S, positive,
                      negative).next_to(axes, UP)
        symmetricity_str = None
        if self.C1 == 0 and self.C2 != 0:
            symmetricity_str = "Symmetric"
        elif self.C1 != 0 and self.C2 == 0:
            symmetricity_str = "AntiSymmetric"
        symmetricity = Tex(symmetricity_str)
        eta = MathTex(
            r"\eta = "+f"{self.eta:.2f}").next_to(symmetricity, RIGHT)
        d = MathTex(r"d = "+f"{self.d:.2f}").next_to(eta, RIGHT)
        mode = MathTex(r"n_{mode} = "+f"{self.n}").next_to(d, RIGHT)
        text2 = VGroup(symmetricity, eta, d, mode).next_to(axes, DOWN)
        self.add(axes)
        self.add(text)
        self.add(text2)

        def func(x, y):
            return self.u_z(x, y, t.get_value()).real

        # adjust the rate of axis to make the vector field look good
        rate_z = .05

        def Field():
            vgroup = VGroup()
            for x in np.arange(0, 30.1, .5):
                for y in np.arange(-self.d, self.d+.1, .1):
                    result = func(x, y)
                    vector = Dot(radius=rate_z*result if result > 0 else -rate_z*result,
                                 color=GREEN if result > 0 else BLUE).move_to(axes.coords_to_point(x, y))
                    vgroup.add(vector)
            return vgroup

        field = Field()

        def update_field(mobj):
            vgroup = Field()
            mobj.become(vgroup)

        field.add_updater(update_field)

        self.add(field)
        self.add(t)
        self.wait(2*PI)


class SHDispersionCurve(Scene):
    d = 5  # half of the thickness of the plate
    C1 = 0  # amplitude 1 (eq.48, P.316) symmetric when C1=0 and C2!=0
    C2 = 1  # amplitude 2 (eq.48, P.316) anti-symmetric when C2=0 and C1!=0

    # the value of omega, xi, c_S should match the requirement of the mode
    # omega = 1  # SH angular frequency
    xi = 1  # SH wave number (eq.34, P.314)
    c_S = 1  # SH wave speed(eq.36, P.315)

    n = 0  # mode number

    @property
    def eta(self):
        """SH Wave frequency"""
        if self.C1 == 0 and self.C2 != 0:  # symmetric
            eta = self.n*PI/self.d  # (eq.52, P.316)
        elif self.C2 == 0 and self.C1 != 0:  # anti-symmetric
            eta = (2*self.n+1)*PI/2/self.d  # (eq.56, P.317)
        return eta

    def c(self, omega):
        """SH Wave Speed"""
        result = self.c_S/np.sqrt(1-(self.eta*self.d)
                                  ** 2*(self.c_S/(omega*self.d))**2)
        return result

    def omega_critical(self):
        """Critical Angular Frequency

        Note: Cut-off Frequency (eq.63, P.320)
        """
        return self.c_S*self.eta

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 5, .5],  # x must be greater than 0
            y_range=[0.9, 2, .5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 5.01, .5),
            },
            y_axis_config={
                "numbers_to_include": [1, 1.5, 2],
            },
            tips=False,  # 坐标箭头
        )
        axes_labels = axes.get_axis_labels(
            x_label=Tex(r"$\omega$"),
            y_label=Tex(r"$c(\omega)$"),
        )
        self.add(axes)

        S0 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=RED,
                       use_vectorized=True)
        self.n = 1
        S1 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=RED,
                       use_vectorized=True)
        self.n = 2
        S2 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=RED,
                       use_vectorized=True)
        self.n = 3
        S3 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=RED,
                       use_vectorized=True)

        self.n = 0
        self.C1 = 1
        self.C2 = 0
        A0 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=BLUE,
                       use_vectorized=True)
        self.n = 1
        A1 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=BLUE,
                       use_vectorized=True)
        self.n = 2
        A2 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=BLUE,
                       use_vectorized=True)
        self.n = 3
        A3 = axes.plot(lambda x: self.c(x),
                       x_range=[self.omega_critical()+.01, 5, .01],
                       discontinuities=[self.omega_critical()],
                       dt=0.01,
                       color=BLUE,
                       use_vectorized=True)

        self.add(S0, S1, S2, S3, A0, A1, A2, A3)
        # hide the values larger than the y-axis
        mask = Rectangle(height=4, width=12, color=BLACK).shift(
            UP*4.05).set_fill(BLACK, opacity=1)
        self.add(mask)
        equation = MathTex(
            r"c(\omega) = \frac{c_S}{\sqrt{1-(\eta d)^2(\frac{c_S}{\omega d})^2}}}")
        c_S = Tex(r"$c_S$="+f"{self.c_S}").next_to(equation, RIGHT)
        d = Tex(r"$d$="+f"{self.d}").next_to(c_S, RIGHT)
        above_axes = VGroup(equation, c_S, d).next_to(axes, UP)
        self.add(above_axes, axes_labels)

        eta_S = MathTex(r"\eta_S = \frac{n\pi}{d}", color=RED)
        eta_A = MathTex(
            r"\eta_A = \frac{(2n+1) \pi}{2d}", color=BLUE).next_to(eta_S, RIGHT)
        ns = MathTex(r"n = 0, 1, 2, 3").next_to(eta_A, RIGHT)
        below_axes = VGroup(eta_S, eta_A, ns).next_to(axes, DOWN)
        self.add(below_axes)


class LambWaveSymmetricVField(Scene):
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
        return (self.omega**2/self.c_P**2-self.xi**2)**(1/2)

    @property
    def eta_S(self):
        """(eq.83, P.325)"""
        return (self.omega**2/self.c_S**2-self.xi**2)**(1/2)

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
        return (self.Omega**2/self.k**2-self.xi_**2)**(1/2)

    @property
    def eta_S_(self):
        """(eq.105, P.328)"""
        return (self.Omega**2-self.xi_**2)**(1/2)

    @property
    def D_(self):
        """(eq.108, P.328)"""
        return (self.xi_**2-self.eta_S_**2)**2*np.cos(self.eta_P_)*np.sin(self.eta_S_)+4*self.xi_**2*self.eta_P_*self.eta_S_*np.sin(self.eta_P_)*np.cos(self.eta_S_)

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

    def u_x(self, x, y, t):
        """(eq. 127, P. 334)"""
        return -1j*self.C*(2*self.xi**2*self.eta_S*np.cos(self.eta_S*self.d)*np.cos(self.eta_P*y)-self.eta_S*(self.xi**2-self.eta_S**2)*np.cos(self.eta_P*self.d)*np.cos(self.eta_S*y))*np.exp(1j*(self.xi*x-self.omega*t))

    def u_y(self, x, y, t):
        """(eq. 127, P. 334)"""
        return self.C*self.xi*(2*self.eta_P*self.eta_S*np.cos(self.eta_S*self.d)*np.sin(self.eta_P*y)+(self.xi**2-self.eta_S**2)*np.cos(self.eta_P*self.d)*np.sin(self.eta_S*y))*np.exp(1j*(self.xi*x-self.omega*t))

    @property
    def c_L(self):
        """Lamb wave speed"""
        return self.omega/self.xi

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 60, 10],
            y_range=[-self.d, self.d, .5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 60.01, 10),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-self.d-.01, self.d+.01, .5),
            },
            tips=False,  # 坐标箭头
        )

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        self.xi, self.omega = self.get_xi_and_omega_by_fd(.2)  # S0

        def func(x, y):
            P = self.u_x(x, y, t.get_value()).real
            SV = self.u_y(x, y, t.get_value()).real
            return P, SV

        # adjust the rate of axis to make the vector field look good
        rate_x = .15
        rate_y = .03

        def Field():
            vgroup = VGroup()
            for x in np.arange(0, 60.1, 1):
                for y in np.arange(-self.d, self.d+.1, .1):
                    P, SV = func(x, y)
                    vector = Arrow(start=axes.coords_to_point(x, y),
                                   end=axes.coords_to_point(
                                       x+rate_x*P, y+rate_y*SV),
                                   buff=0)
                    vgroup.add(vector)
            return vgroup

        field = Field()

        def update_field(mobj):
            vgroup = Field()
            mobj.become(vgroup)

        field.add_updater(update_field)

        self.add(axes, field, t)
        self.wait(2*PI/self.omega)


class LambWaveAntiSymmetricVField(Scene):
    """Different in D, D_, u_x, u_y"""
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
        return (self.omega**2/self.c_P**2-self.xi**2)**(1/2)

    @property
    def eta_S(self):
        """(eq.83, P.325)"""
        return (self.omega**2/self.c_S**2-self.xi**2)**(1/2)

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
        return (self.Omega**2/self.k**2-self.xi_**2)**(1/2)

    @property
    def eta_S_(self):
        """(eq.105, P.328)"""
        return (self.Omega**2-self.xi_**2)**(1/2)

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

        So that the omega and xi result in desired wave mode

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

    def u_x(self, x, y, t):
        """(eq. 135, P. 336)"""
        return 1j*self.C*self.eta_S*(2*self.xi**2*np.sin(self.eta_S*self.d)*np.sin(self.eta_P*y)-(self.xi**2-self.eta_S**2)*np.sin(self.eta_P*self.d)*np.sin(self.eta_S*y))*np.exp(1j*(self.xi*x-self.omega*t))

    def u_y(self, x, y, t):
        """(eq. 135, P. 336)

        Note: the sign of the equation is different from the book, in order to generate the same wave field as the book
        """
        return -self.C*self.xi*(2*self.eta_P*self.eta_S*np.sin(self.eta_S*self.d)*np.cos(self.eta_P*y)+(self.xi**2-self.eta_S**2)*np.sin(self.eta_P*self.d)*np.cos(self.eta_S*y))*np.exp(1j*(self.xi*x-self.omega*t))

    @property
    def c_L(self):
        """Lamb wave speed"""
        return self.omega/self.xi

    def construct(self):
        axes = Axes(
            # 坐标轴数值范围和步长
            x_range=[0, 60, 10],
            y_range=[-self.d, self.d, .5],
            # 坐标轴长度（比例）
            x_length=12,
            y_length=4,
            axis_config={"color": GREEN},
            x_axis_config={
                "numbers_to_include": np.arange(0, 60.01, 10),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-self.d-.01, self.d+.01, .5),
            },
            tips=False,  # 坐标箭头
        )

        t = ValueTracker(0)
        t.add_updater(lambda m, dt: m.increment_value(dt))

        self.xi, self.omega = self.get_xi_and_omega_by_fd(.1)  # A0

        def func(x, y):
            P = self.u_x(x, y, t.get_value()).real
            SV = self.u_y(x, y, t.get_value()).real
            return P, SV

        # adjust the rate of axis to make the vector field look good
        rate_x = .5
        rate_y = .5

        def Field():
            vgroup = VGroup()
            for x in np.arange(0, 60.1, 1):
                for y in np.arange(-self.d, self.d+.1, .1):
                    P, SV = func(x, y)
                    vector = Arrow(start=axes.coords_to_point(x, y),
                                   end=axes.coords_to_point(
                                       x+rate_x*P, y+rate_y*SV),
                                   buff=0)
                    vgroup.add(vector)
            return vgroup

        field = Field()

        def update_field(mobj):
            vgroup = Field()
            mobj.become(vgroup)

        field.add_updater(update_field)

        self.add(axes, field, t)
        self.wait(2*PI/self.omega)
