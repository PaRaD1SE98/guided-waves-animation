# doc: https://docs.manim.community/en/stable/index.html
# run: manim -pqh scene.py <ClassName>
import math

from manim import *


class WavePropInBar(Scene):
    def func(self, x, c, t):
        X = x - c*t
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
        c = 1
        # 设置一个变量t，用于控制动画
        t = ValueTracker(0)

        sin_graph = axes.plot(lambda x: self.func(
            x, c, t.get_value()), color=BLUE)
        sin_graph.add_updater(lambda mobj: mobj.become(
            axes.plot(lambda x: self.func(x, c, t.get_value()), color=BLUE)))
        label = Tex("t = ").next_to(axes, UP)
        t_number = DecimalNumber(t.get_value(), num_decimal_places=2)
        t_number.add_updater(lambda mobj: mobj.set_value(
            t.get_value()).next_to(label, RIGHT))
        plot = VGroup(axes, sin_graph, t_number, axes_labels, label)
        self.add(plot)
        self.play(t.animate.set_value(10), rate_func=linear)


class EulerFormula(Scene):
    def construct(self):
        # axes
        x_axis_pos_y = 2.5
        x_axis_range = (-5, 5)
        x_start = np.array([x_axis_range[0], x_axis_pos_y, 0])
        x_end = np.array([x_axis_range[1], x_axis_pos_y, 0])

        y_axis_pos_x = -3.5
        y_axis_range = (4, -3)
        y_start = np.array([y_axis_pos_x, y_axis_range[0], 0])
        y_end = np.array([y_axis_pos_x, y_axis_range[1], 0])

        self.x_rate, self.y_rate = .5, .3

        self.x_axis = Arrow(x_start, x_end, color=GREEN)
        self.y_axis = Arrow(y_start, y_end, color=GREEN)
        self.add(self.x_axis, self.y_axis)

        axis_label_x = MathTex(
            r"i \sin(\theta) / \theta").next_to(self.x_axis, RIGHT)
        axis_label_y = MathTex(
            r"\cos(\theta) / \theta").next_to(self.y_axis, DOWN)
        self.add(axis_label_x, axis_label_y)

        # circle
        self.circle_radius = 1
        self.origin = np.array([y_axis_pos_x, x_axis_pos_y, 0])
        self.circle = Circle(radius=self.circle_radius, color=BLUE)
        self.circle.move_to(self.origin)
        self.add(self.circle)

        # orbit dot
        self.orbit_dot = Dot(radius=.05, color=YELLOW)
        self.orbit_dot.move_to(self.circle.point_from_proportion(0))
        # self.add(self.orbit_dot)

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
            return Line(pos_line_start, pos_line_end, color=PURPLE)

        self.orbit_dot_to_x_axis_line = always_redraw(
            get_orbit_dot_to_x_axis_line)

        # orbit_dot to y axis line
        def get_orbit_dot_to_y_axis_line():
            pos_line_start = self.orbit_dot.get_center()
            pos_line_end = np.array(
                [y_axis_pos_x, self.orbit_dot.get_center()[1], 0])
            return Line(pos_line_start, pos_line_end, color=PURPLE)

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
            return Line(pos_line_start, pos_line_end, color=PURPLE)

        self.orbit_dot_to_y_curve_line = always_redraw(
            get_orbit_dot_to_y_curve_line)

        # orbit_dot to y axis line
        def get_orbit_dot_to_x_curve_line():
            pos_line_start = self.orbit_dot.get_center()
            pos_line_end = np.array(
                [self.x_curve_dot.get_center()[0], self.orbit_dot.get_center()[1], 0])
            return Line(pos_line_start, pos_line_end, color=MAROON)

        self.orbit_dot_to_x_curve_line = always_redraw(
            get_orbit_dot_to_x_curve_line)

        self.equation = MathTex(
            r"e^{i \theta} = \cos(\theta) + i \sin(\theta)").move_to([0, 0, 0])
        
        # ticks
        x_labels = [
            MathTex("\pi"), MathTex("2 \pi"),
            MathTex("3 \pi"), MathTex("4 \pi"), MathTex("5 \pi")
        ]
        y_labels = [
            MathTex("\pi"), MathTex("2 \pi"),
            MathTex("3 \pi"), MathTex("4 \pi"), MathTex("5 \pi")
        ]

        for i, label in enumerate(x_labels):
            label.move_to(self.origin).shift(1.25*UP)
            label.shift(self.x_rate*PI * (i+1) * RIGHT)
            self.add(label)

        for i, label in enumerate(y_labels):
            label.move_to(self.origin).shift(1.25*LEFT)
            label.shift(self.y_rate*PI * (i+1) * DOWN)
            self.add(label)
        
        # theta value
        self.theta_label = MathTex(r"\theta = ").next_to(self.equation, DOWN)
        self.add(self.theta_label)

        def get_theta():
            return DecimalNumber(self.t, num_decimal_places=2).next_to(self.theta_label, RIGHT)

        self.theta = always_redraw(get_theta)

        self.add(self.orbit_dot)
        self.add(self.arrow)
        self.add(self.x_curve_dot)
        self.add(self.x_curve_line)
        self.add(self.y_curve_dot)
        self.add(self.y_curve_line)
        # self.add(self.orbit_dot_to_x_axis_line)
        # self.add(self.orbit_dot_to_y_axis_line)
        self.add(self.origin_to_x_axis_arrow)
        self.add(self.origin_to_y_axis_arrow)
        self.add(self.orbit_dot_to_y_curve_line)
        self.add(self.orbit_dot_to_x_curve_line)
        self.add(self.equation)
        self.add(self.theta)

        self.wait(2*PI * 2.5)
