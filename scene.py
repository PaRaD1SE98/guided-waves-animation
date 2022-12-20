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
