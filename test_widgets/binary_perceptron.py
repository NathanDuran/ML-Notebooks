import plotly.graph_objects as go
from ipywidgets import widgets
import numpy as np

num_values = 10
number_ranges = np.linspace(-10, 10, num=num_values)


def generate_contour_points(x_range, y_range, w1=0.0, w2=1.0, bw=-0.5):
    a = (-w1 / w2)
    b = 1
    c = (-bw / w2)
    points = []
    for x in x_range:
        row = []
        for y in y_range:
            sum = (a * x) + (b * y) + c  # Do absolute for non-negative distances
            sqrs = np.sqrt(np.square(a) + np.square(b))
            dist = sum / sqrs
            row.append(dist)
        points.append(row)
    return points


class BinaryPerceptronGraph:
    def __init__(self, data):
        self.data = data
        self.learning_rate = None
        self.train_function = None
        self.total_steps = 0
        self.train_step = 0
        self.epoch = 1
        self.epoch_error = 0

        # Create the function toggle
        self.logic_func_toggle = widgets.ToggleButtons(options=['AND', 'OR', 'XOR'], value='AND',
                                                       description='Function:')

        sldr_ranges = [-1, 1]
        # Create weight slider controls
        self.weight1_sldr = widgets.FloatSlider(value=0.0, min=sldr_ranges[0], max=sldr_ranges[1], step=0.01,
                                                description='Weight 1:',
                                                continuous_update=True, orientation='horizontal', readout=True,
                                                readout_format='.1f')
        self.weight2_sldr = widgets.FloatSlider(value=0.0, min=sldr_ranges[0], max=sldr_ranges[1], step=0.01,
                                                description='Weight 2:',
                                                continuous_update=True, orientation='horizontal', readout=True,
                                                readout_format='.1f')
        self.bias_sldr = widgets.FloatSlider(value=0.5, min=sldr_ranges[0], max=sldr_ranges[1], step=0.01,
                                             description='Bias:',
                                             continuous_update=True, orientation='horizontal', readout=True,
                                             readout_format='.1f')

        # Create train step and epoch buttons
        self.step_btn = widgets.Button(description='Step',
                                       disabled=False,
                                       button_style='success',
                                       tooltip='Training step on single instance of data',
                                       icon='step-forward')

        self.epoch_btn = widgets.Button(description='Epoch',
                                        disabled=False,
                                        button_style='info',
                                        tooltip='Training one epoch on all instances of data',
                                        icon='play')

        # Create output text boxes
        self.output_txt = widgets.HTMLMath(value='')

        # Create scatter and line plots
        self.function_plot = go.Scatter(x=self.data['x1'], y=self.data['x2'],
                                        name='',
                                        showlegend=False,
                                        visible=True,
                                        mode='markers',
                                        marker=dict(
                                            symbol=['circle' if self.data.iloc[i]['AND'] else 'x' for i in
                                                    range(len(data))],
                                            color=['green' if self.data.iloc[i]['AND'] else 'red' for i in
                                                   range(len(data))],
                                            size=10))

        self.line_plot = go.Scatter(x=number_ranges, y=[0.5] * num_values,
                                    showlegend=False,
                                    visible=True,
                                    mode='lines',
                                    line=dict(color='blue', width=3))

        contour_colours = [[0, 'red'],  [0.4, 'white'], [0.6, 'white'], [1, 'green']]
        self.contour_plot = go.Contour(
            z=generate_contour_points(number_ranges, number_ranges),
            x=number_ranges, y=[i + 1 for i in number_ranges], # TODO +1 because thats what w2 is getting set to?
            transpose=True,
            showscale=True,
            colorscale=contour_colours,
            # contours_coloring='heatmap',
            contours=dict(start=-10, end=10, size=2),
            opacity=0.75)

        # Create a figure from the data plots and set some layout variables
        self.figure = go.Figure(data=[self.function_plot, self.line_plot, self.contour_plot],
                                layout=go.Layout(margin={'t': 20, 'b': 0, 'l': 0}))
        self.figure.update_xaxes(range=[-0.5, 1.5], tick0=0.0, dtick=0.5)
        self.figure.update_yaxes(range=[-0.5, 1.5], tick0=0.0, dtick=0.5)
        self.graph = go.FigureWidget(self.figure)

    # Define some functions to update the plot from inputs
    def logic_func_toggle_change(self, change):
        # Get the new function name (AND, OR, XOR)
        function = self.logic_func_toggle.value
        # Update the plot shapes and colours
        with self.graph.batch_update():
            self.graph.data[0].marker.symbol = ['circle' if self.data.iloc[i][function] else 'x' for i in
                                                range(len(self.data))]
            self.graph.data[0].marker.color = ['green' if self.data.iloc[i][function] else 'red' for i in
                                               range(len(self.data))]

    def weight_sldr_change(self, change):
        # Get the slider values
        w1 = float(self.weight1_sldr.value)
        w2 = float(self.weight2_sldr.value)
        b = float(self.bias_sldr.value)
        # Update the line plot
        self.update_line_plot(w1, w2, b)

    def step_btn_press(self, change):
        self.run_train_step()

    def epoch_btn_press(self, change):
        for i in range(len(self.data)):
            self.run_train_step()

    def update_line_plot(self, w1, w2, b):
        # Generate the x and y ranges
        x_range = number_ranges
        w2 = w2 if w2 != 0.0 else 1.0  # Little cheat to prevent divide by 0 error
        y_range = [((-w1 / w2) * x) + (-b / w2) for x in x_range]

        # Update the line
        with self.graph.batch_update():
            self.graph.data[1].x = x_range
            self.graph.data[1].y = y_range

    def run_train_step(self):
        self.train_function(self.data[self.logic_func_toggle.value], self.train_step, self.train_step + 1)
        self.train_step += 1
        self.total_steps += 1
        if self.train_step == len(self.data):
            self.train_step = 0
            self.epoch += 1
            self.epoch_error = 0

    def update_step(self, model, w1, w2, bw, error):
        num_decimals = 6

        # Update forward pass string
        x1 = self.data['x1'][self.train_step]
        x2 = self.data['x2'][self.train_step]

        weight_sum = (x1 * model['weight_1']) + (x2 * model['weight_2']) + model['bias_weight']
        sum_str = '$$sum:(' + str(x1) + ' \\times ' + str(round(model['weight_1'], num_decimals)) + \
                  ') + (' + str(x2) + ' \\times ' + str(round(model['weight_2'], num_decimals)) + \
                  ') + (1 \\times ' + str(round(model['bias_weight'], num_decimals)) + \
                  ') = ' + str(round(weight_sum, num_decimals)) + '$$'

        activation = 1 if weight_sum > 0 else 0
        act_str = '> 0 \\therefore ' if weight_sum > 0 else '\\leq 0 \\therefore '
        activation_str = '$$activation:' + str(round(weight_sum, num_decimals)) + act_str + str(activation) + '$$'

        # Update error string and epoch error
        target = self.data[self.logic_func_toggle.value][self.train_step]
        error_str = '$$error:' + str(target) + "-" + str(activation) + '=' + str(abs(error)) + '$$'
        self.epoch_error += abs(error)

        # Update weights strings
        w1_str = '$$w1=' + str(round(model['weight_1'], num_decimals)) + '+' + str(error) + \
                 ' \\times ' + str(x1) + ' \\times ' + str(self.learning_rate) + '=' + \
                 str(round(w1, num_decimals)) + '$$'
        w2_str = '$$w2=' + str(round(model['weight_2'], num_decimals)) + '+' + str(error) + \
                 ' \\times ' + str(x2) + ' \\times ' + str(self.learning_rate) + '=' + \
                 str(round(w2, num_decimals)) + '$$'
        bw_str = '$$bw=' + str(round(model['bias_weight'], num_decimals)) + '+' + str(error) + \
                 ' \\times 1 \\times ' + str(self.learning_rate) + '=' + \
                 str(round(bw, num_decimals)) + '$$'

        epoch_error = self.epoch_error / self.train_step if self.train_step != 0 else 0
        # Update the output text
        state_str = '<b>Step: ' + str(self.total_steps + 1) + '&emsp;Epoch: ' + str(self.epoch) + \
                    '&emsp;Epoch Error: ' + str(round(epoch_error, num_decimals)) + \
                    '&emsp;Inputs: [' + str(x1) + ', ' + str(x2) + ']' + '&emsp;' + \
                    'Expected output: ' + str(target) + '</b><br><br>'
        self.output_txt.value = state_str + sum_str + '<br>' + activation_str + '<br>' + error_str + '<br>' + \
                                w1_str + w2_str + bw_str

        # Update the decision boundary
        self.update_line_plot(w1, w2, bw)

    def create_decision_boundary_graph(self):
        # Attach listeners to the controls
        self.logic_func_toggle.observe(self.logic_func_toggle_change, names='value')
        self.weight1_sldr.observe(self.weight_sldr_change, names='value')
        self.weight2_sldr.observe(self.weight_sldr_change, names='value')
        self.bias_sldr.observe(self.weight_sldr_change, names='value')

        sliders = widgets.VBox([self.weight1_sldr, self.weight2_sldr, self.bias_sldr])
        controls = widgets.HBox([sliders, self.logic_func_toggle])
        widget = widgets.VBox([controls, self.graph])
        return widget

    def create_train_graph(self, train_function, learning_rate):
        # Set the train function and learning rate
        self.train_function = train_function
        self.learning_rate = learning_rate

        # Attach listeners to the controls
        self.logic_func_toggle.observe(self.logic_func_toggle_change, names='value')
        self.step_btn.on_click(self.step_btn_press)
        self.epoch_btn.on_click(self.epoch_btn_press)

        buttons = widgets.VBox([self.step_btn, self.epoch_btn])
        controls = widgets.HBox([buttons, self.logic_func_toggle])
        widget = widgets.VBox([controls, self.graph, self.output_txt])
        return widget
