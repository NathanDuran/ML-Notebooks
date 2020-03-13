import plotly.graph_objects as go
from ipywidgets import widgets
import numpy as np
import pandas as pd


class BinaryPerceptronGraph:
    def __init__(self):
        self.data = pd.DataFrame({'x1': [0, 0, 1, 1], 'x2': [0, 1, 0, 1],
                                  'AND': [0, 0, 0, 1],  'OR': [0, 1, 1, 1], 'XOR': [0, 1, 1, 0]})
        self.model = {'w1': 0.0, 'w2': 1.0, 'bw': -0.5}
        self.learning_rate = None
        self.train_function = None
        self.total_steps = 0
        self.train_step = 0
        self.epoch = 1
        self.epoch_error = 0
        self.use_contour = False

        # Create the function toggle
        self.logic_func_toggle = widgets.ToggleButtons(options=['AND', 'OR', 'XOR'], value='AND',
                                                       description='Function:')

        sldr_ranges = [-1, 1]
        # Create weight slider controls
        self.weight1_sldr = widgets.FloatSlider(value=self.model['w1'], min=sldr_ranges[0], max=sldr_ranges[1], step=0.01,
                                                description='Weight 1:',
                                                continuous_update=True, orientation='horizontal', readout=True,
                                                readout_format='.1f')
        self.weight2_sldr = widgets.FloatSlider(value=self.model['w2'], min=sldr_ranges[0], max=sldr_ranges[1], step=0.01,
                                                description='Weight 2:',
                                                continuous_update=True, orientation='horizontal', readout=True,
                                                readout_format='.1f')
        self.bias_sldr = widgets.FloatSlider(value=self.model['bw'], min=sldr_ranges[0], max=sldr_ranges[1], step=0.01,
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
                                            symbol=['circle' if self.data.iloc[i]['AND'] else 'x' for i in range(len(self.data))],
                                            color=['green' if self.data.iloc[i]['AND'] else 'red' for i in range(len(self.data))],
                                            size=10))

        self.line_plot = go.Scatter(x=np.linspace(-10, 10, num=10), y=[0.5] * 10,
                                    showlegend=False,
                                    visible=True,
                                    mode='lines',
                                    line=dict(color='blue', width=3))

        self.contour_plot = go.Contour(
            z=self.generate_decision_boundary(np.array(self.data[['x1', 'x2']].values.tolist()), lambda x: self.predict(x)),
            x=np.linspace(-0.5, 1.5, 200), y=np.linspace(-0.5, 1.5, 200),
            transpose=False,
            showscale=False,
            colorscale=[[0, 'red'], [1, 'green']],
            line_smoothing=1.0,
            opacity=0.25)

        # Create a figure from the data plots and set some layout variables
        figure_data = [self.function_plot, self.line_plot]
        if self.use_contour:
            figure_data += self.contour_plot
        self.figure = go.Figure(data=figure_data,
                                layout=go.Layout(margin={'t': 20, 'b': 0, 'l': 0}))
        self.figure.update_xaxes(range=[-0.75, 1.75], tick0=0.0, dtick=0.5)
        self.figure.update_yaxes(range=[-0.75, 1.75], tick0=0.0, dtick=0.5)
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
        bw = float(self.bias_sldr.value)
        self.model['w1'], self.model['w2'], self.model['bw'] = w1, w2, bw

        # Update the decision boundary
        self.update_line_plot()
        if self.use_contour:
            self.update_contour_plot()

    def step_btn_press(self, change):
        self.run_train_step()

    def epoch_btn_press(self, change):
        for i in range(len(self.data)):
            self.run_train_step()

    def update_line_plot(self):
        # Generate the x and y ranges
        x_range = np.linspace(-10, 10, num=10)
        w2 = self.model['w2'] if self.model['w2'] != 0.0 else 1.0  # Little cheat to prevent divide by 0 error
        y_range = [((-self.model['w1'] / w2) * x) + (-self.model['bw'] / w2) for x in x_range]

        # Update the line
        with self.graph.batch_update():
            self.graph.data[1].x = x_range
            self.graph.data[1].y = y_range

    def update_contour_plot(self):
        # Generate the new decision boundary values for contour plot
        z = self.generate_decision_boundary(np.array(self.data[['x1', 'x2']].values.tolist()), lambda x: self.predict(x))

        # Update the plot
        with self.graph.batch_update():
            self.graph.data[2].z = z

    def run_train_step(self):
        inputs = self.data[['x1', 'x2']].values.tolist()
        self.train_function(inputs, self.data[self.logic_func_toggle.value], self.train_step, self.train_step + 1)
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

        # Save the model
        self.model['w1'], self.model['w2'], self.model['bw'] = w1, w2, bw

        # Update the decision boundary
        self.update_line_plot()
        if self.use_contour:
            self.update_contour_plot()

    def predict(self, x):
        predictions = []
        # Loop over all of the input examples
        for i in range(len(x)):
            # Calculate output
            weight_sum = (x[i][0] * self.model['w1']) + (x[i][1] * self.model['w2']) + (1 * self.model['bw'])
            # Activation (step) function
            if weight_sum > 0:
                activation = 1
            else:
                activation = 0
            predictions.append(activation)
        return np.array(predictions)

    def generate_decision_boundary(self, x, pred_func):
        # Set min and max values and give it some padding
        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        h = 0.01

        # Generate a grid of points with distance h between them
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Predict the function value for the whole grid
        z = pred_func(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        return z

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


