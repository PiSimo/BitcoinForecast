# BitcoinForecast 

Predict bitcoin value for the next 9minutes, with Recurrental Neural Network GRU.
<br />
<h1>Requirements:</h1>
<br/>

<ul>
<li>Python3</li>
<li><a href="http://keras.io/">Keras 2</a></li>
<li><a href="http://www.numpy.org/">numpy</a></li>
<li><a href="http://matplotlib.org/">MatploitLib</a></li>
</ul>
<br />
<h1>Model:</h1>
<div style="margin-left:auto;margin-right:auto;"><img width=550 height=800 src="https://cloud.githubusercontent.com/assets/17238972/25303841/4945f448-275b-11e7-8ad9-e4c9601a7d3a.png"></div>
<p>Generated with <a href="https://pisimo.github.io/DeepChart/">deepchart</a></p>
<h1>Instructions</h1>
<p>Clone the repo:</p>
<code>
git clone https://github.com/PiSimo/BitcoinForecast.git
</code>
<br />
<br />
<b>Training on new data:</b><br />
<code>python3 network.py -train <i>dataset_path</i> -iterations <i>number_of_training_iterations</i></code>
<p>To finetune the new model with an old one just add <code>-finetune <i>base_model_path</i></code> to the line above.</p>
<p>At the end of the training you will have an updated model.h5 with the new weights and you will see a plot with the test results.</p>
<br />
<b>Running:</b><br />
<code>python3 network.py -run <i>dataset_path</i> -model <i>model_path</i></code>
<p>The dataset is also required when you run, to perform normalization.</p>
<p>To visualize a plot with the real and predicted results enter Crtl-C and type no ,the program will create chart.png with the results.</p>

<br/>
Working example with <a href="https://github.com/PiSimo/BitcoinForecast/blob/master/model.h5">this model</a>:
<br/>
(Red:Predicted,Green:Real values)
<img src="https://cloud.githubusercontent.com/assets/17238972/24326997/630cf3c2-11bc-11e7-8edb-07be895e16ea.png" />

