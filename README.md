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
<img width=300 height=700 src="https://cloud.githubusercontent.com/assets/17238972/25045757/1fb9e1c6-212e-11e7-80db-acb4665d4dbb.png">
<h1>Instructions</h1>
<p>Clone the repo:</p>
<code>
git clone https://github.com/PiSimo/BitcoinForecast.git
</code>
<br />
<br />
<b>Training on new data:</b>
<code>python3 network.py train</code>
<p>Enter the path to your dataset (you can create an updated one with grabber.py).</p>
<b>Fine tune:</b>
<p>If you want to fine-tune on your new data (should be done to increase accuracy) type yes when it asks to fine-tune.
Then insert the path to the trained model ('model.h5' is already in the folder).
</p>
<p>At the end of the training you will have an updated model.h5 with the new weights and you will see a plot with the test results.</p>
<br />
<b>Running:</b>
<code>python3 network.py run</code>
<p>Enter your dataset path the same you have used for training (needed for normalization and denormalization)</p>
<p>Enter your .h5 trained model, and then the main loop will start</p>
<p>To visualize a plot with the real and predicted results enter Crtl-C and type no ,the program will create chart.png with the results</p>

<br/>
Working example with <a href="https://github.com/PiSimo/BitcoinForecast/blob/master/model.h5">this model</a>:
<br/>
(Red:Predicted,Green:Real values)
<img src="https://cloud.githubusercontent.com/assets/17238972/24326997/630cf3c2-11bc-11e7-8edb-07be895e16ea.png" />

