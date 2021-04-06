# Snake-pytorch
 Simple q deep learning implementation of snake game

<hr>

 Game made in pygame, ML agent in pytorch. NN work on Linear QNet with nodes: 15+256+256+256+3. On entry model gets information about 7 directions (front, left, right, front-left, front-right, back-left, back-right) if each of them is empty, in which quarter of snake head the apple is and 4-elements array of booleans in witch direction snake goes at this moment (example [1,0,0,0]).
Agent returns wector of propability ([forward, right, left]) in witch direction should go in next round - game take the highest value as an direction.
<hr>

<h6>Look of app:</h6>

![GUI example](/img/snake01.png)

<h6>Example death of snake:</h6>

![GUI example](/img/snake_death.png)

<h6>Learning statistics:</h6>

![GUI example](/img/runTest_model3_15_256X3_3_57.png)

<p><b>x</b> - numbers of games</p>
<p><b>y</b> - scores</p>
<p><b>Title</b> - training Q deep leaerning agent - snake like game</p>

<hr>
Best model from 15-256-256-256-256-3 Linear NN gets up to 71 points - but this record was too far from average, the 15+256+256+256+3 Linear NN works better for this example.

<hr>
Snake is not in the middle of the board so he turns more likely on the right than left, to awoid that the piont of snake start position should always be randomize between games.
<hr>
Snake has very limited vision, for better results and higher scores in future it should have at least 7 by 7 rectangles vision.
