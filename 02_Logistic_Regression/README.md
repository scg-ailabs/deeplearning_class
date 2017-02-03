## Logistic Regression ##
### Binary classification : output y is binary variable $\to$Bernoulli distribution over y conditioned on $x$ ###
$$
P(Y|X) = p(x;\theta)^y(1-p(x;\theta))^{1-y}
$$
#### Let $ p(x) $ be a linear function of $x$####
- $p(x)=W^Tx+b\quad \quad \quad \quad$ : unbounded
- $p(x) = max\{0, min\{1, W^Tx+b\}\}$: cannot see "diminishing returns"
- It is hard to calculate gradient!!

#### Let $\log{p(x)}$ be a linear function of $x$ ####
- $\log{p(x)} = W^Tx+b\quad \quad \quad$
- solving for p, this gives
$$
p(x;w, b)=e^{W^Tx+b}
$$
- unbounded in (+) direction
- ${p(x)} = min\{1, e^{W^Tx+b}\}$
- It is still hard to calculate gradient!!

#### Let $\log{\frac{p}{1-p}}$ be a linear function of $x$####
- $\log{\frac{p(x)}{1-p(x)}} = W^Tx+b\quad \quad$ : bounded
- solving for p, this gives
$$
p(x;w,b) = \frac{e^{W^Tx+b}}{1+e^{W^Tx+b}}=\frac{1}{1+e^{-(W^Tx+b)}}
$$
- The decision boundary separating the two predicted classes is the solution of $W^Tx+b=0$, which is a point  if $x$ is one dimensional, a line  if it is two dimensional, etc.
- $p(x)$ is called logistic function, which is a type of sigmoid function.

#### Likelihood Function for Logistic Regression ####
$$
L(w, b)=\prod^n_{i=1}p(x_i)^{y_i}(1-p(x_i))^{1-y_i}
$$
The log-likelihood turns products into sums:
$$
\ell(w, b)=\sum^n_{i=1}\{y_i\log p(x_i)+(1-y_i)\log (1-p(x_i))\}\\
\quad\quad\quad= \sum^n_{i=1}\log(1-p(x_i))+\sum^n_{i=1}y_i\log\frac{p(x_i)}{1-p(x_i)}\\
\quad\quad= \sum^n_{i=1}\log(1-p(x_i)) + \sum^n_{i=1}y_i(W^Tx+b)\\
\quad\quad\quad= \sum^n_{i=1}-log(1+e^{W^Tx+b}) + \sum^n_{i=1}y_i(W^Tx+b)
$$

#### Cost Function ( minimize negative log likelihood ) ####
$$
J(\theta) = -\frac{1}{n}\sum_{i=1}^{n} [y_i\log(h_\theta(x_{i})) + (1-y_{i})\log(1-h_\theta(x_{i}))]
\\
\frac{\partial}{\partial\theta_j}J(\theta)=\frac{1}{n}\sum^n_{i=1}(h_\theta (x_i) - y_i)x_{ij}
$$
$where, h_\theta(x) = \frac{1}{1+e^{-(W^Tx+b)}} $ : logistic function