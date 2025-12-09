#' @title Sigmoid Activation Function
#' @description Computes the Sigmoid function: $f(x) = \frac{1}{1 + e^{-x}}$.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the sigmoid function, guaranteed to be a matrix.
#' @export
activation_sigmoid <- function(x) {
  x <- as.matrix(x)
  # Ensure the output retains the matrix structure
  as.matrix(1 / (1 + exp(-x)))
}

#' @title Sigmoid Backward Pass (Derivative)
#' @description Computes the derivative of the Sigmoid function: $f'(x) = f(x) * (1 - f(x))$.
#'
#' @param output The output of the forward pass, $f(x)$.
#' @param grad_output The gradient of the loss with respect to the output.
#' @return The gradient with respect to the input, $x$, guaranteed to be a matrix.
#' @export
activation_sigmoid_grad <- function(output, grad_output) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  # Ensure the output retains the matrix structure
  as.matrix(grad_output * output * (1 - output))
}

#' @title Hyperbolic Tangent (Tanh) Activation Function
#' @description Computes the Tanh function: $f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the Tanh function, guaranteed to be a matrix.
#' @export
#'
activation_tanh <- function(x) {
  x <- as.matrix(x)
  # Ensure the output retains the matrix structure
  as.matrix(tanh(x)) # R has a built-in tanh function
}

#' @title Tanh Backward Pass (Derivative)
#' @description Computes the derivative of the Tanh function: $f'(x) = 1 - f(x)^2$.
#'
#' @param output The output of the forward pass, $f(x)$.
#' @param grad_output The gradient of the loss with respect to the output.
#' @return The gradient with respect to the input, $x$, guaranteed to be a matrix.
#' @export
activation_tanh_grad <- function(output, grad_output) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  # Ensure the output retains the matrix structure
  as.matrix(grad_output * (1 - output^2))
}

#' @title Rectified Linear Unit (ReLU) Activation Function
#' @description Computes the ReLU function: $f(x) = \max(0, x)$.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the ReLU function, guaranteed to be a matrix.
#' @export
#'
activation_relu <- function(x) {
  x <- as.matrix(x)
  # Ensure the output retains the matrix structure
  as.matrix(x * (x > 0)) # Element-wise max function
}

#' @title ReLU Backward Pass (Derivative)
#' @description Computes the derivative of the ReLU function: $f'(x) = 1$ if $x > 0$, $0$ otherwise.
#'
#' @param output The output of the forward pass, $f(x)$.
#' @param grad_output The gradient of the loss with respect to the output.
#' @return The gradient with respect to the input, $x$, guaranteed to be a matrix.
#' @export
activation_relu_grad <- function(output, grad_output) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  # The derivative is 1 where the output > 0 (input > 0), and 0 otherwise.
  # Ensure the output retains the matrix structure
  as.matrix(grad_output * (output > 0))
}

#' @title Leaky ReLU Activation Function
#' @description Computes the Leaky ReLU function: $f(x) = x$ if $x > 0$, $\alpha x$ otherwise.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @param alpha The slope for negative inputs (default: 0.01).
#' @return The result of the Leaky ReLU function, guaranteed to be a matrix.
#' @export
#'
activation_leaky_relu <- function(x, alpha = 0.01) {
  x <- as.matrix(x)
  # Ensure the output retains the matrix structure
  as.matrix(ifelse(x > 0, x, alpha * x))
}

#' @title Leaky ReLU Backward Pass (Derivative)
#' @description Computes the derivative of the Leaky ReLU function: $f'(x) = 1$ if $x > 0$, $\alpha$ otherwise.
#'
#' @param output The output of the forward pass.
#' @param grad_output The gradient of the loss with respect to the output.
#' @param alpha The slope for negative inputs (default: 0.01).
#' @return The gradient with respect to the input, $x$, guaranteed to be a matrix.
#' @export
activation_leaky_relu_grad <- function(output, grad_output, alpha = 0.01) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  # Ensure the output retains the matrix structure
  as.matrix(grad_output * ifelse(output > 0, 1, alpha))
}

#' @title Gaussian Error Linear Unit (GeLU) Activation Function
#' @description Computes the GeLU function: $f(x) = x \cdot \Phi(x)$, where $\Phi(x)$ is the
#' cumulative distribution function (CDF) of the standard normal distribution.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the GeLU function, guaranteed to be a matrix.
#' @export
#'
activation_gelu <- function(x) {
  x <- as.matrix(x)
  # R's pnorm is the CDF $\Phi(x)$
  # Ensure the output retains the matrix structure
  as.matrix(x * pnorm(x))
}

#' @title GeLU Backward Pass (Derivative)
#' @description Computes the derivative of the GeLU function: $f'(x) = \Phi(x) + x \cdot \phi(x)$,
#' where $\phi(x)$ is the probability density function (PDF) of the standard normal distribution.
#'
#' @param x The original input to the forward pass.
#' @param grad_output The gradient of the loss with respect to the output.
#' @return The gradient with respect to the input, $x$, guaranteed to be a matrix.
#' @export
activation_gelu_grad <- function(x, grad_output) {
  x <- as.matrix(x)
  grad_output <- as.matrix(grad_output)

  cdf <- pnorm(x)
  pdf <- dnorm(x)

  # f'(x) = $\Phi(x) + x \cdot \phi(x)$
  derivative <- cdf + x * pdf
  # Ensure the output retains the matrix structure
  as.matrix(grad_output * derivative)
}

