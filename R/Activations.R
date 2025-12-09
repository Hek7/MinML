#' @title Sigmoid Activation Function
#' @description Computes the Sigmoid function: \eqn{f(x) = 1 / (1 + exp(-x))}.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the sigmoid function as a matrix.
#' @export
activation_sigmoid <- function(x) {
  x <- as.matrix(x)
  as.matrix(1 / (1 + exp(-x)))
}

#' @title Sigmoid Backward Pass (Derivative)
#' @description Computes the derivative of the Sigmoid function: \eqn{f'(x) = f(x) * (1 - f(x))}.
#'
#' @param output Output of the forward pass.
#' @param grad_output Gradient of the loss w.r.t the output.
#' @return Gradient w.r.t the input, as a matrix.
#' @export
activation_sigmoid_grad <- function(output, grad_output) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  as.matrix(grad_output * output * (1 - output))
}

#' @title Hyperbolic Tangent (Tanh) Activation Function
#' @description Computes the Tanh function: \eqn{f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))}.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the Tanh function as a matrix.
#' @export
activation_tanh <- function(x) {
  x <- as.matrix(x)
  as.matrix(tanh(x))
}

#' @title Tanh Backward Pass (Derivative)
#' @description Computes the derivative of the Tanh function: \eqn{f'(x) = 1 - f(x)^2}.
#'
#' @param output Output of the forward pass.
#' @param grad_output Gradient of the loss w.r.t the output.
#' @return Gradient w.r.t the input, as a matrix.
#' @export
activation_tanh_grad <- function(output, grad_output) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  as.matrix(grad_output * (1 - output^2))
}

#' @title ReLU Activation Function
#' @description Computes the ReLU function: \eqn{f(x) = max(0, x)}.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the ReLU function as a matrix.
#' @export
activation_relu <- function(x) {
  x <- as.matrix(x)
  as.matrix(x * (x > 0))
}

#' @title ReLU Backward Pass (Derivative)
#' @description Computes the derivative of ReLU: \eqn{f'(x) = 1} if x > 0, 0 otherwise.
#'
#' @param output Output of the forward pass.
#' @param grad_output Gradient of the loss w.r.t the output.
#' @return Gradient w.r.t the input, as a matrix.
#' @export
activation_relu_grad <- function(output, grad_output) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  as.matrix(grad_output * (output > 0))
}

#' @title Leaky ReLU Activation Function
#' @description Computes Leaky ReLU: \eqn{f(x) = x} if x > 0, \eqn{\alpha x} otherwise.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @param alpha Slope for negative inputs (default 0.01).
#' @return The result of the Leaky ReLU as a matrix.
#' @export
activation_leaky_relu <- function(x, alpha = 0.01) {
  x <- as.matrix(x)
  as.matrix(ifelse(x > 0, x, alpha * x))
}

#' @title Leaky ReLU Backward Pass (Derivative)
#' @description Computes derivative of Leaky ReLU: \eqn{f'(x) = 1} if x > 0, \eqn{\alpha} otherwise.
#'
#' @param output Output of the forward pass.
#' @param grad_output Gradient of the loss w.r.t the output.
#' @param alpha Slope for negative inputs (default 0.01).
#' @return Gradient w.r.t the input, as a matrix.
#' @export
activation_leaky_relu_grad <- function(output, grad_output, alpha = 0.01) {
  output <- as.matrix(output)
  grad_output <- as.matrix(grad_output)
  as.matrix(grad_output * ifelse(output > 0, 1, alpha))
}

#' @title GeLU Activation Function
#' @description Computes GeLU: \eqn{f(x) = x * \Phi(x)}, where \eqn{\Phi(x)} is the CDF of standard normal.
#'
#' @param x A numeric vector or matrix of pre-activation inputs.
#' @return The result of the GeLU function as a matrix.
#' @export
activation_gelu <- function(x) {
  x <- as.matrix(x)
  as.matrix(x * pnorm(x))
}

#' @title GeLU Backward Pass (Derivative)
#' @description Computes derivative of GeLU: \eqn{f'(x) = \Phi(x) + x * \phi(x)}, where \eqn{\phi(x)} is the PDF of standard normal.
#'
#' @param x Input to the forward pass.
#' @param grad_output Gradient of the loss w.r.t the output.
#' @return Gradient w.r.t the input, as a matrix.
#' @export
activation_gelu_grad <- function(x, grad_output) {
  x <- as.matrix(x)
  grad_output <- as.matrix(grad_output)
  derivative <- pnorm(x) + x * dnorm(x)
  as.matrix(grad_output * derivative)
}


