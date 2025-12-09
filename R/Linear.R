
#' @title Initialize Weights to Zeros
#' @description Initializes all weight matrix W elements to zero.
#'
#' @param input_features The number of input connections
#' @param output_features The number of output connections
#' @return A matrix of zeros.
#' @export
initialize_zeros <- function(input_features, output_features) {
  matrix(
    0,
    nrow = input_features,
    ncol = output_features
  )
}

# Linear Module

#' @title Linear Layer Module
#' @description Creates a fully connected layer module.
#'
#' @param input_features The size of the input vector.
#' @param output_features The size of the output vector
#' @return A list representing the Linear Module with initialized weights and biases.
#' @export
Linear <- function(input_features, output_features) {

  # init weights
  weights <- initialize_zeros(input_features, output_features)

  # init biases
  biases <- rep(0, output_features)

  module <- list(
    params = list(
      W = weights,
      B = biases
    ),

    grads = list(
      dW = NULL,
      dB = NULL
    ),
    # Storage for intermediate values for backward pass
    cache = list(
      X = NULL
    ),

    # --- Forward Pass Method ---
    forward = function(x) {
      # Save input X to cache for the backward pass
      module$cache$X <- x

      output <- x %*% module$params$W
      output <- sweep(output, 2, module$params$B, FUN = "+")

      return(output)
    },

    # Backward pass
    backward = function(grad_output) {
      # Retrieve cached input (X) from the forward pass
      X <- module$cache$X

      # Gradient for Weights (dW)
      module$grads$dW <- t(X) %*% grad_output

      # Gradient for Biases (dB)
      module$grads$dB <- colSums(grad_output)

      # Gradient for Input (dX)
      grad_input <- grad_output %*% t(module$params$W)

      # Clear the cache
      module$cache$X <- NULL

      return(grad_input)
    }
  )

  return(module)
}
