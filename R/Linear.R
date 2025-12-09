#' @title Xavier Initialization
#' @description Initializes weights using the Xavier uniform distribution.
#'
#' @param input_features The number of input connections (n_in).
#' @param output_features The number of output connections (n_out).
#' @return A matrix of initialized weights.
#' @export
initialize_xavier <- function(input_features, output_features) {
  # Calculate the range for the uniform distribution:
  # r = sqrt(6 / (n_in + n_out))
  limit <- sqrt(6 / (input_features + output_features))

  # Generate a uniform random matrix between -limit and +limit
  # Using runif to generate random numbers from a uniform distribution.
  random_weights <- runif(
    n = input_features * output_features,
    min = -limit,
    max = limit
  )

  # Reshape into the required matrix
  matrix(
    random_weights,
    nrow = input_features,
    ncol = output_features
  )
}


# Linear Module

#' @title Linear Layer Module
#' @description Creates a fully connected layer module.
#' @param input_features The size of the input vector.
#' @param output_features The size of the output vector
#' @return A list representing the Linear Module with initialized weights and biases.
#' @export
Linear <- function(input_features, output_features) {

  # init weights Xavier
  weights <- initialize_xavier(input_features, output_features)

  # init biases
  biases <- rep(0, output_features)

  # Internal container
  # We use this list to hold the state that 'forward' and 'backward' will mutate.
  module <- list(
    params = list(
      W = weights,
      B = biases
    ),
    grads = list(
      W = NULL,
      B = NULL
    ),
    cache = list(
      X = NULL
    )
  )

  result <- list(
    # Forward Pass Method
    forward = function(x) {
      x <- as.matrix(x)
      module$cache$X <<- x

      output <- x %*% module$params$W
      output <- sweep(output, 2, module$params$B, FUN = "+")

      return(as.matrix(output))
    },

    # Backward pass
    backward = function(grad_output) {
      X <- as.matrix(module$cache$X)
      grad_output <- as.matrix(grad_output)

      module$grads$W <<- as.matrix(t(X)) %*% grad_output
      module$grads$B <<- colSums(grad_output)

      grad_input <- grad_output %*% t(module$params$W)

      return(as.matrix(grad_input))
    },

    get_params = function() {
      return(module$params)
    },

    get_grads = function() {
      return(module$grads)
    },

    set_params = function(new_params) {
      for (pname in names(new_params)) {
        module$params[[pname]] <<- new_params[[pname]]
      }
    }
  )

  return(result)
}
