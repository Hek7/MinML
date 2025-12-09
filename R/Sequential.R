
#' @title Sequential Module
#' @description Creates a sequential container module that stacks multiple layers
#' and activations together.
#' @param ... A series of MinML modules.
#' @return A list representing the Sequential Module.
#' @export
Sequential <- function(...) {
  modules <- list(...)

  if (length(modules) == 0) {
    stop("Sequential module must contain at least one layer or activation.")
  }

  module <- list(
    name = "Sequential",
    modules = modules,
    cache = list(),

    # Forward pass
    forward = function(x) {
      current_output <- x

      for (i in seq_along(module$modules)) {
        m <- module$modules[[i]]

        if ("forward" %in% names(m)) {
          current_output <- m$forward(current_output)
        } else if (is.function(m)) {
          current_output <- m(current_output)
          current_output <- as.matrix(current_output)
        } else {
          stop(paste("Module at index", i, "is missing"))
        }

        module$cache[[i]] <<- current_output
      }
      return(current_output)
    },

    # Backward Pass
    backward = function(grad_output) {
      current_grad <- as.matrix(grad_output)

      for (i in rev(seq_along(module$modules))) {
        m <- module$modules[[i]]

        if ("backward" %in% names(m)) {
          current_grad <- m$backward(current_grad)
        } else if (is.function(m)) {
          # Activation handling
          activation_output <- module$cache[[i]]

          if (identical(m, activation_relu)) {
            current_grad <- activation_relu_grad(activation_output, current_grad)
          } else if (identical(m, activation_leaky_relu)) {
            current_grad <- activation_leaky_relu_grad(activation_output, current_grad)
          } else if (identical(m, activation_sigmoid)) {
            current_grad <- activation_sigmoid_grad(activation_output, current_grad)
          } else if (identical(m, activation_tanh)) {
            current_grad <- activation_tanh_grad(activation_output, current_grad)
          } else if (identical(m, activation_gelu)) {
            current_grad <- activation_gelu_grad(activation_output, current_grad)
          } else {
            stop(paste("Unsupported activation function found at index", i))
          }
          current_grad <- as.matrix(current_grad)
        } else {
          stop(paste("Module at index", i, "is missing a 'backward' method"))
        }
      }
      return(current_grad)
    },

    # --- UPDATED GETTERS ---

    get_params = function() {
      all_params <- list()
      for (i in seq_along(module$modules)) {
        m <- module$modules[[i]]
        # Check for the accessor method
        if ("get_params" %in% names(m)) {
          params <- m$get_params()

          # Fix Name Collisions: Prefix parameters with layer index
          # e.g., "layer1.W", "layer1.B"
          names(params) <- paste0("layer", i, ".", names(params))

          all_params <- c(all_params, params)
        }
      }
      return(all_params)
    },

    get_grads = function() {
      all_grads <- list()
      for (i in seq_along(module$modules)) {
        m <- module$modules[[i]]
        # Check for the accessor method
        if ("get_grads" %in% names(m)) {
          grads <- m$get_grads()

          # Apply same naming convention to grads so optimizer matches them
          names(grads) <- paste0("layer", i, ".", names(grads))

          all_grads <- c(all_grads, grads)
        }
      }
      return(all_grads)
    }
  )

  return(module)
}
