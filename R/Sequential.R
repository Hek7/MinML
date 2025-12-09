
#' @title Sequential Module
#' @description Creates a sequential container module that stacks multiple layers
#' and activations together.
#'
#' @param ... A series of MinML modules.
#'        The modules must have defined 'forward' and 'backward' methods.
#' @return A list representing the Sequential Module.
#'
Sequential <- function(...) {
  modules <- list(...)

  # list not empty check
  if (length(modules) == 0) {
    stop("Sequential module must contain at least one layer or activation.")
  }

  module <- list(
    name = "Sequential",
    modules = modules,
    cache = list(), # Used to store intermediates of backward

    # Forward pass
    forward = function(x) {
      current_output <- x

      # Iterate through each module
      for (i in seq_along(module$modules)) {
        m <- module$modules[[i]]

        if ("forward" %in% names(m)) {
          current_output <- m$forward(current_output)

        } else if (is.function(m)) {
          current_output <- m(current_output)

        } else {
          stop(paste("Module at index", i, "is missing"))
        }

        # Store the output for use during the backward pass (for activations)
        # Note: layers handle this themselves in their backward functions
        module$cache[[i]] <- current_output
      }

      return(current_output)
    },

    # --- Backward Pass ---
    backward = function(grad_output) {
      current_grad <- grad_output

      # Iterate backward through the modules
      for (i in rev(seq_along(module$modules))) {
        m <- module$modules[[i]]

        # Check if the module is a Layer or an Activation
        if ("backward" %in% names(m)) {
          # Layer backward function returns grad_input and updates dW/dB internally.

          current_grad <- m$backward(current_grad)
        } else if (is.function(m)) {

          # grab cached output for the activation function
          activation_output <- module$cache[[i]]

          # Manually map activations based on the function names.

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

        } else {
          stop(paste("Module at index", i, "is missing a 'backward' method"))
        }
      }

      return(current_grad) # Returns the gradient w.r.t the initial input X
    }

    get_params = function() {
      all_params <- list()
      for (m in module$modules) {
        if ("params" %in% names(m)) {
          all_params <- c(all_params, list(m$params))
        }
      }
      return(all_params)
    }

    get_grads = function() {
      all_grads <- list()
      for (m in module$modules) {
        if ("grads" %in% names(m)) {
          all_grads <- c(all_grads, list(m$grads))
        }
      }
      return(all_grads)
    }




  )

  # Return the composite sequential module
  return(module)
}
