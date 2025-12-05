
#' @title SGD Parameter Update Step
#' @description Applies the SGD update rule (theta_new = theta_old - lr * grad)
#' to a list of parameters using a corresponding list of gradients.
#'
#' @param current_params A named list of the current model parameters.
#' @param gradients A named list of the calculated gradients.
#' @param learning_rate The step size (alpha).
#' @return A new list of updated parameters.
sgd_update_params <- function(current_params, gradients, learning_rate) {

  if (learning_rate <= 0) {
    stop("Learning rate must be positive.")
  }

  if (!identical(names(current_params), names(gradients))) {
    stop("Parameter and Gradient lists must have identical names.")
  }

  updated_params <- list()

  for (name in names(current_params)) {
    param <- current_params[[name]]
    grad <- gradients[[name]]

    # Check for dimension
    if (!identical(dim(param), dim(grad))) {
      stop(sprintf("Dimension mismatch for parameter '%s'. Shapes must match.", name))
    }

    # Update
    updated_params[[name]] <- param - learning_rate * grad
  }

  return(updated_params)
}
