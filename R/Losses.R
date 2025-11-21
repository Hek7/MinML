# Losses

#' @title Squared Loss (MSE)
#' @description Calculates the Mean Squared Error (MSE) between true values (y_true)
#' and predicted values (y_pred).
#' @param y_true A numeric vector of the true values.
#' @param y_pred A numeric vector of the predicted values.
#' @return The calculated MSE as a single numeric value.
#' @examples

squared_loss <- function(y_true, y_pred) {
  # same length check
  if (length(y_true) != length(y_pred)) {
    stop("y_true and y_pred must have the same length.")
  }

  errors <- y_pred - y_true
  squared_errors <- errors^2
  mse <- mean(squared_errors)

  return(mse)
}

#' @title Squared Loss (MSE) Gradient
#' @description Calculates the gradient of the MSE loss with respect to the prediction.
#' @param y_true A numeric vector of the true values.
#' @param y_pred A numeric vector of the predicted values.
#' @return The gradient matrix/vector.
squared_loss_grad <- function(y_true, y_pred) {
  N <- nrow(y_pred)

  grad <- (2 / N) * (y_pred - y_true)

  return(grad)
}
