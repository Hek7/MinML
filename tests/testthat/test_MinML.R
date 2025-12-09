library(testthat)

context("MinML Tests")

set.seed(123)

X <- matrix(rnorm(20), nrow = 4, ncol = 5)
y <- matrix(rnorm(4), nrow = 4, ncol = 1)
learning_rate <- 0.01

test_that("Linear layer forward/backward works", {
  layer <- Linear(input_features = 5, output_features = 3)

  # Forward pass
  out <- layer$forward(X)
  expect_equal(dim(out), c(4,3))

  # Backward pass
  grad_out <- matrix(1, nrow=4, ncol=3)
  grad_input <- layer$backward(grad_out)
  expect_equal(dim(grad_input), dim(X))

  # Check gradients exist
  grads <- layer$get_grads()
  expect_true(all(c("W","B") %in% names(grads)))
  expect_equal(dim(grads$W), c(5,3))
  expect_equal(length(grads$B), 3)
})

# Activations
activations <- list(
  relu = activation_relu,
  sigmoid = activation_sigmoid,
  tanh = activation_tanh,
  leaky_relu = function(x) activation_leaky_relu(x, alpha=0.01),
  gelu = activation_gelu
)

grad_functions <- list(
  relu = activation_relu_grad,
  sigmoid = activation_sigmoid_grad,
  tanh = activation_tanh_grad,
  leaky_relu = function(out, grad) activation_leaky_relu_grad(out, grad, alpha=0.01),
  gelu = activation_gelu_grad
)

test_that("Activations forward/backward", {
  for (name in names(activations)) {
    act <- activations[[name]]
    grad_act <- grad_functions[[name]]

    out <- act(X)
    expect_equal(dim(out), dim(X))

    grad_out <- matrix(1, nrow=4, ncol=5)
    grad_in <- grad_act(out, grad_out)
    expect_equal(dim(grad_in), dim(X))
  }
})

# Squared Loss
test_that("Squared loss and gradient work", {
  y_pred <- matrix(rnorm(4), nrow=4, ncol=1)
  loss <- squared_loss(y, y_pred)
  expect_true(is.numeric(loss))

  grad <- squared_loss_grad(y, y_pred)
  expect_equal(dim(grad), dim(y_pred))
})

# Test SGD update
test_that("SGD updates parameters correctly", {
  params <- list(W=matrix(1,2,2), B=rep(0,2))
  grads  <- list(W=matrix(0.5,2,2), B=rep(0.1,2))

  updated <- sgd_update_params(params, grads, learning_rate=0.1)
  expect_equal(updated$W, params$W - 0.1*grads$W)
  expect_equal(updated$B, params$B - 0.1*grads$B)
})

# Test Sequential Model
test_that("Sequential forward/backward works", {
  model <- Sequential(
    Linear(5,4),
    activation_relu,
    Linear(4,1),
    activation_sigmoid
  )

  # Forward pass
  y_hat <- model$forward(X)
  expect_equal(dim(y_hat), c(4,1))

  # Backward pass
  grad_loss <- matrix(1, nrow=4, ncol=1)
  grad_input <- model$backward(grad_loss)
  expect_equal(dim(grad_input), dim(X))

  # Params and grads
  params <- model$get_params()
  grads  <- model$get_grads()

  expect_true(all(c("layer1.W","layer1.B","layer3.W","layer3.B") %in% names(params)))
  expect_true(all(c("layer1.W","layer1.B","layer3.W","layer3.B") %in% names(grads)))
})

# Test Training Loop single step
test_that("Single SGD step updates Sequential model", {
  model <- Sequential(
    Linear(5,4),
    activation_relu,
    Linear(4,1)
  )

  y_pred_before <- model$forward(X)
  grad_loss <- squared_loss_grad(y, y_pred_before)
  model$backward(grad_loss)

  # Update parameters
  for (i in seq_along(model$modules)) {
    m <- model$modules[[i]]
    if ("get_params" %in% names(m) && "get_grads" %in% names(m) && "set_params" %in% names(m)) {
      updated <- sgd_update_params(m$get_params(), m$get_grads(), learning_rate)
      m$set_params(updated)
    }
  }

  y_pred_after <- model$forward(X)
  # Output should change
  expect_false(all(y_pred_before == y_pred_after))
})
