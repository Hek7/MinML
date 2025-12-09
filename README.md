
<!-- README.md is generated from README.Rmd. Please edit that file -->

# MinML

<!-- badges: start -->

<!-- badges: end -->

The MinML package is a lightweight and modular machine learning
framework for R, created for learning how to implement a ML package from
the ground up in R for more customizbility.

## Installation

You can install the development version of MinML from
[GitHub](https://github.com/) with:

``` r
# install.packages("pak")
pak::pak("Hek7/MinML")
```

## TODO

The finished touches are to write a suite of tests for th test_that
folder and add to the functions with more optimizers and losses.

## Example

### Example on how to build and train a model in MinML

``` r
library(MinML)

# Dummy normal Data 
X <- matrix(rnorm(50), nrow = 10, ncol = 5)

# Dummy targets 
y_true <- matrix(rnorm(10), nrow = 10, ncol = 1)

# Define model with linear and activations
model <- Sequential(
  Linear(input_features = 5, output_features = 10),
  activation_relu,
  Linear(input_features = 10, output_features = 1),
  activation_sigmoid
)

# Forward pass returns predictions for current weights
y_pred <- model$forward(X)

cat("Prediction dimensions:", dim(y_pred), "\n")
#> Prediction dimensions: 10 1

# Calculate the loss other losses to get added
loss <- squared_loss(y_true, y_pred)
cat("Current MSE Loss:", loss, "\n")
#> Current MSE Loss: 1.808478

# Backward pass to get the grad loss 
grad_loss <- squared_loss_grad(y_true, y_pred)

# populates the weight and biases grads 
model$backward(grad_loss)
#>                [,1]          [,2]          [,3]          [,4]          [,5]
#>  [1,]  1.379593e-02  4.480820e-04  0.0296109226 -2.696864e-02  2.364779e-02
#>  [2,] -1.025373e-03 -2.924688e-03  0.0035401858 -4.991159e-03  3.770150e-03
#>  [3,] -2.989546e-04 -1.350317e-04 -0.0002388264  2.322285e-04 -5.437974e-05
#>  [4,]  1.172883e-03 -6.604360e-04  0.0027025593 -2.699465e-03  3.912006e-03
#>  [5,] -1.449795e-02 -2.186940e-02 -0.0175663688  1.430846e-02 -1.099508e-03
#>  [6,]  2.142826e-05 -1.405059e-04  0.0004936034 -4.559050e-04  1.780022e-04
#>  [7,] -1.677165e-02 -4.482891e-03 -0.0234956942  1.273631e-02 -1.849412e-02
#>  [8,] -1.191894e-02  1.215185e-02 -0.0213308079  2.368331e-02 -3.123182e-02
#>  [9,]  6.594566e-05 -4.514603e-05  0.0001198832 -9.911831e-05  1.512183e-04
#> [10,] -2.307671e-02 -4.029047e-03 -0.0495441274  3.831571e-02 -3.241723e-02

#Some getters for getting the grads 
all_grads <- model$get_grads()
```

### Example: Training Loop

``` r
epochs <- 100
learning_rate <- 0.01

# Initialize a vector to store loss history
loss_history <- numeric(epochs)

for (epoch in 1:epochs) {
  
  # Forward pass 
  y_pred <- model$forward(X)
  
  # Compute Loss
  loss <- squared_loss(y_true, y_pred)
  
  # Store loss in history
  loss_history[epoch] <- loss
  
  # Backward pass 
  grad_loss <- squared_loss_grad(y_true, y_pred)
  
  # Backprop for grads 
  model$backward(grad_loss)
  
  # Optimize using SGD  
  for (i in seq_along(model$modules)) {
    m <- model$modules[[i]]
    
    # Only update layers that have params
    if ("get_params" %in% names(m) && "get_grads" %in% names(m)) {
      
      # Get current params & grads
      params <- m$get_params()
      grads  <- m$get_grads()
      
      # Compute updated params
      updated <- sgd_update_params(params, grads, learning_rate)
      
      m$set_params(updated)
    
      }
    }
  }

  # Print progress every 10 epochs
  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch %d/%d - Loss: %.6f\n", epoch, epochs, loss))
  }
#> Epoch 100/100 - Loss: 1.597576

cat("Training complete.\n")
#> Training complete.

# Plot loss history
plot(1:epochs, loss_history, type="l", col="blue", lwd=2,
     xlab="Epoch", ylab="Loss (MSE)", main="Training Loss over Epochs")
```

<img src="man/figures/README-training example-1.png" width="100%" />

### Different Activation Functions

Other activation functions:

``` r
# Create models with different activations
model_sigmoid <- Sequential(
  Linear(2, 16),
  activation_sigmoid,
  Linear(16, 1)
)

model_tanh <- Sequential(
  Linear(2, 16),
  activation_tanh,
  Linear(16, 1)
)

model_leaky_relu <- Sequential(
  Linear(2, 16),
  function(x) activation_leaky_relu(x, alpha = 0.1),
  Linear(16, 1)
)

model_gelu <- Sequential(
  Linear(2, 16),
  activation_gelu,
  Linear(16, 1)
)
```
