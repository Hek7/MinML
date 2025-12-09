
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
#> Current MSE Loss: 0.6590584

# Backward pass to get the grad loss 
grad_loss <- squared_loss_grad(y_true, y_pred)

# populates the weight and biases grads 
model$backward(grad_loss)
#>                [,1]          [,2]          [,3]          [,4]          [,5]
#>  [1,] -2.349855e-03 -1.631331e-03  3.277391e-03 -0.0041331347 -5.102495e-03
#>  [2,] -1.075789e-02  6.013663e-03  6.784315e-03 -0.0100201131 -1.698694e-02
#>  [3,] -1.440114e-05  1.107837e-03 -1.956503e-04  0.0004668532  2.642539e-04
#>  [4,]  6.857832e-03  4.760887e-03 -9.564757e-03  0.0120621661  1.489115e-02
#>  [5,] -4.422086e-04  4.066641e-04  6.328995e-05 -0.0005987881 -2.571472e-05
#>  [6,]  3.144394e-02 -1.500407e-02 -2.098913e-02  0.0265131156  4.025372e-02
#>  [7,]  7.800663e-03  2.081562e-03  4.148020e-03  0.0140994461  7.730058e-03
#>  [8,] -7.653304e-03  1.315083e-06  3.437655e-03 -0.0062853233 -1.046578e-02
#>  [9,] -1.956570e-04  1.770963e-03 -1.235513e-03 -0.0008418589 -1.719328e-03
#> [10,] -1.959535e-05  1.507413e-03 -2.662177e-04  0.0006352384  3.595653e-04

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

  # Print progress every 10 epochs
  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch %d/%d - Loss: %.6f\n", epoch, epochs, loss))
  }
}
#> Epoch 10/100 - Loss: 0.654944
#> Epoch 20/100 - Loss: 0.650651
#> Epoch 30/100 - Loss: 0.646617
#> Epoch 40/100 - Loss: 0.642832
#> Epoch 50/100 - Loss: 0.639271
#> Epoch 60/100 - Loss: 0.635909
#> Epoch 70/100 - Loss: 0.632725
#> Epoch 80/100 - Loss: 0.629700
#> Epoch 90/100 - Loss: 0.626815
#> Epoch 100/100 - Loss: 0.624056

cat("Training complete.\n")
#> Training complete.
```

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

model_gelu <- Sequential(
  Linear(2, 16),
  activation_gelu,
  Linear(16, 1)
)
```
