
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
#> Current MSE Loss: 1.628466

# Backward pass to get the grad loss 
grad_loss <- squared_loss_grad(y_true, y_pred)

# populates the weight and biases grads 
model$backward(grad_loss)
#>                [,1]         [,2]          [,3]         [,4]         [,5]
#>  [1,] -0.0193285741  0.004394359  0.0163674524 -0.017524437 -0.027542586
#>  [2,] -0.0023950607 -0.012483688  0.0018785691  0.014641358  0.011619516
#>  [3,] -0.0060534863 -0.001031099  0.0009470001 -0.010379965 -0.009258305
#>  [4,] -0.0033475453 -0.022723582 -0.0017460978  0.008110680  0.010814871
#>  [5,]  0.0008099892 -0.005755135 -0.0028203764  0.021881868  0.008507631
#>  [6,] -0.0144335021 -0.037834660 -0.0034505894 -0.013123891  0.002473453
#>  [7,] -0.0057162248 -0.002192389  0.0023906942 -0.008186953 -0.008183297
#>  [8,] -0.0097625683 -0.009750820  0.0042637387 -0.007179971 -0.005258643
#>  [9,] -0.0083219202  0.002743776  0.0018663903 -0.027111748 -0.023072447
#> [10,] -0.0054247517 -0.036823935 -0.0028295799  0.013143489  0.017525674

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
#> Epoch 100/100 - Loss: 1.108620

cat("Training complete.\n")
#> Training complete.

# Plot loss history
plot(1:epochs, loss_history, type="l", col="blue", lwd=2,
     xlab="Epoch", ylab="Loss (MSE)", main="Training Loss over Epochs")
```

<img src="man/figures/README-training example-1.png" width="100%" />

``` r
m
#> function (x) 
#> {
#>     x <- as.matrix(x)
#>     as.matrix(1/(1 + exp(-x)))
#> }
#> <bytecode: 0x00000227798af310>
#> <environment: namespace:MinML>
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
