
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
#> Current MSE Loss: 1.320475

# Backward pass to get the grad loss 
grad_loss <- squared_loss_grad(y_true, y_pred)

# populates the weight and biases grads 
model$backward(grad_loss)
#>               [,1]         [,2]          [,3]          [,4]          [,5]
#>  [1,] -0.018373951 -0.041110153  0.0086994641  0.0268395193  0.0044722750
#>  [2,]  0.046603543  0.033975373  0.0038912040  0.0173854485  0.0193534514
#>  [3,]  0.003462783 -0.006177783  0.0108879832 -0.0202108261  0.0051046936
#>  [4,]  0.001474807  0.000624759  0.0005182731 -0.0004250605  0.0006603488
#>  [5,]  0.008867062  0.001492264 -0.0015506561  0.0012230090 -0.0006349525
#>  [6,]  0.011895971  0.002002008 -0.0020803463  0.0016407779 -0.0008518466
#>  [7,] -0.002606051  0.005930438 -0.0013369640  0.0091214198  0.0110346134
#>  [8,] -0.024811267 -0.015355432  0.0027061014 -0.0004286350 -0.0207932042
#>  [9,] -0.002097002 -0.020315447 -0.0005575302  0.0077868134  0.0080968985
#> [10,] -0.080574212 -0.034132924 -0.0283152021  0.0232226443 -0.0360773260

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
#> Epoch 10/100 - Loss: 1.304374
#> Epoch 20/100 - Loss: 1.285731
#> Epoch 30/100 - Loss: 1.266161
#> Epoch 40/100 - Loss: 1.247576
#> Epoch 50/100 - Loss: 1.231416
#> Epoch 60/100 - Loss: 1.215397
#> Epoch 70/100 - Loss: 1.199607
#> Epoch 80/100 - Loss: 1.184131
#> Epoch 90/100 - Loss: 1.169031
#> Epoch 100/100 - Loss: 1.154216

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
