
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

The package’s primary interface is centered around the Module class for
defining network architectures and the Tensor class for handling data
and computing gradients. Here is a basic example

//\`\`\`{r example} library(MinML)

# Example: Initialize a simple network

model \<- MinML::Module_Sequential( MinML::Layer_Linear(input_features =
10, output_features = 64), MinML::Activation_ReLU(),
MinML::Layer_Linear(input_features = 64, output_features = 1) )

print(model) \`\`\`
