**A neural network from scratch in Rust.**

Synapse is a small reverse-mode autograd engine and a tiny neural network library, built with no ML or tensor crates. The whole thing is small enough to read node by node in an afternoon.

## The idea

At the center is `Value`, a scalar that remembers how it was computed. Every add, multiply, subtract, divide, power, negation, or tanh you apply to a `Value` records a node in a computation graph, with pointers back to the values that produced it. Call `backward()` on the final output and the graph walks itself in reverse topological order, seeding the output gradient to 1.0 and pushing gradient contributions back through every operation using the chain rule. If a value gets used more than once in an expression, its gradient contributions from each use are accumulated rather than overwritten, so shared subexpressions come out correct.

On top of `Value` sits a minimal neural network stack: a `Neuron` is a set of weights, a bias, and an optional tanh nonlinearity, a `Layer` is a set of neurons, and an `MLP` is a stack of layers. Forward pass is just `Value` arithmetic, so gradients for every weight and bias in the whole network fall out of a single `backward()` call. Training is plain gradient descent: compute the loss, call `backward()`, nudge every parameter by `-lr * grad`, zero the grads, repeat.

## The XOR demo

`src/main.rs` builds a 2-4-4-1 MLP (two hidden layers of four tanh neurons each) and trains it on the four XOR points with mean squared error. XOR is the classic case that a single-layer network cannot represent, so it is a decent sanity check that the hidden layers and the autograd are actually doing something. Running it prints the loss falling every 20 steps and the final rounded prediction for each of the four inputs.

## Usage

```
cargo run --release
```

Trains the XOR network for 300 steps and prints the loss curve and final predictions.

```
cargo test
```

Runs three kinds of checks:

- a gradient check that compares the autograd gradient on a small expression against a numerical finite-difference gradient
- a shared-node test (`d = a * a`) that confirms the gradient comes out to `2a` and not `a`
- a training test that trains the XOR MLP and asserts the loss drops below a threshold and all four inputs classify correctly

## Layout

- `src/value.rs` the `Value` type and the autograd engine
- `src/nn.rs` `Neuron`, `Layer`, `MLP`
- `src/main.rs` the XOR training demo
- `tests/xor_training.rs` the end-to-end training test
- `DESIGN.md` a longer writeup of the graph, the backward pass, and the training loop

By Pavan Nallamothu.
