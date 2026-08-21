# Design

## The Value graph

`Value` wraps a shared, mutable node behind `Rc<RefCell<Node>>`. Each node stores its scalar `data`, its accumulated `grad`, which operation produced it (`Leaf`, `Add`, `Mul`, `Pow(n)`, `Tanh`, `Neg`), and a list of the `Value`s that were its direct inputs (`prev`).

Cloning a `Value` clones the `Rc`, not the node, so two clones point at the same underlying data. This matters because Rust expressions like `a * a` or `a + a` naturally clone `a` twice, and both clones need to refer to the same node for gradient accumulation to be meaningful. Operator overloads (`Add`, `Mul`, `Sub`, `Div`, `Neg`, plus the `pow` and `tanh` methods) each compute the forward value immediately and create a new node whose `prev` points at the input node(s).

Division is implemented as multiplication by `a.pow(-1.0)` rather than as its own op, and subtraction as addition of a negation, so the only primitive backward rules that need to exist are for add, mul, pow, tanh, and neg.

## Backward pass and topological order

`backward()` on a `Value` needs to visit every node that contributed to it exactly once, and it needs to visit a node only after every node that depends on it has already pushed its gradient contribution downstream. That is a reverse topological order.

`build_topo` does a recursive depth-first walk from the output node: visit a node's parents first, then append the node itself to the `topo` list, tracking visited nodes by the node's pointer identity (`Rc::as_ptr` cast to `usize`) so a node reachable through multiple paths is only processed once. This produces a list where every node appears after all of its dependencies.

`backward()` seeds the output's own `grad` to `1.0` (d(output)/d(output) = 1), then walks `topo` in reverse. For each node, `propagate()` looks at its `op` and pushes gradient contributions to its `prev` nodes using the local derivative for that op:

- `Add`: gradient passes through unchanged to both inputs
- `Mul`: each input gets `grad_out` times the *other* input's data
- `Pow(n)`: input gets `grad_out * n * data^(n-1)`
- `Tanh`: input gets `grad_out * (1 - tanh(x)^2)`, using the already-computed output data
- `Neg`: input gets `-grad_out`

Because the walk is in reverse topological order, by the time a node is processed, every node downstream of it (closer to the output) has already deposited its full contribution into that node's `grad`. So a node's `grad` is complete before it is used to compute further contributions to its own parents.

## Gradient accumulation

Every `propagate()` step uses `+=` on the parent's `grad`, never `=`. That is the entire mechanism for handling shared subexpressions: if a node is `prev` for more than one downstream node (for example `d = a * a` treats `a` as both inputs of the multiply, or a weight is reused across the whole forward pass of a network with many training points), each downstream user adds its own contribution, and by the time `backward()` finishes, `grad` holds the sum, matching the multivariable chain rule. Before each new training step, `zero_grad()` resets every parameter's `grad` back to zero so contributions from the previous step do not bleed into the next.

## The MLP

`Neuron` holds a `Vec<Value>` of weights and one bias `Value`, initialized from a seeded RNG (`ChaCha8Rng`) so runs are deterministic. Its `forward` computes `bias + sum(w_i * x_i)` using `Value` arithmetic and optionally applies `tanh`. `Layer` is a `Vec<Neuron>` whose `forward` maps the same input across all neurons and collects their outputs. `MLP` is a `Vec<Layer>` built from a sizes list like `[2, 4, 4, 1]`, chaining each layer's output into the next layer's input. Every layer except the last applies `tanh`, the last is linear so the raw output can be compared against a 0/1 target.

`parameters()` walks the layers and neurons and flattens every weight and bias into one `Vec<Value>`, which is what `zero_grad()` and `step(lr)` operate over.

## The training loop

For each step: `zero_grad()` clears every parameter's gradient, then for each of the four XOR examples the inputs are wrapped as fresh `Value` leaves, pushed through `mlp.forward`, and the squared difference between the prediction and the target is accumulated into a running `loss`. The loss is divided by the number of examples to get a mean, then `loss.backward()` populates every parameter's `grad` in one call. `step(lr)` then updates every parameter with a plain gradient descent step, `data -= lr * grad`. Repeating this for a few hundred steps is enough for the loss to collapse from around 1.8 to effectively zero and for every XOR input to round to the correct label.
