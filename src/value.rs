use std::cell::RefCell;
use std::collections::HashSet;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
enum Op {
    Leaf,
    Add,
    Mul,
    Pow(f64),
    Tanh,
    Neg,
}

struct Node {
    data: f64,
    grad: f64,
    op: Op,
    prev: Vec<Value>,
}

/// A single scalar in the computation graph. Cloning a `Value` shares the
/// underlying node (via `Rc<RefCell<_>>`), so the same node can appear more
/// than once in an expression and still accumulate gradients correctly.
#[derive(Clone)]
pub struct Value(Rc<RefCell<Node>>);

impl Value {
    pub fn new(data: f64) -> Self {
        Value(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op: Op::Leaf,
            prev: vec![],
        })))
    }

    fn from_op(data: f64, op: Op, prev: Vec<Value>) -> Self {
        Value(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op,
            prev,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn set_data(&self, x: f64) {
        self.0.borrow_mut().data = x;
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn pow(&self, n: f64) -> Value {
        let data = self.data().powf(n);
        Value::from_op(data, Op::Pow(n), vec![self.clone()])
    }

    pub fn tanh(&self) -> Value {
        let data = self.data().tanh();
        Value::from_op(data, Op::Tanh, vec![self.clone()])
    }

    pub fn relu(&self) -> Value {
        let data = self.data();
        if data > 0.0 {
            self.clone()
        } else {
            Value::new(0.0)
        }
    }

    /// Reverse-mode backward pass. Builds a topological order over the graph
    /// (each node after all the nodes it depends on), seeds this node's
    /// gradient to 1.0, then walks the order in reverse, pushing gradient
    /// contributions to each parent. Because parents accumulate (`+=`)
    /// rather than overwrite, a node used in more than one place in the
    /// expression correctly sums the gradient from every use.
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<usize> = HashSet::new();
        build_topo(self, &mut visited, &mut topo);

        self.0.borrow_mut().grad = 1.0;

        for v in topo.iter().rev() {
            v.propagate();
        }
    }

    fn propagate(&self) {
        let node = self.0.borrow();
        let grad_out = node.grad;
        match node.op {
            Op::Leaf => {}
            Op::Add => {
                for p in &node.prev {
                    p.0.borrow_mut().grad += grad_out;
                }
            }
            Op::Mul => {
                let a = &node.prev[0];
                let b = &node.prev[1];
                let a_data = a.data();
                let b_data = b.data();
                a.0.borrow_mut().grad += grad_out * b_data;
                b.0.borrow_mut().grad += grad_out * a_data;
            }
            Op::Pow(n) => {
                let a = &node.prev[0];
                let a_data = a.data();
                a.0.borrow_mut().grad += grad_out * n * a_data.powf(n - 1.0);
            }
            Op::Tanh => {
                let a = &node.prev[0];
                a.0.borrow_mut().grad += grad_out * (1.0 - node.data * node.data);
            }
            Op::Neg => {
                let a = &node.prev[0];
                a.0.borrow_mut().grad += -grad_out;
            }
        }
    }
}

fn build_topo(v: &Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
    if visited.contains(&v.id()) {
        return;
    }
    visited.insert(v.id());
    let prev = v.0.borrow().prev.clone();
    for p in &prev {
        build_topo(p, visited, topo);
    }
    topo.push(v.clone());
}

impl Add for &Value {
    type Output = Value;
    fn add(self, other: &Value) -> Value {
        let data = self.data() + other.data();
        Value::from_op(data, Op::Add, vec![self.clone(), other.clone()])
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, other: Value) -> Value {
        &self + &other
    }
}

impl Mul for &Value {
    type Output = Value;
    fn mul(self, other: &Value) -> Value {
        let data = self.data() * other.data();
        Value::from_op(data, Op::Mul, vec![self.clone(), other.clone()])
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        &self * &other
    }
}

impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value {
        Value::from_op(-self.data(), Op::Neg, vec![self.clone()])
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        -&self
    }
}

impl Sub for &Value {
    type Output = Value;
    fn sub(self, other: &Value) -> Value {
        self + &(-other)
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, other: Value) -> Value {
        &self - &other
    }
}

impl Div for &Value {
    type Output = Value;
    fn div(self, other: &Value) -> Value {
        self * &other.pow(-1.0)
    }
}

impl Div for Value {
    type Output = Value;
    fn div(self, other: Value) -> Value {
        &self / &other
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_grad<F: Fn(f64) -> f64>(f: F, x: f64, eps: f64) -> f64 {
        (f(x + eps) - f(x - eps)) / (2.0 * eps)
    }

    #[test]
    fn gradient_check_expression() {
        // f(a) = (a * a + 2) * tanh(a) - a / 2
        let f = |a: f64| (a * a + 2.0) * a.tanh() - a / 2.0;
        let x = 0.8f64;

        let a = Value::new(x);
        let two = Value::new(2.0);
        let half = Value::new(2.0);
        let out = &(&(&(&a * &a) + &two) * &a.tanh()) - &(&a / &half);
        out.backward();

        let expected = numeric_grad(f, x, 1e-6);
        let got = a.grad();
        assert!(
            (expected - got).abs() < 1e-4,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn shared_node_gradient_accumulates() {
        // d = a * a  =>  dd/da = 2a
        let a = Value::new(3.0);
        let d = &a * &a;
        d.backward();
        assert!((a.grad() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn shared_node_in_larger_expression() {
        // e = a*a + a  used twice: d(e)/da = 2a + 1
        let a = Value::new(-2.0);
        let e = &(&a * &a) + &a;
        e.backward();
        let expected = 2.0 * -2.0 + 1.0;
        assert!((a.grad() - expected).abs() < 1e-9);
    }
}
