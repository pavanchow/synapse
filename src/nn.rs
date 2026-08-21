use crate::value::Value;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool, rng: &mut ChaCha8Rng) -> Self {
        let weights = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
            .collect();
        let bias = Value::new(0.0);
        Neuron {
            weights,
            bias,
            nonlin,
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        let mut act = self.bias.clone();
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            act = &act + &(w * xi);
        }
        if self.nonlin {
            act.tanh()
        } else {
            act
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool, rng: &mut ChaCha8Rng) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlin, rng)).collect();
        Layer { neurons }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// `sizes` is [nin, nhidden1, nhidden2, ..., nout]. All layers except
    /// the last apply tanh, the last layer is linear.
    pub fn new(sizes: &[usize], seed: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut layers = vec![];
        for i in 0..sizes.len() - 1 {
            let nonlin = i != sizes.len() - 2;
            layers.push(Layer::new(sizes[i], sizes[i + 1], nonlin, &mut rng));
        }
        MLP { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }

    pub fn step(&self, lr: f64) {
        for p in self.parameters() {
            let new_data = p.data() - lr * p.grad();
            p.set_data(new_data);
        }
    }
}
