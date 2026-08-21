use synapse::{Value, MLP};

fn xor_data() -> Vec<([f64; 2], f64)> {
    vec![
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
}

fn main() {
    let data = xor_data();
    // 2 inputs -> 4 hidden -> 4 hidden -> 1 output, two hidden layers.
    let mlp = MLP::new(&[2, 4, 4, 1], 42);
    let lr = 0.1;
    let steps = 300;

    for step in 0..steps {
        mlp.zero_grad();

        let mut loss = Value::new(0.0);
        for (x, y) in &data {
            let inputs: Vec<Value> = x.iter().map(|v| Value::new(*v)).collect();
            let out = mlp.forward(&inputs);
            let pred = out[0].clone();
            let target = Value::new(*y);
            let diff = &pred - &target;
            loss = &loss + &(&diff * &diff);
        }
        let n = Value::new(data.len() as f64);
        loss = &loss / &n;

        loss.backward();
        mlp.step(lr);

        if step % 20 == 0 || step == steps - 1 {
            println!("step {step:4} loss {:.6}", loss.data());
        }
    }

    println!("\nfinal predictions:");
    for (x, y) in &data {
        let inputs: Vec<Value> = x.iter().map(|v| Value::new(*v)).collect();
        let out = mlp.forward(&inputs);
        println!(
            "  {:?} -> {:.4} (target {}, rounds to {})",
            x,
            out[0].data(),
            y,
            if out[0].data() > 0.5 { 1 } else { 0 }
        );
    }
}
