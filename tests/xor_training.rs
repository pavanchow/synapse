use synapse::{Value, MLP};

fn xor_data() -> Vec<([f64; 2], f64)> {
    vec![
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]
}

#[test]
fn trains_xor_below_threshold() {
    let data = xor_data();
    let mlp = MLP::new(&[2, 4, 4, 1], 42);
    let lr = 0.1;
    let steps = 400;

    let mut final_loss = f64::MAX;
    for _ in 0..steps {
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
        final_loss = loss.data();
    }

    assert!(
        final_loss < 0.05,
        "final loss {final_loss} did not go below threshold"
    );

    for (x, y) in &data {
        let inputs: Vec<Value> = x.iter().map(|v| Value::new(*v)).collect();
        let out = mlp.forward(&inputs);
        let rounded = if out[0].data() > 0.5 { 1.0 } else { 0.0 };
        assert_eq!(rounded, *y, "input {x:?} classified wrong");
    }
}
