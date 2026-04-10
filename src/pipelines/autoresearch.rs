use anyhow::Result;

pub async fn run_loop() -> Result<()> {
    let mut best_score = 0.0;

    for i in 0..10 {
        println!("Iteration {}", i);

        // TODO:
        // 1. generate idea
        // 2. run experiment
        // 3. evaluate

        let score = rand::random::<f64>();

        if score > best_score {
            best_score = score;
            println!("New best: {}", best_score);
        }
    }

    Ok(())
}
