use crate::{get_grid_from_seed, PerlinNoise, Vec3};
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;

pub fn intial_world_generation_test() {
    println!("Starting intial world generation test");

    let mut generation_times: Vec<Duration> = Vec::with_capacity(100);
    let mut flatten_times: Vec<Duration> = Vec::with_capacity(100);
    let test_count = 100;
    
    for i in 0..test_count {
        println!("loop {i}");

        let seed = 42 + i;
        let mut start_location = Vec3::from(100000.0, 10272.0, 100000.0);
        let temp = PerlinNoise::new(seed, 0.00003);
        let mut found = false;
        
        // Ensures camera always starts on land
        while !found {
            if temp.get_noise_at_point(start_location.x, start_location.z) > 0.05 {
                found = true;
            } else {
                    start_location.x += 1.0;
            }
        }

        println!("{:?}", start_location);
        {
            let initial = Instant::now();
            let mut temp = get_grid_from_seed(seed, 321, [start_location.x as i32, start_location.y as i32, start_location.z as i32]);
            
            let elapsed1 = initial.elapsed();
            generation_times.push(elapsed1);
        
            let initial = Instant::now();
            let flat_world = temp.flatten_world();

            let elapsed2 = initial.elapsed();
            flatten_times.push(elapsed2);
            println!("{i}: {}, {}", elapsed1.as_millis(), elapsed2.as_millis());
        }

        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    let mut file = File::create("world_generation_times.csv").expect("Failed to create file");
    
    // Write CSV header
    writeln!(file, "generation_time_ms,flatten_time_ms").expect("Failed to write header");
    
    // Write the timing data
    for i in 0..(test_count as usize) {
        let gen_time = generation_times[i].as_millis();
        let flatten_time = flatten_times[i].as_millis();
        writeln!(file, "{},{}", gen_time, flatten_time).expect("Failed to write data");
    }
    
    // Calculate and print statistics to console
    let gen_avg = generation_times.iter().sum::<Duration>() / generation_times.len() as u32;
    let flatten_avg = flatten_times.iter().sum::<Duration>() / flatten_times.len() as u32;
    
    println!("Results saved to world_generation_times.csv");
    println!("Average generation time: {:?}", gen_avg);
    println!("Average flatten time: {:?}", flatten_avg);
}
