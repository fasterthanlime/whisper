//! Basic LoRA adapter usage example.
//!
//! This example demonstrates:
//! - Creating a LoRA configuration
//! - Initializing a LoRA layer
//! - Performing a forward pass with a tensor
//! - Inspecting shapes and parameter counts

use anyhow::Result;
use candle_core::{Device, Tensor};
use peft_rs::{Adapter, LoraConfig, LoraLayer};

fn main() -> Result<()> {
    println!("=== Basic LoRA Example ===\n");

    // Set up device (CPU for simplicity)
    let device = Device::Cpu;

    // Define model dimensions
    let in_features = 768;
    let out_features = 768;

    // Create LoRA configuration
    // - r: rank of low-rank decomposition (smaller = fewer parameters)
    // - alpha: scaling factor (typically alpha/r ratio determines adaptation strength)
    // - dropout: dropout probability (0.0 = no dropout)
    let config = LoraConfig {
        r: 8,
        alpha: 16,
        dropout: 0.0,
        ..Default::default()
    };

    println!("LoRA Configuration:");
    println!("  Rank (r): {}", config.r);
    println!("  Alpha: {}", config.alpha);
    println!(
        "  Scaling factor: {}",
        config.alpha as f64 / config.r as f64
    );
    println!("  Dropout: {}\n", config.dropout);

    // Create a LoRA layer
    // This initializes the low-rank matrices A and B
    let lora_layer = LoraLayer::new_with_zeros(in_features, out_features, config, &device)?;

    // Print parameter information
    let num_params = lora_layer.num_parameters();
    println!("Layer Information:");
    println!("  Input features: {}", in_features);
    println!("  Output features: {}", out_features);
    println!("  Total trainable parameters: {}", num_params);
    println!(
        "  Parameter breakdown: A({} × {}) + B({} × {}) = {} + {} = {}",
        in_features,
        8,
        8,
        out_features,
        in_features * 8,
        8 * out_features,
        num_params
    );

    // Calculate parameter reduction compared to full fine-tuning
    let full_params = in_features * out_features;
    let reduction_ratio = full_params as f64 / num_params as f64;
    println!(
        "  Compared to full fine-tuning: {:.2}x parameter reduction\n",
        reduction_ratio
    );

    // Create a sample input tensor
    // Shape: [batch_size, sequence_length, hidden_dim]
    let batch_size = 2;
    let seq_length = 10;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_length, in_features), &device)?;

    println!("Input tensor shape: {:?}", input.shape());

    // Perform forward pass
    // LoRA computes: output = input + (input @ A @ B) * scaling
    let output = lora_layer.forward(&input, None)?;

    println!("Output tensor shape: {:?}", output.shape());

    // Verify shapes match
    assert_eq!(
        input.shape().dims(),
        output.shape().dims(),
        "Input and output shapes should match"
    );

    println!("\n✓ Forward pass successful!");
    println!("✓ LoRA layer is ready for training or inference");

    Ok(())
}
