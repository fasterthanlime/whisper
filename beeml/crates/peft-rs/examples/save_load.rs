//! Save and load adapter weights example.
//!
//! This example demonstrates:
//! - Creating an adapter with trained weights
//! - Saving adapter weights to a safetensors file
//! - Creating a new adapter instance
//! - Loading weights from file
//! - Verifying the loaded weights match the original

use anyhow::Result;
use candle_core::{Device, Tensor};
use peft_rs::{
    load_adapter_weights, save_adapter_weights, Adapter, LoraConfig, LoraLayer, SaveLoad,
};
use tempfile::TempDir;

fn main() -> Result<()> {
    println!("=== Save and Load Adapter Weights Example ===\n");

    // Set up device
    let device = Device::Cpu;

    // Define model dimensions
    let in_features = 256;
    let out_features = 256;

    // Create LoRA configuration
    let config = LoraConfig {
        r: 8,
        alpha: 16,
        dropout: 0.0,
        ..Default::default()
    };

    println!("Configuration:");
    println!("  Rank: {}", config.r);
    println!("  Alpha: {}", config.alpha);
    println!("  Input/Output features: {}\n", in_features);

    // ============================================================================
    // Step 1: Create and "train" an adapter
    // ============================================================================

    println!("Step 1: Creating original adapter");

    let original_adapter =
        LoraLayer::new_with_zeros(in_features, out_features, config.clone(), &device)?;

    println!(
        "  Created adapter with {} parameters",
        original_adapter.num_parameters()
    );

    // Create sample input for testing
    let test_input = Tensor::randn(0f32, 1f32, (1, 4, in_features), &device)?;
    println!("  Test input shape: {:?}", test_input.shape());

    // Run forward pass with original adapter
    let original_output = original_adapter.forward(&test_input, None)?;
    println!("  Original output shape: {:?}", original_output.shape());

    // Get a sample value from the output for verification
    let original_sample = original_output.flatten_all()?.get(0)?.to_scalar::<f32>()?;
    println!("  Sample output value: {:.6}", original_sample);

    // ============================================================================
    // Step 2: Save adapter weights to file
    // ============================================================================

    println!("\nStep 2: Saving adapter weights");

    // Create temporary directory for saving weights
    let temp_dir = TempDir::new()?;
    let weights_path = temp_dir.path().join("adapter_weights.safetensors");

    println!("  Saving to: {:?}", weights_path);

    // Save the adapter weights
    save_adapter_weights(&original_adapter, &weights_path)?;

    println!("  ✓ Weights saved successfully");

    // Verify file exists and check size
    let file_size = std::fs::metadata(&weights_path)?.len();
    println!("  File size: {} bytes", file_size);

    // ============================================================================
    // Step 3: Create a new adapter and load weights
    // ============================================================================

    println!("\nStep 3: Creating new adapter and loading weights");

    // Create a fresh adapter with the same configuration
    let mut loaded_adapter = LoraLayer::new_with_zeros(in_features, out_features, config, &device)?;

    println!("  Created new adapter instance");

    // Load the saved weights
    load_adapter_weights(&mut loaded_adapter, &weights_path, &device)?;

    println!("  ✓ Weights loaded successfully");

    // ============================================================================
    // Step 4: Verify loaded weights match original
    // ============================================================================

    println!("\nStep 4: Verifying loaded weights");

    // Run forward pass with loaded adapter
    let loaded_output = loaded_adapter.forward(&test_input, None)?;

    // Get sample value from loaded output
    let loaded_sample = loaded_output.flatten_all()?.get(0)?.to_scalar::<f32>()?;
    println!("  Loaded output sample: {:.6}", loaded_sample);

    // Compare outputs
    let difference = (original_sample - loaded_sample).abs();
    println!("  Difference: {:.10}", difference);

    // Verify outputs are very close (accounting for floating point precision)
    if difference < 1e-6 {
        println!("  ✓ Outputs match! (difference < 1e-6)");
    } else {
        println!("  ⚠ Warning: Outputs differ by {}", difference);
    }

    // Compare tensors element-wise
    let output_diff = original_output.sub(&loaded_output)?;
    let max_diff = output_diff
        .flatten_all()?
        .abs()?
        .max(0)?
        .to_scalar::<f32>()?;

    println!("  Maximum element-wise difference: {:.10}", max_diff);

    if max_diff < 1e-5 {
        println!("  ✓ All weights verified! (max diff < 1e-5)");
    } else {
        anyhow::bail!("Weights do not match! Max difference: {}", max_diff);
    }

    let original_state = original_adapter.state_dict()?;
    let loaded_state = loaded_adapter.state_dict()?;
    let lora_a_diff = loaded_state["lora_a.weight"]
        .sub(&original_state["lora_a.weight"])?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    let lora_b_diff = loaded_state["lora_b.weight"]
        .sub(&original_state["lora_b.weight"])?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;

    println!("  Max lora_a weight diff: {:.10}", lora_a_diff);
    println!("  Max lora_b weight diff: {:.10}", lora_b_diff);

    if lora_a_diff >= 1e-6 || lora_b_diff >= 1e-6 {
        anyhow::bail!(
            "Loaded state dict does not match original weights (a: {}, b: {})",
            lora_a_diff,
            lora_b_diff
        );
    }

    // ============================================================================
    // Step 5: Demonstrate state dict inspection
    // ============================================================================

    println!("\nStep 5: Inspecting saved state dict");

    let state_dict = original_state;
    println!("  State dict contains {} tensors:", state_dict.len());

    for (name, tensor) in &state_dict {
        println!(
            "    - {}: shape={:?}, dtype={:?}",
            name,
            tensor.shape().dims(),
            tensor.dtype()
        );
    }

    // ============================================================================
    // Summary
    // ============================================================================

    println!("\n=== Summary ===");
    println!(
        "✓ Created adapter with {} parameters",
        original_adapter.num_parameters()
    );
    println!("✓ Saved weights to safetensors format");
    println!("✓ Loaded weights into new adapter instance");
    println!("✓ Verified outputs match (max diff: {:.10})", max_diff);
    println!("✓ Adapter weights are portable and can be shared!");

    // Clean up happens automatically when temp_dir goes out of scope
    println!("\n(Temporary files cleaned up automatically)");

    Ok(())
}
