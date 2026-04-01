//! Multi-adapter registry example.
//!
//! This example demonstrates:
//! - Creating multiple adapters (LoRA and IA3)
//! - Registering them with AdapterRegistry
//! - Switching between adapters at runtime
//! - Performing forward passes with different active adapters

use anyhow::Result;
use candle_core::{Device, Tensor};
use peft_rs::{Adapter, AdapterRegistry, Ia3Config, Ia3Layer, LoraConfig, LoraLayer};

fn main() -> Result<()> {
    println!("=== Multi-Adapter Registry Example ===\n");

    // Set up device
    let device = Device::Cpu;

    // Define model dimensions
    let in_features = 512;
    let out_features = 512;

    // Create sample input
    let batch_size = 1;
    let seq_length = 8;
    let input = Tensor::randn(0f32, 1f32, (batch_size, seq_length, in_features), &device)?;

    println!("Input tensor shape: {:?}\n", input.shape());

    // ============================================================================
    // Create LoRA adapters with different configurations
    // ============================================================================

    println!("Creating LoRA adapters...");

    // LoRA adapter optimized for efficiency (low rank)
    let lora_config_small = LoraConfig {
        r: 4,
        alpha: 8,
        dropout: 0.0,
        ..Default::default()
    };
    let lora_small =
        LoraLayer::new_with_zeros(in_features, out_features, lora_config_small, &device)?;
    println!(
        "  lora_small: rank={}, params={}",
        4,
        lora_small.num_parameters()
    );

    // LoRA adapter with higher capacity (high rank)
    let lora_config_large = LoraConfig {
        r: 16,
        alpha: 32,
        dropout: 0.0,
        ..Default::default()
    };
    let lora_large =
        LoraLayer::new_with_zeros(in_features, out_features, lora_config_large, &device)?;
    println!(
        "  lora_large: rank={}, params={}",
        16,
        lora_large.num_parameters()
    );

    // ============================================================================
    // Create IA3 adapter (even more parameter-efficient)
    // ============================================================================

    println!("\nCreating IA3 adapter...");

    let ia3_config = Ia3Config {
        target_modules: vec!["default".to_string()],
        feedforward_modules: vec!["default".to_string()], // Mark as feedforward
        init_ia3_weights: true,
        fan_in_fan_out: false,
    };
    let ia3_layer = Ia3Layer::new(in_features, out_features, true, ia3_config, &device)?;
    println!("  ia3: params={}", ia3_layer.num_parameters());

    // ============================================================================
    // Register adapters in registry
    // ============================================================================

    println!("\n--- Setting up LoRA adapter registry ---");
    let mut lora_registry: AdapterRegistry<LoraLayer> = AdapterRegistry::new();

    lora_registry.register_adapter("small", lora_small)?;
    println!("✓ Registered 'small' adapter");

    lora_registry.register_adapter("large", lora_large)?;
    println!("✓ Registered 'large' adapter");

    println!("\nRegistry state:");
    println!("  Total adapters: {}", lora_registry.len());
    println!("  Adapter names: {:?}", lora_registry.adapter_names());
    println!(
        "  Active adapter: {}",
        lora_registry.active_adapter_name().unwrap_or("none")
    );

    // ============================================================================
    // Switch between adapters and run forward passes
    // ============================================================================

    println!("\n--- Testing adapter switching ---");

    // Forward with first adapter (automatically set as active)
    println!("\n1. Using 'small' adapter:");
    let output1 = lora_registry.forward(&input, None)?;
    println!("   Output shape: {:?}", output1.shape());
    let active_adapter = lora_registry.get_active_adapter()?;
    println!("   Parameters: {}", active_adapter.num_parameters());

    // Switch to second adapter
    lora_registry.set_active_adapter("large")?;
    println!("\n2. Switched to 'large' adapter:");
    let output2 = lora_registry.forward(&input, None)?;
    println!("   Output shape: {:?}", output2.shape());
    let active_adapter = lora_registry.get_active_adapter()?;
    println!("   Parameters: {}", active_adapter.num_parameters());

    // Switch back to first adapter
    lora_registry.set_active_adapter("small")?;
    println!("\n3. Switched back to 'small' adapter:");
    let output3 = lora_registry.forward(&input, None)?;
    println!("   Output shape: {:?}", output3.shape());

    // ============================================================================
    // Create a separate registry for IA3
    // ============================================================================

    println!("\n--- Setting up IA3 adapter registry ---");
    let mut ia3_registry: AdapterRegistry<Ia3Layer> = AdapterRegistry::new();

    ia3_registry.register_adapter("ia3_default", ia3_layer)?;
    println!("✓ Registered 'ia3_default' adapter");

    println!("\n4. Using IA3 adapter:");
    let output4 = ia3_registry.forward(&input, None)?;
    println!("   Output shape: {:?}", output4.shape());
    let active_ia3 = ia3_registry.get_active_adapter()?;
    println!("   Parameters: {}", active_ia3.num_parameters());

    // ============================================================================
    // Summary
    // ============================================================================

    println!("\n=== Summary ===");
    println!("✓ Successfully created and registered multiple adapters");
    println!("✓ Demonstrated adapter switching with AdapterRegistry");
    println!("✓ LoRA registry has {} adapters", lora_registry.len());
    println!("✓ IA3 registry has {} adapter", ia3_registry.len());
    println!(
        "✓ All outputs have correct shape: {:?}",
        output1.shape().dims()
    );

    Ok(())
}
