use mlx_rs::module::Module;
use mlx_rs::ops::indexing::IndexOp;

use crate::config::T5Config;
use crate::load::load_weights_direct;
use crate::model::T5ForConditionalGeneration;
use crate::test_helpers::{assert_close, load_reference, model_dir};
use crate::tokenize;

const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-4;

fn load_model() -> Option<T5ForConditionalGeneration> {
    let model_dir = model_dir()?;
    let config = T5Config::charsiu_g2p();
    let mut model = T5ForConditionalGeneration::new(config).ok()?;
    let stats = load_weights_direct(&mut model, &model_dir).ok()?;
    assert_eq!(stats.loaded, 172, "Expected 172 tensors loaded");
    Some(model)
}

#[test]
fn test_embedding_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let expected_embed = ref_tensors.get("encoder.embed_out").unwrap();

    let actual_embed = model.encoder.embed_tokens.forward(input_ids).unwrap();
    // Reference is [seq_len, d_model], actual is [1, seq_len, d_model]
    let actual_embed = actual_embed.index((0,));

    assert_close(
        "encoder.embed_out",
        &actual_embed,
        expected_embed,
        ATOL,
        RTOL,
    );
}

#[test]
fn test_position_bias_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let expected_bias = ref_tensors.get("encoder.position_bias").unwrap();
    // Expected shape: [6, 15, 15] (heads, qlen, klen)

    let actual_bias = model.encoder.position_bias.forward(15, 15).unwrap();
    // Our output: [1, 6, 15, 15] — remove batch dim
    let actual_bias = actual_bias.index((0,));

    assert_close(
        "encoder.position_bias",
        &actual_bias,
        expected_bias,
        ATOL,
        RTOL,
    );
}

#[test]
fn test_encoder_block_0_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let expected_out = ref_tensors.get("encoder.block.0.out").unwrap();

    // Run embedding + position bias + block 0
    let embed = model.encoder.embed_tokens.forward(input_ids).unwrap();
    let bias = model.encoder.position_bias.forward(15, 15).unwrap();
    let block_out = model.encoder.blocks[0]
        .forward(&embed, Some(&bias))
        .unwrap();
    let block_out = block_out.index((0,));

    assert_close("encoder.block.0.out", &block_out, expected_out, ATOL, RTOL);
}

#[test]
fn test_encoder_block_0_attn_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let expected_attn = ref_tensors.get("encoder.block.0.attn_out").unwrap();

    let embed = model.encoder.embed_tokens.forward(input_ids).unwrap();
    let bias = model.encoder.position_bias.forward(15, 15).unwrap();

    // Run just the self-attention sublayer (pre-norm + attention)
    let normed = model.encoder.blocks[0]
        .self_attn_norm
        .forward(&embed)
        .unwrap();
    let attn_out = model.encoder.blocks[0]
        .self_attn
        .forward(&normed, &normed, Some(&bias), None)
        .unwrap();
    let attn_out = attn_out.index((0,));

    assert_close(
        "encoder.block.0.attn_out",
        &attn_out,
        expected_attn,
        ATOL,
        RTOL,
    );
}

#[test]
fn test_full_encoder_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let expected_final = ref_tensors.get("encoder.final_out").unwrap();

    let actual_final = model.encode(input_ids).unwrap();
    let actual_final = actual_final.index((0,));

    assert_close(
        "encoder.final_out",
        &actual_final,
        expected_final,
        ATOL,
        RTOL,
    );
}

#[test]
fn test_decoder_self_attn_bias_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let expected_bias = ref_tensors.get("decoder.self_attn_bias").unwrap();
    // Expected: [6, 12, 12]

    let actual_bias = model.decoder.position_bias.forward(12, 12).unwrap();
    let actual_bias = actual_bias.index((0,));

    assert_close(
        "decoder.self_attn_bias",
        &actual_bias,
        expected_bias,
        ATOL,
        RTOL,
    );
}

#[test]
fn test_decoder_block_0_self_attn_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let decoder_input_ids = ref_tensors.get("decoder_input_ids").unwrap();
    let expected_self_attn = ref_tensors.get("decoder.block.0.self_attn_out").unwrap();

    let dec_len = decoder_input_ids.shape()[0] as i32;
    let decoder_input_ids = decoder_input_ids.reshape(&[1, dec_len]).unwrap();

    let embed = model
        .decoder
        .embed_tokens
        .forward(&decoder_input_ids)
        .unwrap();
    let self_attn_bias = model
        .decoder
        .position_bias
        .forward(dec_len, dec_len)
        .unwrap();

    let normed = model.decoder.blocks[0]
        .self_attn_norm
        .forward(&embed)
        .unwrap();
    // No separate causal mask — the decoder position bias (is_decoder=true) already encodes causality
    let self_attn_out = model.decoder.blocks[0]
        .self_attn
        .forward(&normed, &normed, Some(&self_attn_bias), None)
        .unwrap();
    let self_attn_out = self_attn_out.index((0,));

    assert_close(
        "decoder.block.0.self_attn_out",
        &self_attn_out,
        expected_self_attn,
        ATOL,
        RTOL,
    );
}

#[test]
fn test_decoder_block_0_out_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let decoder_input_ids = ref_tensors.get("decoder_input_ids").unwrap();
    let expected_out = ref_tensors.get("decoder.block.0.out").unwrap();

    let dec_len = decoder_input_ids.shape()[0] as i32;
    let decoder_input_ids = decoder_input_ids.reshape(&[1, dec_len]).unwrap();

    let encoder_output = model.encode(input_ids).unwrap();
    let embed = model
        .decoder
        .embed_tokens
        .forward(&decoder_input_ids)
        .unwrap();
    let self_attn_bias = model
        .decoder
        .position_bias
        .forward(dec_len, dec_len)
        .unwrap();

    let block_out = model.decoder.blocks[0]
        .forward(&embed, &encoder_output, Some(&self_attn_bias), None, None)
        .unwrap();
    let block_out = block_out.index((0,));

    assert_close("decoder.block.0.out", &block_out, expected_out, ATOL, RTOL);
}

#[test]
fn test_decoder_final_out_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let decoder_input_ids = ref_tensors.get("decoder_input_ids").unwrap();
    let expected_final = ref_tensors.get("decoder.final_out").unwrap();

    let dec_len = decoder_input_ids.shape()[0] as i32;
    let decoder_input_ids = decoder_input_ids.reshape(&[1, dec_len]).unwrap();

    let encoder_output = model.encode(input_ids).unwrap();
    let decoder_out = model
        .decoder
        .forward(&decoder_input_ids, &encoder_output)
        .unwrap();
    let decoder_out = decoder_out.index((0,));

    assert_close(
        "decoder.final_out",
        &decoder_out,
        expected_final,
        ATOL,
        RTOL,
    );
}

#[test]
fn test_full_logits_match_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let decoder_input_ids = ref_tensors.get("decoder_input_ids").unwrap();
    let expected_logits = ref_tensors.get("logits").unwrap();

    // decoder_input_ids is [12], need to make it [1, 12]
    let dec_len = decoder_input_ids.shape()[0] as i32;
    let decoder_input_ids = decoder_input_ids.reshape(&[1, dec_len]).unwrap();

    let encoder_output = model.encode(input_ids).unwrap();
    let logits = model.decode(&decoder_input_ids, &encoder_output).unwrap();
    let logits = logits.index((0,));

    assert_close("logits", &logits, expected_logits, ATOL, RTOL);
}

#[test]
fn test_greedy_generation_matches_reference() {
    let Some((_dir, ref_tensors)) = load_reference() else {
        return;
    };
    let Some(model) = load_model() else { return };

    let input_ids = ref_tensors.get("input_ids").unwrap();
    let expected_ids = ref_tensors.get("generated_ids").unwrap();
    expected_ids.eval().unwrap();

    let generated = model.generate(input_ids, 64).unwrap();
    let expected: Vec<i32> = (0..expected_ids.shape()[0])
        .map(|i| {
            let v = expected_ids.index((i as i32,));
            v.item::<i32>()
        })
        .collect();

    // Expected includes decoder_start_token (0) at front and eos (1) at end
    // Our generate() strips both
    let expected_stripped: Vec<i32> = expected
        .iter()
        .copied()
        .filter(|&id| id != 0 && id != 1)
        .collect();

    assert_eq!(
        generated,
        expected_stripped,
        "Generated IDs don't match reference.\n\
         Expected: {expected_stripped:?}\n\
         Got:      {generated:?}\n\
         Expected IPA: {:?}\n\
         Got IPA:      {:?}",
        tokenize::decode_byt5(&expected_stripped),
        tokenize::decode_byt5(&generated),
    );
}
