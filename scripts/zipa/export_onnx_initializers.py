#!/usr/bin/env python3

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


NAME_MAP = {
    "encoder_embed.conv.0.weight": "encoder_embed.conv0.weight",
    "encoder_embed.conv.0.bias": "encoder_embed.conv0.bias",
    "encoder_embed.conv.4.weight": "encoder_embed.conv1.weight",
    "encoder_embed.conv.4.bias": "encoder_embed.conv1.bias",
    "encoder_embed.conv.7.weight": "encoder_embed.conv2.weight",
    "encoder_embed.conv.7.bias": "encoder_embed.conv2.bias",
    "encoder_embed.convnext.depthwise_conv.weight": "encoder_embed.convnext.depthwise_conv.weight",
    "encoder_embed.convnext.depthwise_conv.bias": "encoder_embed.convnext.depthwise_conv.bias",
    "encoder_embed.convnext.pointwise_conv1.weight": "encoder_embed.convnext.pointwise_conv1.weight",
    "encoder_embed.convnext.pointwise_conv1.bias": "encoder_embed.convnext.pointwise_conv1.bias",
    "encoder_embed.convnext.pointwise_conv2.weight": "encoder_embed.convnext.pointwise_conv2.weight",
    "encoder_embed.convnext.pointwise_conv2.bias": "encoder_embed.convnext.pointwise_conv2.bias",
    "onnx::MatMul_11174": "encoder_embed.out.weight",
    "encoder_embed.out.bias": "encoder_embed.out.bias",
    "encoder_embed.out_norm.log_scale": "encoder_embed.out_norm.log_scale",
    "encoder_embed.out_norm.bias": "encoder_embed.out_norm.bias",
    "onnx::MatMul_12110": "ctc_output.linear.weight",
    "ctc_output.1.bias": "ctc_output.linear.bias",
}

STAGE0_LAYER_MATMULS = {
    0: {
        "self_attn_weights.in_proj.weight": "onnx::MatMul_11184",
        "self_attn_weights.linear_pos.weight": "onnx::MatMul_11203",
        "feed_forward1.in_proj.weight": "onnx::MatMul_11210",
        "feed_forward1.out_proj.weight": "onnx::MatMul_11211",
        "nonlin_attention.in_proj.weight": "onnx::MatMul_11216",
        "nonlin_attention.out_proj.weight": "onnx::MatMul_11220",
        "self_attn1.in_proj.weight": "onnx::MatMul_11221",
        "self_attn1.out_proj.weight": "onnx::MatMul_11223",
        "conv_module1.in_proj.weight": "onnx::MatMul_11224",
        "conv_module1.out_proj.weight": "onnx::MatMul_11225",
        "feed_forward2.in_proj.weight": "onnx::MatMul_11226",
        "feed_forward2.out_proj.weight": "onnx::MatMul_11227",
        "self_attn2.in_proj.weight": "onnx::MatMul_11228",
        "self_attn2.out_proj.weight": "onnx::MatMul_11230",
        "conv_module2.in_proj.weight": "onnx::MatMul_11231",
        "conv_module2.out_proj.weight": "onnx::MatMul_11232",
        "feed_forward3.in_proj.weight": "onnx::MatMul_11233",
        "feed_forward3.out_proj.weight": "onnx::MatMul_11234",
    },
    1: {
        "self_attn_weights.in_proj.weight": "onnx::MatMul_11235",
        "self_attn_weights.linear_pos.weight": "onnx::MatMul_11254",
        "feed_forward1.in_proj.weight": "onnx::MatMul_11261",
        "feed_forward1.out_proj.weight": "onnx::MatMul_11262",
        "nonlin_attention.in_proj.weight": "onnx::MatMul_11267",
        "nonlin_attention.out_proj.weight": "onnx::MatMul_11271",
        "self_attn1.in_proj.weight": "onnx::MatMul_11272",
        "self_attn1.out_proj.weight": "onnx::MatMul_11274",
        "conv_module1.in_proj.weight": "onnx::MatMul_11275",
        "conv_module1.out_proj.weight": "onnx::MatMul_11276",
        "feed_forward2.in_proj.weight": "onnx::MatMul_11277",
        "feed_forward2.out_proj.weight": "onnx::MatMul_11278",
        "self_attn2.in_proj.weight": "onnx::MatMul_11279",
        "self_attn2.out_proj.weight": "onnx::MatMul_11281",
        "conv_module2.in_proj.weight": "onnx::MatMul_11282",
        "conv_module2.out_proj.weight": "onnx::MatMul_11283",
        "feed_forward3.in_proj.weight": "onnx::MatMul_11284",
        "feed_forward3.out_proj.weight": "onnx::MatMul_11285",
    },
}

STAGE1_LAYER0_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11301",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11320",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11327",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11328",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11333",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11337",
    "self_attn1.in_proj.weight": "onnx::MatMul_11338",
    "self_attn1.out_proj.weight": "onnx::MatMul_11340",
    "conv_module1.in_proj.weight": "onnx::MatMul_11341",
    "conv_module1.out_proj.weight": "onnx::MatMul_11342",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11343",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11344",
    "self_attn2.in_proj.weight": "onnx::MatMul_11345",
    "self_attn2.out_proj.weight": "onnx::MatMul_11347",
    "conv_module2.in_proj.weight": "onnx::MatMul_11348",
    "conv_module2.out_proj.weight": "onnx::MatMul_11349",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11350",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11351",
}

STAGE1_LAYER1_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11352",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11371",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11378",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11379",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11384",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11388",
    "self_attn1.in_proj.weight": "onnx::MatMul_11389",
    "self_attn1.out_proj.weight": "onnx::MatMul_11391",
    "conv_module1.in_proj.weight": "onnx::MatMul_11392",
    "conv_module1.out_proj.weight": "onnx::MatMul_11393",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11394",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11395",
    "self_attn2.in_proj.weight": "onnx::MatMul_11396",
    "self_attn2.out_proj.weight": "onnx::MatMul_11398",
    "conv_module2.in_proj.weight": "onnx::MatMul_11399",
    "conv_module2.out_proj.weight": "onnx::MatMul_11400",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11401",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11402",
}

STAGE2_LAYER0_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11422",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11441",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11448",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11449",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11454",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11458",
    "self_attn1.in_proj.weight": "onnx::MatMul_11459",
    "self_attn1.out_proj.weight": "onnx::MatMul_11461",
    "conv_module1.in_proj.weight": "onnx::MatMul_11462",
    "conv_module1.out_proj.weight": "onnx::MatMul_11463",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11464",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11465",
    "self_attn2.in_proj.weight": "onnx::MatMul_11466",
    "self_attn2.out_proj.weight": "onnx::MatMul_11468",
    "conv_module2.in_proj.weight": "onnx::MatMul_11469",
    "conv_module2.out_proj.weight": "onnx::MatMul_11470",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11471",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11472",
}

STAGE2_LAYER1_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11473",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11492",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11499",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11500",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11505",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11509",
    "self_attn1.in_proj.weight": "onnx::MatMul_11510",
    "self_attn1.out_proj.weight": "onnx::MatMul_11512",
    "conv_module1.in_proj.weight": "onnx::MatMul_11513",
    "conv_module1.out_proj.weight": "onnx::MatMul_11514",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11515",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11516",
    "self_attn2.in_proj.weight": "onnx::MatMul_11517",
    "self_attn2.out_proj.weight": "onnx::MatMul_11519",
    "conv_module2.in_proj.weight": "onnx::MatMul_11520",
    "conv_module2.out_proj.weight": "onnx::MatMul_11521",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11522",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11523",
}

STAGE2_LAYER2_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11524",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11543",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11550",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11551",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11556",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11560",
    "self_attn1.in_proj.weight": "onnx::MatMul_11561",
    "self_attn1.out_proj.weight": "onnx::MatMul_11563",
    "conv_module1.in_proj.weight": "onnx::MatMul_11564",
    "conv_module1.out_proj.weight": "onnx::MatMul_11565",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11566",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11567",
    "self_attn2.in_proj.weight": "onnx::MatMul_11568",
    "self_attn2.out_proj.weight": "onnx::MatMul_11570",
    "conv_module2.in_proj.weight": "onnx::MatMul_11571",
    "conv_module2.out_proj.weight": "onnx::MatMul_11572",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11573",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11574",
}

STAGE3_LAYER0_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11594",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11613",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11620",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11621",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11626",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11630",
    "self_attn1.in_proj.weight": "onnx::MatMul_11631",
    "self_attn1.out_proj.weight": "onnx::MatMul_11633",
    "conv_module1.in_proj.weight": "onnx::MatMul_11634",
    "conv_module1.out_proj.weight": "onnx::MatMul_11635",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11636",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11637",
    "self_attn2.in_proj.weight": "onnx::MatMul_11638",
    "self_attn2.out_proj.weight": "onnx::MatMul_11640",
    "conv_module2.in_proj.weight": "onnx::MatMul_11641",
    "conv_module2.out_proj.weight": "onnx::MatMul_11642",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11643",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11644",
}

STAGE3_LAYER1_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11645",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11664",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11671",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11672",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11677",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11681",
    "self_attn1.in_proj.weight": "onnx::MatMul_11682",
    "self_attn1.out_proj.weight": "onnx::MatMul_11684",
    "conv_module1.in_proj.weight": "onnx::MatMul_11685",
    "conv_module1.out_proj.weight": "onnx::MatMul_11686",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11687",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11688",
    "self_attn2.in_proj.weight": "onnx::MatMul_11689",
    "self_attn2.out_proj.weight": "onnx::MatMul_11691",
    "conv_module2.in_proj.weight": "onnx::MatMul_11692",
    "conv_module2.out_proj.weight": "onnx::MatMul_11693",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11694",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11695",
}

STAGE3_LAYER2_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11696",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11715",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11722",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11723",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11728",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11732",
    "self_attn1.in_proj.weight": "onnx::MatMul_11733",
    "self_attn1.out_proj.weight": "onnx::MatMul_11735",
    "conv_module1.in_proj.weight": "onnx::MatMul_11736",
    "conv_module1.out_proj.weight": "onnx::MatMul_11737",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11738",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11739",
    "self_attn2.in_proj.weight": "onnx::MatMul_11740",
    "self_attn2.out_proj.weight": "onnx::MatMul_11742",
    "conv_module2.in_proj.weight": "onnx::MatMul_11743",
    "conv_module2.out_proj.weight": "onnx::MatMul_11744",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11745",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11746",
}

STAGE3_LAYER3_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11747",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11766",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11773",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11774",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11779",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11783",
    "self_attn1.in_proj.weight": "onnx::MatMul_11784",
    "self_attn1.out_proj.weight": "onnx::MatMul_11786",
    "conv_module1.in_proj.weight": "onnx::MatMul_11787",
    "conv_module1.out_proj.weight": "onnx::MatMul_11788",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11789",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11790",
    "self_attn2.in_proj.weight": "onnx::MatMul_11791",
    "self_attn2.out_proj.weight": "onnx::MatMul_11793",
    "conv_module2.in_proj.weight": "onnx::MatMul_11794",
    "conv_module2.out_proj.weight": "onnx::MatMul_11795",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11796",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11797",
}

STAGE4_LAYER0_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11817",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11836",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11843",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11844",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11849",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11853",
    "self_attn1.in_proj.weight": "onnx::MatMul_11854",
    "self_attn1.out_proj.weight": "onnx::MatMul_11856",
    "conv_module1.in_proj.weight": "onnx::MatMul_11857",
    "conv_module1.out_proj.weight": "onnx::MatMul_11858",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11859",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11860",
    "self_attn2.in_proj.weight": "onnx::MatMul_11861",
    "self_attn2.out_proj.weight": "onnx::MatMul_11863",
    "conv_module2.in_proj.weight": "onnx::MatMul_11864",
    "conv_module2.out_proj.weight": "onnx::MatMul_11865",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11866",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11867",
}

STAGE4_LAYER1_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11868",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11887",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11894",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11895",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11900",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11904",
    "self_attn1.in_proj.weight": "onnx::MatMul_11905",
    "self_attn1.out_proj.weight": "onnx::MatMul_11907",
    "conv_module1.in_proj.weight": "onnx::MatMul_11908",
    "conv_module1.out_proj.weight": "onnx::MatMul_11909",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11910",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11911",
    "self_attn2.in_proj.weight": "onnx::MatMul_11912",
    "self_attn2.out_proj.weight": "onnx::MatMul_11914",
    "conv_module2.in_proj.weight": "onnx::MatMul_11915",
    "conv_module2.out_proj.weight": "onnx::MatMul_11916",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11917",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11918",
}

STAGE4_LAYER2_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11919",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11938",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11945",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11946",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11951",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11955",
    "self_attn1.in_proj.weight": "onnx::MatMul_11956",
    "self_attn1.out_proj.weight": "onnx::MatMul_11958",
    "conv_module1.in_proj.weight": "onnx::MatMul_11959",
    "conv_module1.out_proj.weight": "onnx::MatMul_11960",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11961",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11962",
    "self_attn2.in_proj.weight": "onnx::MatMul_11963",
    "self_attn2.out_proj.weight": "onnx::MatMul_11965",
    "conv_module2.in_proj.weight": "onnx::MatMul_11966",
    "conv_module2.out_proj.weight": "onnx::MatMul_11967",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11968",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11969",
}

STAGE5_LAYER0_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11989",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_12008",
    "feed_forward1.in_proj.weight": "onnx::MatMul_12015",
    "feed_forward1.out_proj.weight": "onnx::MatMul_12016",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_12021",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_12025",
    "self_attn1.in_proj.weight": "onnx::MatMul_12026",
    "self_attn1.out_proj.weight": "onnx::MatMul_12028",
    "conv_module1.in_proj.weight": "onnx::MatMul_12029",
    "conv_module1.out_proj.weight": "onnx::MatMul_12030",
    "feed_forward2.in_proj.weight": "onnx::MatMul_12031",
    "feed_forward2.out_proj.weight": "onnx::MatMul_12032",
    "self_attn2.in_proj.weight": "onnx::MatMul_12033",
    "self_attn2.out_proj.weight": "onnx::MatMul_12035",
    "conv_module2.in_proj.weight": "onnx::MatMul_12036",
    "conv_module2.out_proj.weight": "onnx::MatMul_12037",
    "feed_forward3.in_proj.weight": "onnx::MatMul_12038",
    "feed_forward3.out_proj.weight": "onnx::MatMul_12039",
}

STAGE5_LAYER1_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_12040",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_12059",
    "feed_forward1.in_proj.weight": "onnx::MatMul_12066",
    "feed_forward1.out_proj.weight": "onnx::MatMul_12067",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_12072",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_12076",
    "self_attn1.in_proj.weight": "onnx::MatMul_12077",
    "self_attn1.out_proj.weight": "onnx::MatMul_12079",
    "conv_module1.in_proj.weight": "onnx::MatMul_12080",
    "conv_module1.out_proj.weight": "onnx::MatMul_12081",
    "feed_forward2.in_proj.weight": "onnx::MatMul_12082",
    "feed_forward2.out_proj.weight": "onnx::MatMul_12083",
    "self_attn2.in_proj.weight": "onnx::MatMul_12084",
    "self_attn2.out_proj.weight": "onnx::MatMul_12086",
    "conv_module2.in_proj.weight": "onnx::MatMul_12087",
    "conv_module2.out_proj.weight": "onnx::MatMul_12088",
    "feed_forward3.in_proj.weight": "onnx::MatMul_12089",
    "feed_forward3.out_proj.weight": "onnx::MatMul_12090",
}

DIRECT_SUFFIXES = [
    "self_attn_weights.in_proj.bias",
    "feed_forward1.in_proj.bias",
    "feed_forward1.out_proj.bias",
    "nonlin_attention.in_proj.bias",
    "nonlin_attention.out_proj.bias",
    "self_attn1.in_proj.bias",
    "self_attn1.out_proj.bias",
    "conv_module1.in_proj.bias",
    "conv_module1.depthwise_conv.weight",
    "conv_module1.depthwise_conv.bias",
    "conv_module1.out_proj.bias",
    "feed_forward2.in_proj.bias",
    "feed_forward2.out_proj.bias",
    "bypass_mid.bypass_scale",
    "self_attn2.in_proj.bias",
    "self_attn2.out_proj.bias",
    "conv_module2.in_proj.bias",
    "conv_module2.depthwise_conv.weight",
    "conv_module2.depthwise_conv.bias",
    "conv_module2.out_proj.bias",
    "feed_forward3.in_proj.bias",
    "feed_forward3.out_proj.bias",
    "norm.log_scale",
    "norm.bias",
    "bypass.bypass_scale",
]


def add_encoder_stage_layer(stage: int, layer: int, matmuls: dict[str, str]) -> None:
    for dst_suffix, src_name in matmuls.items():
        NAME_MAP[src_name] = f"encoder.stage{stage}.layer{layer}.{dst_suffix}"
    for suffix in DIRECT_SUFFIXES:
        NAME_MAP[f"encoder.encoders.{stage}.encoder.layers.{layer}.{suffix}"] = (
            f"encoder.stage{stage}.layer{layer}.{suffix}"
        )


def add_stage0_layer(layer_index: int) -> None:
    prefix = f"encoder.stage0.layer{layer_index}"
    source_prefix = f"encoder.encoders.0.layers.{layer_index}"
    matmuls = STAGE0_LAYER_MATMULS[layer_index]

    for dst_suffix, src_name in matmuls.items():
        NAME_MAP[src_name] = f"{prefix}.{dst_suffix}"

    direct_suffixes = [
        "self_attn_weights.in_proj.bias",
        "feed_forward1.in_proj.bias",
        "feed_forward1.out_proj.bias",
        "nonlin_attention.in_proj.bias",
        "nonlin_attention.out_proj.bias",
        "self_attn1.in_proj.bias",
        "self_attn1.out_proj.bias",
        "conv_module1.in_proj.bias",
        "conv_module1.depthwise_conv.weight",
        "conv_module1.depthwise_conv.bias",
        "conv_module1.out_proj.bias",
        "feed_forward2.in_proj.bias",
        "feed_forward2.out_proj.bias",
        "bypass_mid.bypass_scale",
        "self_attn2.in_proj.bias",
        "self_attn2.out_proj.bias",
        "conv_module2.in_proj.bias",
        "conv_module2.depthwise_conv.weight",
        "conv_module2.depthwise_conv.bias",
        "conv_module2.out_proj.bias",
        "feed_forward3.in_proj.bias",
        "feed_forward3.out_proj.bias",
        "norm.log_scale",
        "norm.bias",
        "bypass.bypass_scale",
    ]
    for suffix in direct_suffixes:
        NAME_MAP[f"{source_prefix}.{suffix}"] = f"{prefix}.{suffix}"


for stage0_layer_index in range(2):
    add_stage0_layer(stage0_layer_index)

NAME_MAP["onnx::Mul_11296"] = "encoder.stage1.downsample.weights"

for dst_suffix, src_name in STAGE1_LAYER0_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage1.layer0.{dst_suffix}"

for suffix in [
    "self_attn_weights.in_proj.bias",
    "feed_forward1.in_proj.bias",
    "feed_forward1.out_proj.bias",
    "nonlin_attention.in_proj.bias",
    "nonlin_attention.out_proj.bias",
    "self_attn1.in_proj.bias",
    "self_attn1.out_proj.bias",
    "conv_module1.in_proj.bias",
    "conv_module1.depthwise_conv.weight",
    "conv_module1.depthwise_conv.bias",
    "conv_module1.out_proj.bias",
    "feed_forward2.in_proj.bias",
    "feed_forward2.out_proj.bias",
    "bypass_mid.bypass_scale",
    "self_attn2.in_proj.bias",
    "self_attn2.out_proj.bias",
    "conv_module2.in_proj.bias",
    "conv_module2.depthwise_conv.weight",
    "conv_module2.depthwise_conv.bias",
    "conv_module2.out_proj.bias",
    "feed_forward3.in_proj.bias",
    "feed_forward3.out_proj.bias",
    "norm.log_scale",
    "norm.bias",
    "bypass.bypass_scale",
]:
    NAME_MAP[f"encoder.encoders.1.encoder.layers.0.{suffix}"] = (
        f"encoder.stage1.layer0.{suffix}"
    )

for dst_suffix, src_name in STAGE1_LAYER1_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage1.layer1.{dst_suffix}"

for suffix in [
    "self_attn_weights.in_proj.bias",
    "feed_forward1.in_proj.bias",
    "feed_forward1.out_proj.bias",
    "nonlin_attention.in_proj.bias",
    "nonlin_attention.out_proj.bias",
    "self_attn1.in_proj.bias",
    "self_attn1.out_proj.bias",
    "conv_module1.in_proj.bias",
    "conv_module1.depthwise_conv.weight",
    "conv_module1.depthwise_conv.bias",
    "conv_module1.out_proj.bias",
    "feed_forward2.in_proj.bias",
    "feed_forward2.out_proj.bias",
    "bypass_mid.bypass_scale",
    "self_attn2.in_proj.bias",
    "self_attn2.out_proj.bias",
    "conv_module2.in_proj.bias",
    "conv_module2.depthwise_conv.weight",
    "conv_module2.depthwise_conv.bias",
    "conv_module2.out_proj.bias",
    "feed_forward3.in_proj.bias",
    "feed_forward3.out_proj.bias",
    "norm.log_scale",
    "norm.bias",
    "bypass.bypass_scale",
]:
    NAME_MAP[f"encoder.encoders.1.encoder.layers.1.{suffix}"] = (
        f"encoder.stage1.layer1.{suffix}"
    )

NAME_MAP["encoder.encoders.1.out_combiner.bypass_scale"] = "encoder.stage1.out_combiner.bypass_scale"

NAME_MAP["onnx::Mul_11417"] = "encoder.stage2.downsample.weights"

for dst_suffix, src_name in STAGE2_LAYER0_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage2.layer0.{dst_suffix}"
for dst_suffix, src_name in STAGE2_LAYER1_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage2.layer1.{dst_suffix}"
for dst_suffix, src_name in STAGE2_LAYER2_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage2.layer2.{dst_suffix}"

for layer_index in range(3):
    for suffix in [
        "self_attn_weights.in_proj.bias",
        "feed_forward1.in_proj.bias",
        "feed_forward1.out_proj.bias",
        "nonlin_attention.in_proj.bias",
        "nonlin_attention.out_proj.bias",
        "self_attn1.in_proj.bias",
        "self_attn1.out_proj.bias",
        "conv_module1.in_proj.bias",
        "conv_module1.depthwise_conv.weight",
        "conv_module1.depthwise_conv.bias",
        "conv_module1.out_proj.bias",
        "feed_forward2.in_proj.bias",
        "feed_forward2.out_proj.bias",
        "bypass_mid.bypass_scale",
        "self_attn2.in_proj.bias",
        "self_attn2.out_proj.bias",
        "conv_module2.in_proj.bias",
        "conv_module2.depthwise_conv.weight",
        "conv_module2.depthwise_conv.bias",
        "conv_module2.out_proj.bias",
        "feed_forward3.in_proj.bias",
        "feed_forward3.out_proj.bias",
        "norm.log_scale",
        "norm.bias",
        "bypass.bypass_scale",
    ]:
        NAME_MAP[f"encoder.encoders.2.encoder.layers.{layer_index}.{suffix}"] = (
            f"encoder.stage2.layer{layer_index}.{suffix}"
        )

NAME_MAP["encoder.encoders.2.out_combiner.bypass_scale"] = "encoder.stage2.out_combiner.bypass_scale"

NAME_MAP["onnx::Mul_11589"] = "encoder.stage3.downsample.weights"
add_encoder_stage_layer(3, 0, STAGE3_LAYER0_MATMULS)
add_encoder_stage_layer(3, 1, STAGE3_LAYER1_MATMULS)
add_encoder_stage_layer(3, 2, STAGE3_LAYER2_MATMULS)
add_encoder_stage_layer(3, 3, STAGE3_LAYER3_MATMULS)
NAME_MAP["encoder.encoders.3.out_combiner.bypass_scale"] = "encoder.stage3.out_combiner.bypass_scale"

NAME_MAP["onnx::Mul_11812"] = "encoder.stage4.downsample.weights"
add_encoder_stage_layer(4, 0, STAGE4_LAYER0_MATMULS)
add_encoder_stage_layer(4, 1, STAGE4_LAYER1_MATMULS)
add_encoder_stage_layer(4, 2, STAGE4_LAYER2_MATMULS)
NAME_MAP["encoder.encoders.4.out_combiner.bypass_scale"] = "encoder.stage4.out_combiner.bypass_scale"

NAME_MAP["onnx::Mul_11984"] = "encoder.stage5.downsample.weights"
add_encoder_stage_layer(5, 0, STAGE5_LAYER0_MATMULS)
add_encoder_stage_layer(5, 1, STAGE5_LAYER1_MATMULS)
NAME_MAP["encoder.encoders.5.out_combiner.bypass_scale"] = "encoder.stage5.out_combiner.bypass_scale"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ZIPA ONNX initializers into a safetensors file for MLX loading."
    )
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.onnx_model)
    tensors = {}
    missing = []

    for initializer in model.graph.initializer:
        mapped = NAME_MAP.get(initializer.name)
        if mapped is None:
            continue
        tensors[mapped] = numpy_helper.to_array(initializer)

    for source_name in NAME_MAP:
        if NAME_MAP[source_name] not in tensors:
            missing.append(source_name)

    if missing:
        raise SystemExit(f"missing required initializers: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    print(f"wrote {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
