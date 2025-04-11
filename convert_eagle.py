import os
import json
import safetensors.torch as safetensors
import torch


def get_lm_head_and_norm(base_model):
    safetensors_files = []
    for file in os.listdir(base_model):
        if file.endswith('.safetensors'):
            safetensors_files.append(os.path.join(base_model, file))
    safetensors_files.sort()
    
    weights = {}
    for i in [-2, -1]:
        weights.update(safetensors.load_file(safetensors_files[i]))
    lm_head_weight = weights['language_model.lm_head.weight']
    norm_weight = weights['language_model.model.norm.weight']
    return lm_head_weight, norm_weight

def split_qkv(config, qkv_weights):
    # q = layer.attention.wq.weight.view(40, 128, 10240)
    # k = layer.attention.wk.weight.view(8, 128, 10240)
    # v = layer.attention.wv.weight.view(8, 128, 10240)
    # linear_qkv = torch.cat((q, k, v), dim=0).view(7168, 10240)
    num_heads = config["num_attention_heads"]
    num_key_value_heads = config["num_key_value_heads"]
    head_dim = config["hidden_size"] // num_heads
    q_size = num_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    qkv_weights = torch.concat(qkv_weights, dim=0)
    q_weights = qkv_weights[:q_size, :]
    k_weights = qkv_weights[q_size:q_size+kv_size, :]
    v_weights = qkv_weights[q_size+kv_size:, :]
    return q_weights, k_weights, v_weights

def split_gate_up(config, gate_up_weights):
    intermediate_size=config["intermediate_size_mlp"]
    gate_up_weights = torch.concat(gate_up_weights, dim=0)
    gate_weights = gate_up_weights[:intermediate_size, :]
    up_weights = gate_up_weights[intermediate_size:, :]
    return gate_weights, up_weights
    
if __name__ == '__main__':
    pt_dir = '/home/scratch.yeyu_hw/shared_files/omni-128e-eagle3'
    pt_files = [f'{pt_dir}/rank_{i}.pt' for i in range(8)]
    base_model = '/home/scratch.crush_data/crush-maverick-17b-128e-instruct-hf-final_vv3/'
    eagle_dir = '/home/scratch.fanrongl_gpu/GitRepo/TRT_LLM/main/TensorRT-LLM/tmp/ckpts/omni-128e-eagle3'

    name_mapping = {
        "hnorm.weight": "midlayer.hidden_norm.weight",
        "fc.weight": "fc.weight",
        "fc.bias": "fc.bias",
        "decoder.layers.0.input_layernorm.weight": "midlayer.input_layernorm.weight",
        "decoder.layers.0.self_attention.linear_proj.weight": "midlayer.self_attn.o_proj.weight",
        "decoder.layers.0.self_attention.linear_qkv.weight": "midlayer.self_attn.qkv_proj.weight",
        "decoder.layers.0.pre_mlp_layernorm.weight": "midlayer.post_attention_layernorm.weight",
        "decoder.layers.0.mlp.linear_fc1.weight": "midlayer.mlp.gate_up_proj.weight",
        "decoder.layers.0.mlp.linear_fc2.weight": "midlayer.mlp.down_proj.weight",
    }

    tp_tensors = {"decoder.layers.0.self_attention.linear_proj.weight": 1,
                "decoder.layers.0.self_attention.linear_qkv.weight": 0,
                "decoder.layers.0.mlp.linear_fc1.weight": 0,
                "decoder.layers.0.mlp.linear_fc2.weight": 1,
                }
    if not os.path.exists(eagle_dir):
        os.makedirs(eagle_dir)

    # config
    with open(f"{base_model}/config.json", 'r') as file:
        config = json.load(file)
    config.update(config.pop("text_config", {}))
    config.pop("vision_config", None)
    config["draft_vocab_size"] = config["vocab_size"]
    config["architectures"] = ["LlamaForCausalLM"]
    config["bias"] = True
    with open(f"{eagle_dir}/config.json", 'w') as file:
        json.dump(config, file, indent=4)

    # weights
    all_params = {}
    for pt_file in pt_files:
        params = torch.load(pt_file, weights_only=False, map_location='cpu')
        for k in tp_tensors:
            if k in all_params:
                all_params[k].append(params[k])
            else:
                all_params[k] = [params[k]]
        
        # For non-TP tensors, just update directly
        for k, v in params.items():
            if k not in tp_tensors:
                all_params[k] = v

    eagle_params = {}
    for k,v in name_mapping.items():
        if isinstance(all_params[k], list):
            if 'linear_qkv' in k:
                q_weights, k_weights, v_weights = split_qkv(config, all_params[k])
                eagle_params["midlayer.self_attn.q_proj.weight"] = q_weights
                eagle_params["midlayer.self_attn.k_proj.weight"] = k_weights
                eagle_params["midlayer.self_attn.v_proj.weight"] = v_weights
            elif 'linear_fc1' in k:
                gate_weights, up_weights = split_gate_up(config, all_params[k])
                eagle_params["midlayer.mlp.gate_proj.weight"] = gate_weights
                eagle_params["midlayer.mlp.up_proj.weight"] = up_weights
            else:
                eagle_params[v] = torch.concat(all_params[k], dim=tp_tensors[k])
        else:
            eagle_params[v] = all_params[k]
    
    lm_head_weight, norm_weight = get_lm_head_and_norm(base_model)
    eagle_params['norm.weight'] = norm_weight
    eagle_params['lm_head.weight'] = lm_head_weight
    torch.save(eagle_params, f'{eagle_dir}/pytorch_model.bin')
        
    # eagle_dir = '/home/scratch.trt_llm_data/llm-models/EAGLE3-LLaMA3.1-Instruct-8B/'
    # # eagle_dir = '/home/scratch.fanrongl_gpu/GitRepo/TRT_LLM/main/TensorRT-LLM/tmp/ckpts/omni-128e-eagle3'
    # weights = torch.load(f'{eagle_dir}/pytorch_model.bin', map_location='cpu')
    # for k,v in weights.items():
    #     print(f"{k}: {v.shape}")
