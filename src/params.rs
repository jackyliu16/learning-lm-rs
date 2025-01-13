use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| match safetensor.tensor(name) {
            Ok(data) => {
                assert!(data.dtype() == Dtype::F32);

                let new_data = data
                    .data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                Tensor::<f32>::new(new_data, &Vec::from(data.shape()))
            }
            Err(e) => Tensor::default(&Vec::new()),
        };

        let for_each_hidden_layers = |name: &str| {
            (0..config.num_hidden_layers)
                .into_iter()
                .map(|i| format!("model.layers.{i}.{name}"))
                .map(|name| get_tensor(&name))
                .collect()
        };

        dbg!(config.tie_word_embeddings);

        // NOTE: tie_encoder_decoder (bool, optional, defaults to False) — Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder and decoder model to have the exact same parameter names.
        let embedding_table = if config.tie_word_embeddings {
            // 由于 lm_head 与 embed_token 相同，因此需要将他们指向同一个张量
            get_tensor("lm_head.weight") // 深度学习中，最后一层一般被称作为 HEAD
        } else {
            get_tensor("model.embed_tokens.weight") // 输入嵌入层 (Input Embedding Layer)
        };

        // NOTE: 我也不知道对不对，这个是根据 model.safetensor 分析写的
        LLamaParams {
            embedding_table,
            rms_att_w: for_each_hidden_layers("input_layernorm.weight"),
            wk: for_each_hidden_layers("self_attn.k_proj.weight"),
            wo: for_each_hidden_layers("self_attn.o_proj.weight"),
            wq: for_each_hidden_layers("self_attn.q_proj.weight"),
            wv: for_each_hidden_layers("self_attn.v_proj.weight"),

            rms_ffn_w: for_each_hidden_layers("post_attention_layernorm.weight"),
            w_down: for_each_hidden_layers("mlp.down_proj.weight"),
            w_gate: for_each_hidden_layers("mlp.gate_proj.weight"),
            w_up: for_each_hidden_layers("mlp.up_proj.weight"),

            lm_head: get_tensor("lm_head.weight"),
            rms_out_w: get_tensor("model.norm.weight"),
        }
    }
}
