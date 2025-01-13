use serde;
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct LlamaConfigJson {
    /// 起始符 token id
    pub bos_token_id: u32,
    /// 结束符 token id
    pub eos_token_id: u32,
    /// 隐藏层大小
    pub hidden_size: usize,
    /// 中间层大小
    pub intermediate_size: usize,
    /// 最大序列长度
    pub max_position_embeddings: usize,
    /// Self-Attention K，V 头数
    pub num_attention_heads: usize,
    /// 隐藏层数
    pub num_hidden_layers: usize,
    /// Self-Attention epsilon 参数
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// RoPE 的 theta 参数
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// 模型数据类型
    pub torch_dtype: String,
    /// 起始和结束 embedding 参数矩阵是否共享同一份数据
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
}

#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}
