// recreating GPT-2 in one file
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde_json;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::fs::File;

use ndarray::{array, concatenate, s, Array1, Array2, Axis};

// String -> Encoded IDs

// Encoded IDs -> Embeddings

struct Tokenizer {
    bytes_to_unicode: HashMap<u8, char>,
    unicode_to_bytes: HashMap<char, u8>,
    token_to_id: HashMap<String, u32>,
    id_to_token: HashMap<u32, String>,
}

impl Tokenizer {
    fn create_bytes_unicode_map() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut bytes_to_unicode = HashMap::new();

        // you can see the ranges at https://www.utf8-chartable.de/
        let ascii_chars = '!' as u32..='~' as u32;
        let latin_1_supplement_chars = '¡' as u32..='¬' as u32;
        let latin_extended_a_chars = '®' as u32..='ÿ' as u32;

        let mut bytes: Vec<u32> = ascii_chars
            .chain(latin_1_supplement_chars)
            .chain(latin_extended_a_chars)
            .collect();
        let mut unicode_values: Vec<u32> = bytes.clone();

        // we want to map every byte to a unicode value
        // we set missing bytes to unicode values outside of the first 255
        let mut n = 0;
        for byte in 0..256 {
            if !bytes.contains(&byte) {
                bytes.push(byte);
                unicode_values.push(256 + n);
                n += 1;
            }
        }

        let unicode_chars: Vec<char> = unicode_values
            .into_iter()
            .map(|x| char::from_u32(x).unwrap())
            .collect();

        for (byte, unicode_char) in bytes.iter().zip(unicode_chars.iter()) {
            bytes_to_unicode.insert(*byte as u8, *unicode_char);
        }

        let unicode_to_bytes: HashMap<char, u8> = bytes_to_unicode
            .iter()
            .map(|(byte, unicode_char)| (*unicode_char, *byte))
            .collect();

        (bytes_to_unicode, unicode_to_bytes)
    }

    fn create_token_id_map() -> std::io::Result<(HashMap<String, u32>, HashMap<u32, String>)> {
        let file = File::open("src/weights/vocab.json")?;
        let reader = std::io::BufReader::new(file);
        let encoder: HashMap<String, u32> = serde_json::from_reader(reader)?;
        let decoder: HashMap<u32, String> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        Ok((encoder, decoder))
    }

    fn new() -> Tokenizer {
        let (bytes_to_unicode, unicode_to_bytes) = Tokenizer::create_bytes_unicode_map();
        let (token_to_id, id_to_token) = Tokenizer::create_token_id_map().unwrap();

        Tokenizer {
            bytes_to_unicode,
            unicode_to_bytes,
            token_to_id,
            id_to_token,
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let text_as_unicode: String = text.bytes().map(|x| self.bytes_to_unicode[&x]).collect();
        let mut text_as_unicode_slice = text_as_unicode.as_str();

        let mut encoded: Vec<u32> = Vec::new();
        while !text_as_unicode_slice.is_empty() {
            let best_token = self
                .token_to_id
                .keys()
                .filter(|&token| text_as_unicode_slice.starts_with(token))
                .max_by_key(|token| token.len())
                .unwrap_or_else(|| panic!("No token found for {:?}", text_as_unicode_slice));

            let token_id = self.token_to_id[best_token];
            encoded.push(token_id);
            text_as_unicode_slice = &text_as_unicode_slice[best_token.len()..];
        }

        encoded
    }

    fn decode(&self, ids: &[u32]) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(
            ids.iter()
                .flat_map(|id| self.id_to_token[id].chars())
                .map(|c| self.unicode_to_bytes[&c])
                .collect(),
        )
    }
}

// struct to store bias and weight
struct LayerParams {
    weight: Array2<f32>,
    bias: Array1<f32>,
}

struct SelfAttentionParams {
    // maps embedding sequence to Q, K, V
    // Q, K, V dimension is 768 / 12 heads = 64
    // size: 768 x (64 * 12 * 3)
    c_attn: LayerParams,
    // applied on concatenated self-attention results
    // size: 768 x 768
    c_proj: LayerParams,
}

// MLP stands for Multi-Layer Perceptron
// Applied after self-attention
struct MLPParams {
    // maps self-attention output to larger intermediate dimension, fully connected
    // size: 768 x (768 * 4)
    c_fc: LayerParams,
    // maps intermediate dimension back to embedding dimension
    // size: 3072 x 768
    c_proj: LayerParams,
}

// Note on the layer norm: for GPT-1, was applied after self-attention and MLP
// For GPT-2, was applied before self-attention and MLP, and there's a final layer norm at the end
// Skip connection comes before layer norm
struct LayerNormParams {
    // size: 768
    // actually is gain, but called weight in the safetensors file
    weight: Array1<f32>,
    // size: 768
    bias: Array1<f32>,
}

// Parameters for a single block
struct BlockParams {
    attn: SelfAttentionParams,
    mlp: MLPParams,
    ln_1: LayerNormParams, // before self-attention
    ln_2: LayerNormParams, // before MLP
}

struct GPTWeights {
    wte: Array2<f32>, // token embeddings. the transpose is used for final linear projection ("weight tying")
    wpe: Array2<f32>, // position embeddings
    h: Vec<BlockParams>, // blocks
    ln_f: LayerNormParams, // final layer norm
}

struct GPT {
    vocab_size: u32,
    context_length: u32,
    embedding_dim: u32,
    num_heads: u32,
    num_layers: u32,
    weights: GPTWeights,
}

fn bytes_to_matrix(data: &[u8], shape: &[usize]) -> Array2<f32> {
    Array2::from_shape_vec(
        (shape[0], shape[1]),
        data.chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    )
    .unwrap()
}

fn bytes_to_vector(data: &[u8], shape: &[usize]) -> Array1<f32> {
    Array1::from_shape_vec(
        shape[0],
        data.chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
    )
    .unwrap()
}

fn tensor_to_matrix(tensors: &SafeTensors, name: &str) -> Array2<f32> {
    println!("tensor name: {}", name);
    let tensor = tensors.tensor(name).unwrap();
    let data = tensor.data();
    let shape = tensor.shape();

    bytes_to_matrix(data, shape)
}

fn tensor_to_vector(tensors: &SafeTensors, name: &str) -> Array1<f32> {
    let tensor = tensors.tensor(name).unwrap();
    let data = tensor.data();
    let shape = tensor.shape();

    bytes_to_vector(data, shape)
}

// Default: 124M parameter model
impl Default for GPT {
    fn default() -> Self {
        let vocab_size = 50257;
        let context_length = 1024;
        let embedding_dim = 768;
        let num_heads = 12;
        let num_layers = 12;

        // load tensors
        let file = File::open("src/weights/model.safetensors").unwrap();
        let weight_buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        let tensors = SafeTensors::deserialize(weight_buffer.as_ref()).unwrap();

        let wte = tensor_to_matrix(&tensors, "wte.weight");
        println!("Loaded wte");
        let wpe = tensor_to_matrix(&tensors, "wpe.weight");
        println!("Loaded wpe");
        let ln_f = LayerNormParams {
            bias: tensor_to_vector(&tensors, "ln_f.bias"),
            weight: tensor_to_vector(&tensors, "ln_f.weight"),
        };
        println!("Loaded ln_f");
        let h = (0..num_layers)
            .map(|i| BlockParams {
                attn: SelfAttentionParams {
                    c_attn: LayerParams {
                        bias: tensor_to_vector(&tensors, &format!("h.{}.attn.c_attn.bias", i)),
                        weight: tensor_to_matrix(&tensors, &format!("h.{}.attn.c_attn.weight", i)),
                    },
                    c_proj: LayerParams {
                        bias: tensor_to_vector(&tensors, &format!("h.{}.attn.c_proj.bias", i)),
                        weight: tensor_to_matrix(&tensors, &format!("h.{}.attn.c_proj.weight", i)),
                    },
                },
                mlp: MLPParams {
                    c_fc: LayerParams {
                        bias: tensor_to_vector(&tensors, &format!("h.{}.mlp.c_fc.bias", i)),
                        weight: tensor_to_matrix(&tensors, &format!("h.{}.mlp.c_fc.weight", i)),
                    },
                    c_proj: LayerParams {
                        bias: tensor_to_vector(&tensors, &format!("h.{}.mlp.c_proj.bias", i)),
                        weight: tensor_to_matrix(&tensors, &format!("h.{}.mlp.c_proj.weight", i)),
                    },
                },
                ln_1: LayerNormParams {
                    bias: tensor_to_vector(&tensors, &format!("h.{}.ln_1.bias", i)),
                    weight: tensor_to_vector(&tensors, &format!("h.{}.ln_1.weight", i)),
                },
                ln_2: LayerNormParams {
                    bias: tensor_to_vector(&tensors, &format!("h.{}.ln_2.bias", i)),
                    weight: tensor_to_vector(&tensors, &format!("h.{}.ln_2.weight", i)),
                },
            })
            .collect();
        println!("Loaded h");

        GPT {
            vocab_size,
            context_length,
            embedding_dim,
            num_heads,
            num_layers,
            weights: GPTWeights { wte, wpe, h, ln_f },
        }
    }
}

// Token IDs to embeddings and positional embeddings
fn initial_embeddings(token_ids: &Vec<u32>, wte: &Array2<f32>, wpe: &Array2<f32>) -> Array2<f32> {
    let mut embeddings = Array2::zeros((token_ids.len(), wte.shape()[1]));
    for (i, token_id) in token_ids.iter().enumerate() {
        embeddings
            .slice_mut(s![i, ..])
            .assign(&wte.slice(s![*token_id as usize, ..]));
    }
    embeddings + wpe.slice(s![..token_ids.len(), ..])
    // final dimension: n_seq x 768
}

struct QKV {
    q: Array2<f32>, // n_seq x 64
    k: Array2<f32>,
    v: Array2<f32>,
}

// transformer block
fn qkv_all_heads(x: &Array2<f32>, attn: &LayerParams, num_heads: u32) -> Vec<QKV> {
    // input should be n_seq x 768
    let qkv = linear(x, &attn.weight, &attn.bias);
    // n_seq * 64 * 12 * 3
    // split the result into q, k, v
    let qkv_dim = x.shape()[1] / num_heads as usize; // 64
    let qkv_chunks = qkv.axis_chunks_iter(Axis(1), qkv_dim).collect::<Vec<_>>(); // 36
    (0..num_heads as usize)
        .map(|i| {
            let q = qkv_chunks[i].to_owned();
            let k = qkv_chunks[i + num_heads as usize].to_owned();
            let v = qkv_chunks[i + 2 * num_heads as usize].to_owned();
            QKV { q, k, v }
        })
        .collect()
}

// Note: we are applying softmax over columns
// In other words, the sum of each row is 1
// We subtract the max value from each row to avoid numerical instability
fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let max_values = x
        .map_axis(Axis(1), |row| row.iter().fold(f32::MIN, |a, &b| a.max(b)))
        .insert_axis(Axis(1));
    let x_normalized = x - max_values;
    let e_x_normalized = x_normalized.mapv(f32::exp);
    let sums = e_x_normalized.sum_axis(Axis(1)).insert_axis(Axis(1));
    e_x_normalized / sums
}

fn masked_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    mask: &Array2<f32>,
) -> Array2<f32> {
    let sqrt_dk = (k.shape()[1] as f32).sqrt();
    let q_times_k = q.dot(&k.t()); // n_seq x n_seq
    let attention = q_times_k / sqrt_dk + mask;
    softmax(&attention).dot(v) // n_seq x 64
}

fn multi_head_attention(
    x: &Array2<f32>,
    attn: &LayerParams,
    proj: &LayerParams,
    num_heads: u32,
) -> Array2<f32> {
    let qkvs = qkv_all_heads(x, attn, num_heads);

    let mut mask = Array2::zeros((x.shape()[0], x.shape()[0])); // n_seq x n_seq
    for (row_idx, mut row) in mask.outer_iter_mut().enumerate() {
        for col in row_idx + 1..row.len() {
            row[col] = f32::MIN;
        }
    }

    let concat_attention = qkvs
        .iter()
        .map(|qkv| masked_attention(&qkv.q, &qkv.k, &qkv.v, &mask))
        .reduce(|a, b| concatenate![Axis(1), a, b])
        .unwrap(); // n_seq x 768

    linear(&concat_attention, &proj.weight, &proj.bias)
}

fn linear(x: &Array2<f32>, w: &Array2<f32>, b: &Array1<f32>) -> Array2<f32> {
    x.dot(w) + b
}

fn layer_norm(x: &Array2<f32>, gain: &Array1<f32>, bias: &Array1<f32>) -> Array2<f32> {
    let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1));
    let variance = x
        .std_axis(Axis(1), 0.0)
        .insert_axis(Axis(1))
        .mapv_into(|v| v.powf(2.0));
    let epsilon = 1e-5;
    gain * (x - mean) / (variance + epsilon).mapv_into(f32::sqrt) + bias
}

// Feed-forward Network
fn gelu(x: &Array2<f32>) -> Array2<f32> {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.mapv(|v| v.powf(3.0)))).mapv(f32::tanh))
}

fn feed_forward(x: &Array2<f32>, mlp_params: &MLPParams) -> Array2<f32> {
    let x_ff = linear(x, &mlp_params.c_fc.weight, &mlp_params.c_fc.bias);
    let x_gelu = gelu(&x_ff);
    linear(&x_gelu, &mlp_params.c_proj.weight, &mlp_params.c_proj.bias)
}

fn transformer_block(
    x: &Array2<f32>,
    attention_params: &SelfAttentionParams,
    mlp_params: &MLPParams,
    ln1_params: &LayerNormParams,
    ln2_params: &LayerNormParams,
    num_heads: u32,
) -> Array2<f32> {
    let x_ln1 = layer_norm(x, &ln1_params.weight, &ln1_params.bias);
    let x_mha = multi_head_attention(
        &x_ln1,
        &attention_params.c_attn,
        &attention_params.c_proj,
        num_heads,
    );
    let x_skip1 = x + x_mha;
    let x_ln2 = layer_norm(&x_skip1, &ln2_params.weight, &ln2_params.bias);
    let x_ff = feed_forward(&x_ln2, &mlp_params);

    x_skip1 + x_ff
}

fn run_gpt(token_ids: &Vec<u32>, gpt: &GPT) -> Array2<f32> {
    let initial_embeddings = initial_embeddings(token_ids, &gpt.weights.wte, &gpt.weights.wpe);
    let mut x = initial_embeddings;
    for block_params in gpt.weights.h.iter() {
        let attention_params = &block_params.attn;
        let mlp_params = &block_params.mlp;
        let ln1_params = &block_params.ln_1;
        let ln2_params = &block_params.ln_2;
        let num_heads = gpt.num_heads;

        x = transformer_block(
            &x,
            attention_params,
            mlp_params,
            ln1_params,
            ln2_params,
            num_heads,
        );
    }

    // final layer norm and linear projection
    layer_norm(&x, &gpt.weights.ln_f.weight, &gpt.weights.ln_f.bias).dot(&gpt.weights.wte.t())
    // n_seq x vocab_size
}

fn generate(token_ids: Vec<u32>, gpt: GPT, num_tokens: usize) -> Vec<u32> {
    let mut token_ids = token_ids;
    for _ in 0..num_tokens {
        let result = run_gpt(&token_ids, &gpt);
        let next_token_probs = result.index_axis(Axis(0), result.shape()[0] - 1);
        // index of largest element
        let next_token = next_token_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        let next_token_id = next_token.unwrap().0 as u32;

        println!("next_token_id: {:?}", next_token_id);

        token_ids.push(next_token_id);
    }
    token_ids
}

fn main() {
    let tokenizer = Tokenizer::new();
    let test = String::from("Hello, my name is");
    let encoded = tokenizer.encode(&test);

    let gpt = GPT::default();
    let result = generate(encoded, gpt, 20);
    let decoded = tokenizer.decode(&result).unwrap();
    println!("result: {:?}", decoded);
}
