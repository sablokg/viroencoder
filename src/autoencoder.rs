use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::{Linear, LinearConfig};
use burn::optim::{GradientsParams, Optimizer, SgdConfig};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Module, Debug)]
pub struct DnaAutoEncoder<B: Backend> {
    conv_encoder: Conv1d<B>,
    fc_encoder: Linear<B>,
    fc_decoder: Linear<B>,
    conv_decoder: Conv1d<B>,
    output_layer: Conv1d<B>,
}

impl<B: Backend> DnaAutoEncoder<B> {
    pub fn new(device: &B::Device, seq_len: usize) -> Self {
        let channels = 4;
        let hidden_channels = 8;
        let compressed_dim = 16;
        Self {
            conv_encoder: Conv1dConfig::new(channels, hidden_channels, 3)
                .with_padding(burn::nn::PaddingConfig1d::Same)
                .init(device),
            fc_encoder: LinearConfig::new(hidden_channels * seq_len, compressed_dim).init(device),
            fc_decoder: LinearConfig::new(compressed_dim, hidden_channels * seq_len).init(device),
            conv_decoder: Conv1dConfig::new(hidden_channels, hidden_channels, 3)
                .with_padding(burn::nn::PaddingConfig1d::Same)
                .init(device),
            output_layer: Conv1dConfig::new(hidden_channels, channels, 3)
                .with_padding(burn::nn::PaddingConfig1d::Same)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, _in_channels, _seq_len] = input.dims();

        let x = self.conv_encoder.forward(input);
        let x = burn::tensor::activation::relu(x);

        // Use the ACTUAL post-conv channel/length dims, not the input's,
        // since hidden_channels != input channels.
        let [_, enc_channels, enc_len] = x.dims();
        let x = x.reshape([batch_size, enc_channels * enc_len]);

        let latent = self.fc_encoder.forward(x);
        let latent = burn::tensor::activation::relu(latent);

        let x = self.fc_decoder.forward(latent);
        let x = burn::tensor::activation::relu(x);

        let x = x.reshape([batch_size, enc_channels, enc_len]);
        let x = self.conv_decoder.forward(x);
        let x = burn::tensor::activation::relu(x);

        let logits = self.output_layer.forward(x);
        burn::tensor::activation::softmax(logits, 1)
    }
}
pub fn one_hot_encode(input: &[String]) -> Vec<[Vec<f32>; 4]> {
    input
        .iter()
        .map(|seq| {
            let mut channels: [Vec<f32>; 4] = Default::default();
            for c in seq.chars() {
                let onehot = match c.to_ascii_uppercase() {
                    'A' => [1.0, 0.0, 0.0, 0.0],
                    'T' => [0.0, 1.0, 0.0, 0.0],
                    'G' => [0.0, 0.0, 1.0, 0.0],
                    'C' => [0.0, 0.0, 0.0, 1.0],
                    'N' => [0.0, 0.0, 0.0, 0.0],
                    _ => continue,
                };
                for ch in 0..4 {
                    channels[ch].push(onehot[ch]);
                }
            }
            channels
        })
        .collect()
}

/// Pads every sequence's per-channel vectors to the length of the longest
/// sequence, returning the padded data and the common sequence length.
pub fn pad_sequences(sequences: &[[Vec<f32>; 4]]) -> (Vec<[Vec<f32>; 4]>, usize) {
    let max_len = sequences.iter().map(|s| s[0].len()).max().unwrap_or(0);
    let padded = sequences
        .iter()
        .map(|seq| {
            let mut out: [Vec<f32>; 4] = Default::default();
            for ch in 0..4 {
                let mut v = seq[ch].clone();
                v.resize(max_len, 0.0);
                out[ch] = v;
            }
            out
        })
        .collect();
    (padded, max_len)
}

/// Flattens a batch of channel-major padded sequences into a single Vec<f32>
/// in [batch, channels, seq_len] row-major order, ready for TensorData.
pub fn flatten_batch(padded: &[[Vec<f32>; 4]]) -> Vec<f32> {
    padded
        .iter()
        .flat_map(|seq| seq.iter().flat_map(|c| c.iter().copied()))
        .collect()
}

pub fn load_sequences(pathfile: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let file = File::open(pathfile)?;
    let reader = BufReader::new(file);
    let mut sequences = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if !line.is_empty() && !line.starts_with('>') {
            sequences.push(line.to_string());
        }
    }
    if sequences.is_empty() {
        return Err("no sequences found in input file".into());
    }
    Ok(sequences)
}

pub fn train_autoencoder(
    pathfile: &str,
    epochs: usize,
    lr: f64,
) -> Result<(DnaAutoEncoder<MyAutodiffBackend>, Vec<f32>), Box<dyn Error>> {
    let device = NdArrayDevice::default();

    let sequences = load_sequences(pathfile)?;
    let encoded = one_hot_encode(&sequences);
    let (padded, seq_len) = pad_sequences(&encoded);

    let batch_size = padded.len();
    let channels = 4;
    let flat = flatten_batch(&padded);

    let train_tensor: Tensor<MyAutodiffBackend, 3> = Tensor::from_data(
        TensorData::new(flat, [batch_size, channels, seq_len]),
        &device,
    );

    let mut model: DnaAutoEncoder<MyAutodiffBackend> = DnaAutoEncoder::new(&device, seq_len);
    let mut optimizer = SgdConfig::new().init();
    let loss_fn = MseLoss::new();

    for epoch in 1..=epochs {
        let output = model.forward(train_tensor.clone());
        let loss = loss_fn.forward(output, train_tensor.clone(), Reduction::Mean);

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(lr, model, grads_params);

        let loss_value: f32 = loss.into_scalar();
        println!("epoch {epoch}/{epochs} - reconstruction loss: {loss_value:.6}");
    }

    let final_output = model.forward(train_tensor.clone());
    let errors = per_sample_errors(final_output, train_tensor);

    Ok((model, errors))
}

/// Mean squared reconstruction error per sample in the batch, used as the
/// anomaly score: higher error = less like the training distribution.
fn per_sample_errors<B: AutodiffBackend>(output: Tensor<B, 3>, target: Tensor<B, 3>) -> Vec<f32> {
    let diff = output - target;
    let squared = diff.clone().mul(diff);
    // average over channels (dim 1) and sequence length (dim 2), keep batch dim
    let per_sample = squared.mean_dim(2).mean_dim(1); // shape [batch, 1, 1]
    let batch = per_sample.dims()[0];
    per_sample
        .reshape([batch])
        .into_data()
        .to_vec::<f32>()
        .expect("failed to read reconstruction errors")
}

/// Given training-set reconstruction errors, computes a mean + k*std
/// threshold and flags which of the provided `errors` exceed it.
pub fn flag_anomalies(train_errors: &[f32], errors: &[f32], k: f32) -> (f32, Vec<bool>) {
    let n = train_errors.len().max(1) as f32;
    let mean = train_errors.iter().sum::<f32>() / n;
    let var = train_errors.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let threshold = mean + k * std;
    let flags = errors.iter().map(|e| *e > threshold).collect();
    (threshold, flags)
}
