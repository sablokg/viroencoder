use burn::module::AutodiffModule;
use burn::module::Module;
use burn::nn::gru::Gru;
use burn::nn::gru::GruConfig;
use burn::nn::{Dropout, DropoutConfig};
use burn::nn::{Linear, LinearConfig};
use burn::nn::{Lstm, LstmConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Shape, Tensor, TensorData};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

type MyBackend = burn::backend::Autodiff<burn::backend::NdArray>;

// Number of one-hot channels: A, T, G, C, N.
// N gets its own channel so it is distinguishable from padding, which is
// now encoded as the all-zero vector reserved exclusively for "no base here".
const CHANNELS: usize = 5;

#[derive(Module, Debug)]
pub struct SeqClassifier<B: Backend> {
    lstm1: Lstm<B>,
    lstm2: Lstm<B>,
    dropout: Dropout,
    gru1: Gru<B>,
    dropout2: Dropout,
    linear: Linear<B>,
}

impl<B: Backend> SeqClassifier<B> {
    pub fn new(device: &B::Device, inputdim: usize, hidden: usize, num_classes: usize) -> Self {
        Self {
            lstm1: LstmConfig::new(inputdim, hidden, true).init(device),
            lstm2: LstmConfig::new(hidden, hidden, true).init(device),
            dropout: DropoutConfig::new(0.3).init(),
            gru1: GruConfig::new(hidden, hidden, true).init(device),
            dropout2: DropoutConfig::new(0.3).init(),
            linear: LinearConfig::new(hidden, num_classes).init(device),
        }
    }

    /// Forward pass. `lengths` gives the true (unpadded) length of each
    /// sequence in the batch, in the same order as the batch dimension of
    /// `input`. This is required so we can pull out each sequence's real
    /// last timestep instead of the padded tensor's final position, which
    /// for shorter sequences would just be trailing zero-padding.
    pub fn forward(&self, input: Tensor<B, 3>, lengths: &[usize]) -> Tensor<B, 2> {
        let (out1, _) = self.lstm1.forward(input, None);
        let (out2, _) = self.lstm2.forward(out1, None);
        let out2 = self.dropout.forward(out2);

        // Gru::forward returns a single Tensor<B, 3>, not a tuple.
        let out3 = self.gru1.forward(out2, None);
        let out3 = self.dropout2.forward(out3);

        let [batch, _seq_len, hidden] = out3.dims();
        debug_assert_eq!(batch, lengths.len(), "lengths must match batch size");

        // Gather each example's real last timestep (lengths[i] - 1) instead
        // of always using the batch's global max_len - 1.
        let rows: Vec<Tensor<B, 2>> = (0..batch)
            .map(|i| {
                let len = lengths[i].max(1);
                out3.clone()
                    .slice([i..i + 1, len - 1..len, 0..hidden])
                    .reshape([1, hidden])
            })
            .collect();
        let last = Tensor::cat(rows, 0);

        self.linear.forward(last)
    }
}

/// One-hot encode a batch of DNA sequences (A/T/G/C/N) and right-pad them
/// to the length of the longest sequence in the batch. N gets its own
/// channel so it's distinguishable from padding (which is now the only
/// thing encoded as an all-zero vector). Returns the padded tensor of
/// shape [batch, max_len, CHANNELS] along with each sequence's true length.
pub fn seqpad<B: Backend>(input: &[String], device: &B::Device) -> (Tensor<B, 3>, Vec<usize>) {
    let encoded: Vec<Vec<f32>> = input
        .iter()
        .map(|val| {
            val.chars()
                .filter_map(|c| match c {
                    'A' => Some([1.0, 0.0, 0.0, 0.0, 0.0]),
                    'T' => Some([0.0, 1.0, 0.0, 0.0, 0.0]),
                    'G' => Some([0.0, 0.0, 1.0, 0.0, 0.0]),
                    'C' => Some([0.0, 0.0, 0.0, 1.0, 0.0]),
                    'N' => Some([0.0, 0.0, 0.0, 0.0, 1.0]),
                    _ => None,
                })
                .flatten()
                .collect()
        })
        .collect();

    let batch = encoded.len();
    let lengths: Vec<usize> = encoded.iter().map(|v| v.len() / CHANNELS).collect();
    let max_len = lengths.iter().copied().max().unwrap_or(0);

    let mut flat: Vec<f32> = Vec::with_capacity(batch * max_len * CHANNELS);
    for (seq, &seq_len) in encoded.iter().zip(lengths.iter()) {
        flat.extend_from_slice(seq);
        flat.extend(std::iter::repeat(0.0).take((max_len - seq_len) * CHANNELS));
    }

    let data = TensorData::new(flat, Shape::new([batch, max_len, CHANNELS]));
    let tensor = Tensor::<B, 3>::from_data(data, device);

    (tensor, lengths)
}

/// Read a two-column CSV (sequence,label) file.
fn read_training_csv(pathfile: &str) -> Result<(Vec<String>, Vec<i64>), Box<dyn Error>> {
    let mut sequences: Vec<String> = Vec::new();
    let mut labels: Vec<i64> = Vec::new();

    let fileopen = File::open(pathfile)?;
    let fileread = BufReader::new(fileopen);
    for line in fileread.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let linevec = line.split(',').collect::<Vec<_>>();
        if linevec.len() < 2 {
            continue;
        }
        sequences.push(linevec[0].trim().to_string());
        labels.push(linevec[1].trim().parse::<i64>()?);
    }

    Ok((sequences, labels))
}

/// Deterministic, dependency-free train/validation split. Takes every
/// `val_every_n`-th record (by original order) as validation so results are
/// reproducible without pulling in a `rand` dependency. Set `val_every_n`
/// to e.g. 5 for an ~80/20 split.
fn train_val_split(
    sequences: Vec<String>,
    labels: Vec<i64>,
    val_every_n: usize,
) -> (Vec<String>, Vec<i64>, Vec<String>, Vec<i64>) {
    let mut train_seq = Vec::new();
    let mut train_lab = Vec::new();
    let mut val_seq = Vec::new();
    let mut val_lab = Vec::new();

    for (i, (seq, lab)) in sequences.into_iter().zip(labels.into_iter()).enumerate() {
        if val_every_n > 0 && i % val_every_n == 0 {
            val_seq.push(seq);
            val_lab.push(lab);
        } else {
            train_seq.push(seq);
            train_lab.push(lab);
        }
    }

    (train_seq, train_lab, val_seq, val_lab)
}

fn to_label_tensor<B: Backend>(labels: &[i64], device: &B::Device) -> Tensor<B, 1, Int> {
    let data = TensorData::new(labels.to_vec(), Shape::new([labels.len()]));
    Tensor::<B, 1, Int>::from_data(data, device)
}

/// Train the model on a (sequence,label) CSV file and return the trained
/// model. Holds out roughly 1/5 of the data (by fixed stride) as a
/// validation set so the reported metric reflects generalization rather
/// than pure training loss.
pub fn trainlstm(pathfile: &str) -> Result<SeqClassifier<MyBackend>, Box<dyn Error>> {
    let device: <MyBackend as Backend>::Device = Default::default();

    let (sequences, labels) = read_training_csv(pathfile)?;
    let (train_seq, train_lab, val_seq, val_lab) = train_val_split(sequences, labels, 5);

    let (train_tensor, train_lengths) = seqpad::<MyBackend>(&train_seq, &device);
    let train_labels = to_label_tensor::<MyBackend>(&train_lab, &device);

    let has_val = !val_seq.is_empty();
    let (val_tensor, val_lengths) = seqpad::<MyBackend>(&val_seq, &device);
    let val_labels = to_label_tensor::<MyBackend>(&val_lab, &device);

    // Precompute eval-mode (inner, non-autodiff backend) copies of the
    // train/val tensors once, up front, so we don't re-encode on every
    // logging interval. These live on `MyBackend::InnerBackend`, which is
    // what `model.valid()` produces below — dropout becomes a no-op on
    // this backend since `ad_enabled()` is false for it.
    type InnerBackend = <MyBackend as AutodiffBackend>::InnerBackend;
    let inner_device: <InnerBackend as Backend>::Device = Default::default();

    let (eval_train_tensor, _) = seqpad::<InnerBackend>(&train_seq, &inner_device);
    let eval_train_labels = to_label_tensor::<InnerBackend>(&train_lab, &inner_device);

    let (eval_val_tensor, _) = seqpad::<InnerBackend>(&val_seq, &inner_device);
    let eval_val_labels = to_label_tensor::<InnerBackend>(&val_lab, &inner_device);

    let mut model: SeqClassifier<MyBackend> = SeqClassifier::new(&device, CHANNELS, 16, 2);
    let mut optim = AdamConfig::new().init();
    let lr = 1e-2;

    for epoch in 0..200 {
        let logits = model.forward(train_tensor.clone(), &train_lengths);
        let loss = burn::nn::loss::CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, train_labels.clone());

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optim.step(lr, model, grads_params);

        if epoch % 20 == 0 {
            // Switch to the inner (non-autodiff) backend for eval so
            // Dropout::forward sees ad_enabled() == false and becomes a
            // no-op, instead of still randomly zeroing activations the
            // way it would if we reused `model` (on the autodiff backend)
            // directly.
            let eval_model = model.valid();

            let train_loss_val = burn::nn::loss::CrossEntropyLossConfig::new()
                .init(&inner_device)
                .forward(
                    eval_model.forward(eval_train_tensor.clone(), &train_lengths),
                    eval_train_labels.clone(),
                )
                .into_scalar();

            if has_val {
                let val_loss_val = burn::nn::loss::CrossEntropyLossConfig::new()
                    .init(&inner_device)
                    .forward(
                        eval_model.forward(eval_val_tensor.clone(), &val_lengths),
                        eval_val_labels.clone(),
                    )
                    .into_scalar();
                println!(
                    "epoch {epoch}: train_loss = {:?}, val_loss = {:?}",
                    train_loss_val, val_loss_val
                );
            } else {
                println!("epoch {epoch}: train_loss = {:?}", train_loss_val);
            }
        }
    }

    Ok(model)
}

/*
/// Read a FASTA file into (header, sequence) pairs. Handles multi-line
/// sequences and strips the leading '>' from headers.
pub fn read_fasta(pathfile: &str) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let fileopen = File::open(pathfile)?;
    let fileread = BufReader::new(fileopen);

    let mut records: Vec<(String, String)> = Vec::new();
    let mut current_header: Option<String> = None;
    let mut current_seq = String::new();

    for line in fileread.lines() {
        let line = line?;
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            if let Some(h) = current_header.take() {
                records.push((h, std::mem::take(&mut current_seq)));
            }
            current_header = Some(header.trim().to_string());
        } else {
            current_seq.push_str(line.trim());
        }
    }
    if let Some(h) = current_header.take() {
        records.push((h, current_seq));
    }

    Ok(records)
}

/// Run inference on every record in a FASTA file using a trained model.
/// Returns (header, class_probabilities) pairs in file order.
pub fn predict_fasta<B: Backend>(
    model: &SeqClassifier<B>,
    device: &B::Device,
    fasta_path: &str,
) -> Result<Vec<(String, Vec<f32>)>, Box<dyn Error>> {
    let records = read_fasta(fasta_path)?;
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let headers: Vec<String> = records.iter().map(|(h, _)| h.clone()).collect();
    let sequences: Vec<String> = records.into_iter().map(|(_, s)| s).collect();

    let (input_tensor, lengths) = seqpad::<B>(&sequences, device);
    let logits = model.forward(input_tensor, &lengths);
    let probs = softmax(logits, 1);

    let [batch, num_classes] = probs.dims();
    let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();

    let mut results = Vec::with_capacity(batch);
    for (i, header) in headers.into_iter().enumerate() {
        let start = i * num_classes;
        let end = start + num_classes;
        results.push((header, probs_data[start..end].to_vec()));
    }

    Ok(results)
}

/// Convenience wrapper: train on a labeled CSV, then predict on a FASTA file,
/// printing the predicted class probabilities for each record.
pub fn train_and_predict(train_csv: &str, fasta_path: &str) -> Result<(), Box<dyn Error>> {
    let model = trainlstm(train_csv)?;
    let device: <MyBackend as Backend>::Device = Default::default();

    let predictions = predict_fasta(&model, &device, fasta_path)?;
    for (header, probs) in predictions {
        println!("{header}: {probs:?}");
    }

    Ok(())
}
*/
