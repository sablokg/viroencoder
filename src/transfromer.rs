use burn::{
    backend::{Autodiff, NdArray},
    config::Config,
    data::{
        dataloader::{DataLoaderBuilder, batcher::Batcher},
        dataset::Dataset,
    },
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{Bool, Int, Tensor, backend::AutodiffBackend, backend::Backend},
    train::{
        ClassificationOutput, LearnerBuilder, TrainOutput, TrainStep, ValidStep,
        metric::{AccuracyMetric, LossMetric},
    },
};
use rand::seq::SliceRandom;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Number of one-hot channels per nucleotide position (A, T, G, C / N).
const N_BASES: usize = 4;

/// Converts each sequence into a flattened one-hot vector, padded/truncated
/// to `maxlen` positions (each position contributes 4 floats), plus the
/// true (unpadded, un-truncated-beyond-maxlen) length of each sequence.
///
/// Any IUPAC ambiguity code other than A/T/G/C (e.g. N, R, Y, W, S, K, M...)
/// is encoded as the all-zero "unknown base" vector rather than being
/// silently dropped, so every character consumes exactly one position and
/// downstream positional encodings stay aligned with the real sequence.
pub fn variantpad(
    input: &[String],
    maxlen: usize,
) -> Result<(Vec<Vec<f32>>, Vec<usize>), Box<dyn Error>> {
    let mut vecstring: Vec<Vec<f32>> = Vec::with_capacity(input.len());
    let mut lengths: Vec<usize> = Vec::with_capacity(input.len());
    for val in input.iter() {
        let mut sequencevec: Vec<Vec<f32>> = Vec::with_capacity(maxlen);
        for c in val.chars().take(maxlen) {
            match c.to_ascii_uppercase() {
                'A' => sequencevec.push(vec![1.0, 0.0, 0.0, 0.0]),
                'T' => sequencevec.push(vec![0.0, 1.0, 0.0, 0.0]),
                'G' => sequencevec.push(vec![0.0, 0.0, 1.0, 0.0]),
                'C' => sequencevec.push(vec![0.0, 0.0, 0.0, 1.0]),
                // Any other IUPAC / ambiguous / unexpected character:
                // encode as "unknown" rather than skipping the position.
                _ => sequencevec.push(vec![0.0, 0.0, 0.0, 0.0]),
            }
        }
        // true length = number of real (unpadded) positions actually written
        lengths.push(sequencevec.len());
        // pad shorter sequences with all-zero rows so every item has
        // exactly `maxlen` positions
        while sequencevec.len() < maxlen {
            sequencevec.push(vec![0.0, 0.0, 0.0, 0.0]);
        }
        vecstring.push(sequencevec.into_iter().flatten().collect());
    }
    Ok((vecstring, lengths))
}

/// Reads a FASTA file into a vec of (header, sequence) pairs, joining
/// multi-line sequences into a single string per record.
pub fn read_fasta(path: &str) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records: Vec<(String, String)> = Vec::new();
    let mut current_header: Option<String> = None;
    let mut current_seq = String::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix('>') {
            if let Some(h) = current_header.take() {
                records.push((h, current_seq.clone()));
                current_seq.clear();
            }
            current_header = Some(header.to_string());
        } else {
            current_seq.push_str(line);
        }
    }
    if let Some(h) = current_header.take() {
        records.push((h, current_seq));
    }

    if records.is_empty() {
        return Err("no records found in FASTA file".into());
    }
    Ok(records)
}

#[derive(Clone, Debug)]
pub struct DnaItem {
    onehot: Vec<f32>,
    length: usize,
    label: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Datasetsiter {
    items: Vec<DnaItem>,
}

impl Datasetsiter {
    /// Loads a comma-separated `sequence,label` file into a dataset.
    ///
    /// Malformed lines (wrong column count, unparseable label, or a label
    /// outside `[0, n_classes)`) are reported and skipped instead of
    /// panicking, so one bad row in a large file doesn't kill the whole run.
    pub fn load(pathfile: &str, maxlen: usize, n_classes: usize) -> Self {
        let pathfileopen = File::open(pathfile).expect("file not present");
        let pathfileread = BufReader::new(pathfileopen);
        let mut vecstring: Vec<String> = Vec::new();
        let mut veclabels: Vec<usize> = Vec::new();

        for (line_no, i) in pathfileread.lines().enumerate() {
            let line = match i {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("warning: skipping unreadable line {}: {e}", line_no + 1);
                    continue;
                }
            };
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let linevec: Vec<&str> = line.split(',').map(str::trim).collect();
            if linevec.len() < 2 {
                eprintln!(
                    "warning: skipping malformed line {} (expected 'sequence,label'): {line}",
                    line_no + 1
                );
                continue;
            }

            let seq = linevec[0];
            if seq.is_empty() {
                eprintln!("warning: skipping line {} with empty sequence", line_no + 1);
                continue;
            }

            let label = match linevec[1].parse::<usize>() {
                Ok(l) => l,
                Err(_) => {
                    eprintln!(
                        "warning: skipping line {} with unparseable label {:?}",
                        line_no + 1,
                        linevec[1]
                    );
                    continue;
                }
            };
            if label >= n_classes {
                eprintln!(
                    "warning: skipping line {} with label {label} out of range [0, {n_classes})",
                    line_no + 1
                );
                continue;
            }

            vecstring.push(seq.to_string());
            veclabels.push(label);
        }

        let mut returnvec: Vec<DnaItem> = Vec::new();
        let (convertvec, lengths) = variantpad(&vecstring, maxlen).unwrap();
        for i in 0..convertvec.len() {
            returnvec.push(DnaItem {
                onehot: convertvec[i].clone(),
                length: lengths[i],
                label: veclabels[i],
            })
        }
        Self { items: returnvec }
    }

    pub fn split(self, train_frac: f32) -> (Self, Self) {
        let mut items = self.items;
        items.shuffle(&mut rand::thread_rng());
        let n_train = (items.len() as f32 * train_frac) as usize;
        let train = items[..n_train].to_vec();
        let valid = items[n_train..].to_vec();
        (Self { items: train }, Self { items: valid })
    }
}

impl Dataset<DnaItem> for Datasetsiter {
    fn get(&self, index: usize) -> Option<DnaItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone)]
pub struct DnaBatcher<B: Backend> {
    device: B::Device,
    max_len: usize,
}

impl<B: Backend> DnaBatcher<B> {
    pub fn new(device: B::Device, max_len: usize) -> Self {
        Self { device, max_len }
    }
}

#[derive(Clone, Debug)]
pub struct DnaBatch<B: Backend> {
    onehot: Tensor<B, 3>,
    /// True at padded positions, false at real positions. Shape [batch, seq_len].
    pad_mask: Tensor<B, 2, Bool>,
    labels: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<DnaItem, DnaBatch<B>> for DnaBatcher<B> {
    fn batch(&self, items: Vec<DnaItem>) -> DnaBatch<B> {
        let batch_size = items.len();
        let flat: Vec<f32> = items.iter().flat_map(|i| i.onehot.clone()).collect();
        let onehot = Tensor::<B, 1>::from_floats(flat.as_slice(), &self.device).reshape([
            batch_size,
            self.max_len,
            N_BASES,
        ]);

        // Build the padding mask: position j is padding iff j >= item.length.
        let mut mask_data: Vec<bool> = Vec::with_capacity(batch_size * self.max_len);
        for item in items.iter() {
            for j in 0..self.max_len {
                mask_data.push(j >= item.length);
            }
        }
        let pad_mask = Tensor::<B, 1, Bool>::from_data(mask_data.as_slice(), &self.device)
            .reshape([batch_size, self.max_len]);

        let label_data: Vec<i32> = items.iter().map(|i| i.label as i32).collect();
        let labels = Tensor::<B, 1, Int>::from_ints(label_data.as_slice(), &self.device);
        DnaBatch {
            onehot,
            pad_mask,
            labels,
        }
    }
}

#[derive(Config)]
pub struct DnaTransformerConfig {
    max_len: usize,
    #[config(default = 64)]
    d_model: usize,
    #[config(default = 4)]
    n_heads: usize,
    #[config(default = 2)]
    n_layers: usize,
    #[config(default = 128)]
    d_ff: usize,
    #[config(default = 2)]
    n_classes: usize,
    #[config(default = 0.1)]
    dropout: f64,
}

#[derive(Module, Debug)]
pub struct DnaTransformer<B: Backend> {
    input_proj: Linear<B>,
    pos_emb: Embedding<B>,
    transformer: TransformerEncoder<B>,
    classifier: Linear<B>,
    dropout: Dropout,
}

impl DnaTransformerConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> DnaTransformer<B> {
        let input_proj = LinearConfig::new(N_BASES, self.d_model).init(device);
        let pos_emb = EmbeddingConfig::new(self.max_len, self.d_model).init(device);

        let transformer =
            TransformerEncoderConfig::new(self.d_model, self.d_ff, self.n_heads, self.n_layers)
                .with_dropout(self.dropout)
                .init(device);

        let classifier = LinearConfig::new(self.d_model, self.n_classes).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();

        DnaTransformer {
            input_proj,
            pos_emb,
            transformer,
            classifier,
            dropout,
        }
    }
}

impl<B: Backend> DnaTransformer<B> {
    fn forward(&self, onehot: Tensor<B, 3>, pad_mask: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
        let [batch_size, seq_len, _] = onehot.dims();
        let device = onehot.device();

        // Linear projection of one-hot vectors into model dimension.
        let projected = self.input_proj.forward(onehot); // [batch, seq_len, d_model]

        let positions = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .reshape([1, seq_len])
            .repeat_dim(0, batch_size);
        let pos_emb = self.pos_emb.forward(positions);

        let x = self.dropout.forward(projected + pos_emb);

        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(x).mask_pad(pad_mask.clone()));

        // Masked mean-pool over the sequence dimension: zero out padded
        // positions before summing, then divide by each sequence's real
        // (unpadded) length instead of the fixed `seq_len`.
        let keep = pad_mask.bool_not().float().unsqueeze_dim::<3>(2); // [batch, seq_len, 1]
        let masked = encoded * keep.clone();
        let summed = masked.sum_dim(1).squeeze::<2>(1); // [batch, d_model]
        let counts = keep.sum_dim(1).squeeze::<2>(1).clamp_min(1.0); // [batch, 1] avoid div-by-zero
        let pooled = summed / counts;

        self.classifier.forward(pooled)
    }

    fn forward_classification(&self, batch: DnaBatch<B>) -> ClassificationOutput<B> {
        let logits = self.forward(batch.onehot, batch.pad_mask);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), batch.labels.clone());
        ClassificationOutput::new(loss, logits, batch.labels)
    }
}

impl<B: AutodiffBackend> TrainStep<DnaBatch<B>, ClassificationOutput<B>> for DnaTransformer<B> {
    fn step(&self, batch: DnaBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward_classification(batch);
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<DnaBatch<B>, ClassificationOutput<B>> for DnaTransformer<B> {
    fn step(&self, batch: DnaBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch)
    }
}

// ---------------------------------------------------------------------------
// Training / inference entry points
// ---------------------------------------------------------------------------

type MyBackend = NdArray<f32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

const MODEL_PATH: &str = "./artifacts/dna_transformer_onehot";
const N_CLASSES: usize = 2; // must match DnaTransformerConfig::n_classes default

pub fn train(pathfile: &str, maxlen: usize) -> Result<String, Box<dyn Error>> {
    let device = <MyBackend as Backend>::Device::default();

    let dataset = Datasetsiter::load(pathfile, maxlen, N_CLASSES);
    println!("Loaded {} sequences", dataset.len());
    let (train_ds, valid_ds) = dataset.split(0.8);

    let batcher_train = DnaBatcher::<MyAutodiffBackend>::new(device.clone(), maxlen);
    let batcher_valid = DnaBatcher::<MyBackend>::new(device.clone(), maxlen);

    let train_loader = DataLoaderBuilder::new(batcher_train)
        .batch_size(8)
        .shuffle(42)
        .num_workers(1)
        .build(train_ds);

    let valid_loader = DataLoaderBuilder::new(batcher_valid)
        .batch_size(8)
        .num_workers(1)
        .build(valid_ds);

    let model_config = DnaTransformerConfig::new(maxlen);
    let model = model_config.init::<MyAutodiffBackend>(&device);

    let optimizer = AdamConfig::new().init();

    let learner = LearnerBuilder::new("./artifacts")
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(30)
        .build(model, optimizer, 1e-3);

    let trained_model = learner.fit(train_loader, valid_loader);

    trained_model
        .clone()
        .save_file(MODEL_PATH, &CompactRecorder::new())
        .expect("Failed to save trained model");

    Ok(format!("Training complete. Model saved to {MODEL_PATH}"))
}

/// Loads a trained model from disk and runs predictions over every
/// sequence found in a FASTA file, printing `header <TAB> predicted_class`.
pub fn predict(fasta_path: &str, maxlen: usize) -> Result<String, Box<dyn Error>> {
    let device = <MyBackend as Backend>::Device::default();

    let records = read_fasta(fasta_path).expect("Failed to read FASTA file");
    let sequences: Vec<String> = records.iter().map(|(_, seq)| seq.clone()).collect();
    let (onehot_vecs, lengths) =
        variantpad(&sequences, maxlen).expect("Failed to encode sequences");

    let model_config = DnaTransformerConfig::new(maxlen);
    let model: DnaTransformer<MyBackend> = model_config
        .init::<MyBackend>(&device)
        .load_file(MODEL_PATH, &CompactRecorder::new(), &device)
        .expect("Failed to load trained model (train it first with no args)");

    let batch_size = onehot_vecs.len();
    let flat: Vec<f32> = onehot_vecs.into_iter().flatten().collect();
    let onehot = Tensor::<MyBackend, 1>::from_floats(flat.as_slice(), &device)
        .reshape([batch_size, maxlen, N_BASES]);

    let mut mask_data: Vec<bool> = Vec::with_capacity(batch_size * maxlen);
    for len in lengths.iter() {
        for j in 0..maxlen {
            mask_data.push(j >= *len);
        }
    }
    let pad_mask = Tensor::<MyBackend, 1, Bool>::from_data(mask_data.as_slice(), &device)
        .reshape([batch_size, maxlen]);

    let logits = model.forward(onehot, pad_mask);
    let predicted: Vec<i32> = logits
        .argmax(1)
        .into_data()
        .to_vec::<i32>()
        .expect("Failed to read predictions");

    println!("header\tpredicted_class");
    for ((header, _), class) in records.iter().zip(predicted.iter()) {
        println!("{header}\t{class}");
    }

    Ok("Prediction has been done".to_string())
}
