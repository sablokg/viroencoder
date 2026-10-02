mod args;
use crate::args::CommandParse;
use crate::args::Commands;
use clap::Parser;
use vornix_banner::{Banner, BuiltinFont, Style, rgb};
mod autoencoder;
use crate::autoencoder::flag_anomalies;
use crate::autoencoder::train_autoencoder;
mod transfromer;
use crate::transfromer::predict;
use crate::transfromer::train;
mod lstmgru;
use crate::lstmgru::trainlstm;
mod rnngru;
use self::rnngru::trainlstmgru;

/*
Gaurav Sablok
gsablok@proton.me
*/

fn main() {
    let style = Style::new().fg(rgb(255, 100, 20)).bold();

    let mut banner = Banner::new("viroEncoder")
        .with_builtin_font(BuiltinFont::Block)
        .with_style(style)
        .centered(true);

    banner.display().unwrap();

    let args = CommandParse::parse();
    match &args.command {
        Commands::Autoencoder {
            pathname,
            epochs,
            lr,
        } => {
            let (model, train_errors) = train_autoencoder(
                pathname,
                epochs.parse::<usize>().unwrap(),
                lr.parse::<f64>().unwrap(),
            )
            .unwrap();

            // Re-run the trained model on the training data to get per-sample scores
            // and flag anomalies relative to the training error distribution.
            let (threshold, flags) = flag_anomalies(&train_errors, &train_errors, 2.0);
            println!("anomaly threshold (mean + 2*std): {threshold:.6}");
            for (i, (err, is_anomaly)) in train_errors.iter().zip(flags.iter()).enumerate() {
                println!(
                    "sequence {i}: reconstruction error = {err:.6}{}",
                    if *is_anomaly { "  <-- ANOMALY" } else { "" }
                );
            }

            let _ = model;
        }
        Commands::TransformerPredict {
            filename,
            maxlength,
        } => {
            let command = predict(filename, *maxlength).unwrap();
            println!("command has finished:{}", command);
        }
        Commands::TransformerTrain {
            filename,
            maxlength,
        } => {
            let command = train(filename, *maxlength).unwrap();
            println!("The command has finished:{}", command);
        }
        Commands::LSTM { pathfile } => {
            let _ = trainlstm(pathfile).unwrap();
            println!("The LSTM model has been trained");
        }
        Commands::LSTMGRU { pathfile } => {
            let _ = trainlstmgru(pathfile).unwrap();
            println!("The lstm gru has finished");
        }
    }
}
