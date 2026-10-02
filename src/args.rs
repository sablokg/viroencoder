use clap::{Parser, Subcommand};
#[derive(Debug, Parser)]
#[command(
    name = "viroencoder",
    version = "1.0",
    about = "Viral Anamoly detection using Autoencoder
       ************************************************
       Gaurav Sablok,
       Email: gsablok@proton.me
      ************************************************"
)]
pub struct CommandParse {
    /// subcommands for the specific actions
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Viral Anamoly detection
    Autoencoder {
        /// path to the filename
        pathname: String,
        /// epochs
        epochs: String,
        /// learning rate
        lr: String,
    },
    /// Transformer Predict
    TransformerPredict {
        /// path to the filename
        filename: String,
        /// maxlength of the sequences
        maxlength: usize,
    },
    /// Transformer Train
    TransformerTrain {
        /// path to the filename
        filename: String,
        /// max sequence length
        maxlength: usize,
    },
    /// Train LSTM
    LSTM {
        /// path to the file
        pathfile: String,
    },
    /// Train the LSTMandGRU with MaxPool and Dropout.
    LSTMGRU {
        /// path to the file
        pathfile: String,
    },
}
