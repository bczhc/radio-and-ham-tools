use std::path::PathBuf;
use clap::Parser;
use hound::{SampleFormat, WavSpec, WavWriter};
use radio_and_ham_tools::iq_raw::{parse_sample_range, read_iq_raw};
use radio_and_ham_tools::IqWriter;

#[derive(Parser)]
struct Args {
    /// Raw IQ file input
    input: PathBuf,
    /// WAV Output
    output: PathBuf,
    /// Start time in format HH:mm:ss
    #[arg(long, default_value = "00:00:00")]
    start: String,
    /// End time in format HH:mm:ss
    #[arg(long)]
    end: Option<String>,
    /// Sample rate
    #[arg(short = 'r', long, default_value = "768000")]
    sample_rate: u64,
}

fn main() ->anyhow::Result<()> {
    let args = Args::parse();
    let sample_range = parse_sample_range(Some(&args.start), args.end.as_ref().map(String::as_str), args.sample_rate)?;
    let iq = read_iq_raw(args.input, false, sample_range.start, sample_range.length)?;
    
    let mut wav_writer = WavWriter::create(args.output, WavSpec {
        channels: 2,
        sample_rate: args.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    })?;

    for (_n, iq) in iq {
        wav_writer.write_iq_s16(iq)?
    }
    
    Ok(())
}