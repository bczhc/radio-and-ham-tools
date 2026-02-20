#![feature(let_chains)]

use hilbert_transform::hilbert;
use num_complex::Complex64;
use radio_and_ham_tools::{create_sdrpp_wav_iq, ffmpeg_read_audio_pcm, sample_formats, IqWriter};

fn main() -> anyhow::Result<()> {
    let input1 = "/home/bczhc/Music/奢香夫人 [rDS1xolXsIQ].webm";
    let input2 = "/home/bczhc/Music/A_Little_Love-冯曦妤-48000.flac";
    let out = "/home/bczhc/out.wav";
    let mut writer = create_sdrpp_wav_iq(out, 48000)?;

    let mut samples1 = ffmpeg_read_audio_pcm::<sample_formats::F64LE>(input1, 48000, 1)?
        .iter()
        .collect::<Vec<_>>();
    let mut samples2 = ffmpeg_read_audio_pcm::<sample_formats::F64LE>(input2, 48000, 1)?
        .iter()
        .collect::<Vec<_>>();

    let regulate_volume = |samples: &mut [f64]| {
        let &max = samples
            .iter()
            .max_by(|&&a, &b| a.partial_cmp(b).unwrap())
            .unwrap();
        let factor = 1.0 / max;
        samples.iter_mut().for_each(|x| *x *= factor * 0.5);
    };
    regulate_volume(&mut samples1);
    regulate_volume(&mut samples2);

    // align to the same length
    let (samples1, samples2, count) = if samples1.len() < samples2.len() {
        (&samples1, &samples2[..samples1.len()], samples1.len())
    } else {
        (&samples2, &samples1[..samples2.len()], samples2.len())
    };
    let samples1_hilbert = hilbert(samples1);
    let samples2_hilbert = hilbert(samples2);
    for idx in 0..count {
        let i = samples1[idx] + samples2_hilbert[idx].im;
        let q = samples1_hilbert[idx].im + samples2[idx];
        writer.write_iq_s16(Complex64::new(i, q))?
    }

    Ok(())
}
