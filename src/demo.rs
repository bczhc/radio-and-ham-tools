use std::f64::consts::PI;
use num_complex::Complex64;
use radio_and_ham_tools::iq_raw::read_iq_raw;
use radio_and_ham_tools::{create_sdrpp_wav_iq, Hertz, IqWriter};

fn main() -> anyhow::Result<()> {
    let mut wav_writer = create_sdrpp_wav_iq("/home/bczhc/smb/smb-tmp/sdr/out.wav", 768000)?;
    
    let file = "/home/bczhc/smb/smb-tmp/sdr/gqrx_20250323_073954_14958950_768000_fc.raw";
    let f = Hertz::from_khz_str("173.7")?;
    let iq = read_iq_raw(file, false, 0, None)?;
    for (n, iq) in iq {
        let t = n as f64 / 768000_f64;
        let shift = Complex64::from_polar(1.0, -2.0 * PI * f.0 as f64 * t);
        wav_writer.write_iq_s16(iq * shift)?;
    }
    Ok(())
}
