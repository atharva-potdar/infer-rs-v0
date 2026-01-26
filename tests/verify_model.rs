use ort::{session::builder::GraphOptimizationLevel, session::Session, value::Tensor};
use std::fs::File;
use std::io::Read;
fn load_npy_data(path: &str) -> Vec<f32> {
    let mut file = File::open(path).expect(&format!("Failed to open {}", path));
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes).unwrap();
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let data_offset = 10 + header_len;
    let mut floats = Vec::new();
    for chunk in bytes[data_offset..].chunks_exact(4) {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(chunk);
        floats.push(f32::from_le_bytes(buf));
    }
    floats
}
#[test]
fn test_verify_model_ort() -> anyhow::Result<()> {
    let model_path = "model_artifacts/mnist_cnn.onnx";
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;
    let input_vec = load_npy_data("model_artifacts/golden_input.npy");
    let golden_output = load_npy_data("model_artifacts/golden_logits.npy");
    println!("DEBUG: Input First 5: {:?}", &input_vec[0..5]);
    println!("DEBUG: Golden First 5: {:?}", &golden_output[0..5]);
    let input_tensor = Tensor::from_array(([1usize, 1, 28, 28], input_vec.into_boxed_slice()))?;
    let inputs = ort::inputs!["input" => input_tensor];
    let outputs = session.run(inputs)?;
    let output_value = &outputs[0];
    let (_shape, data) = output_value.try_extract_tensor::<f32>()?;
    let actual_output: Vec<f32> = data.to_vec();
    println!("DEBUG: Actual First 5: {:?}", &actual_output[0..5]);
    let tolerance = 1e-4f32;
    for (i, (actual, expected)) in actual_output.iter().zip(golden_output.iter()).enumerate() {
        let diff = (actual - expected).abs();
        if diff >= tolerance {
            panic!("Mismatch at {}: Actual {}, Expected {}, Diff {}", i, actual, expected, diff);
        }
    }
    println!("SUCCESS: ORT matches golden output within tolerance.");
    Ok(())
}
