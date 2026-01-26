use burn_import::onnx::ModelGen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::compile_protos("proto/inference/v1/service.proto")?;
    ModelGen::new()
        .input("model_artifacts/mnist_cnn.onnx")
        .out_dir("generated/")
        .development(true)
        .run_from_script();
    Ok(())
}
