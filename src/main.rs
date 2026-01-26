use anyhow::Result;
use ort::{session::Session, session::builder::GraphOptimizationLevel, value::TensorRef};
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};
tonic::include_proto!("inference.v1");

const MODEL_PATH: &str = "model_artifacts/mnist_cnn.onnx";
async fn load_session() -> Result<Arc<Mutex<Session>>> {
    // Session::builder() returns a builder per docs
    let session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)? // optional tuning
        .with_intra_threads(1)? // single-threaded; increase if you plan to use run_async
        .commit_from_file(MODEL_PATH)?;
    Ok(Arc::new(Mutex::new(session)))
}

#[derive(Clone)]
struct InferenceService {
    session: Arc<Mutex<Session>>,
}
#[tonic::async_trait]
impl rpc_inference_service_server::RpcInferenceService for InferenceService {
    async fn rpc_inference(
        &self,
        req: Request<RpcInferenceRequest>,
    ) -> Result<Response<RpcInferenceResponse>, Status> {
        let req = req.into_inner();
        // Validate presence
        if req.tensor.is_empty() || req.shape.is_empty() {
            return Err(Status::invalid_argument("tensor and shape required"));
        }

        let shape_usize: Vec<usize> = req.shape.iter().map(|&d| d as usize).collect();
        let expected = shape_usize.iter().product::<usize>();
        if expected != req.tensor.len() {
            return Err(Status::invalid_argument(
                "shape does not match tensor length",
            ));
        }
        // For a 4D MNIST input (1,1,28,28) example:
        let arr = match ndarray::Array::from_shape_vec(shape_usize.clone(), req.tensor) {
            Ok(a) => a,
            Err(e) => return Err(Status::internal(format!("ndarray error: {}", e))),
        };
        // Convert to ort input via TensorRef (examples in ort docs use TensorRef::from_array_view)
        let input_value = TensorRef::from_array_view(&arr)
            .map_err(|e| Status::internal(format!("ort tensor error: {}", e)))?;
        // Lock the session and run
        let mut session_guard = self.session.lock().await;
        let outputs = session_guard
            .run(ort::inputs![input_value])
            .map_err(|e| Status::internal(format!("session run error: {}", e)))?;
        // Extract first output tensor as f32 slice and return
        let (_shape, out_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| Status::internal(format!("extract error: {}", e)))?;
        let response = RpcInferenceResponse {
            output: out_slice.to_vec(),
        };
        Ok(Response::new(response))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let session = load_session().await?;
    println!("Loaded session, ready to start server");
    let svc = InferenceService { session };
    let addr = "127.0.0.1:50051".parse().expect("invalid addr");
    println!("Serving gRPC on {}", addr);
    tonic::transport::Server::builder()
        .add_service(rpc_inference_service_server::RpcInferenceServiceServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}
