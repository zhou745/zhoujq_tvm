#define CHECK(status)                                                 \
  do {                                                                \
    auto ret = (status);                                              \
    if (ret != 0) {                                                   \
      std::cout << __LINE__ << " Cuda failure: " << ret << std::endl; \
      abort();                                                        \
    }                                                                 \
  } while (0)

#include <assert.h>
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <sys/stat.h>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;

static int B = 32;
static int INPUT_C;
static int INPUT_H;
static int INPUT_W;
static int OUTPUT_SIZE;
static int REPEAT = 10;
static bool TEST_INT8 = true;
static std::string MODEL_NAME;

auto randF() { return rand() / 65536.0f; }

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
 public:
  Logger() : Logger(Severity::kWARNING) {}

  Logger(Severity severity) : reportableSeverity(severity) {}

  void log(Severity severity, const char* msg) override {
    // suppress messages with severity enum value greater than the reportable
    if (severity > reportableSeverity) return;

    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }

  Severity reportableSeverity{Severity::kWARNING};
};

static Logger gLogger;

class Calibrator : public nvinfer1::IInt8LegacyCalibrator {
 public:
  Calibrator(int firstBatch = 0, double cutoff = 0.5, double quantile = 0.5,
             bool readCache = true)
      : mFirstBatch(firstBatch), mReadCache(readCache), index(0) {
    using namespace nvinfer1;
    mInputCount = B * INPUT_C * INPUT_H * INPUT_W;
    CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    reset(cutoff, quantile);
    mData = new float[mInputCount];
  }

  virtual ~Calibrator() { CHECK(cudaFree(mDeviceInput)); }

  int getBatchSize() const override { return 1; }
  double getQuantile() const override { return mQuantile; }
  double getRegressionCutoff() const override { return mCutoff; }

  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) override {
    for (int i = 0; i < mInputCount; i++) mData[i] = randF();
    CHECK(cudaMemcpy(mDeviceInput, mData, mInputCount, cudaMemcpyHostToDevice));

    if (index++ > 0) {
      delete[] mData;
      mData = nullptr;
      return false;
    }

    bindings[0] = mDeviceInput;
    return true;
  }

  const void* readCalibrationCache(size_t& length) override { return nullptr; }

  void writeCalibrationCache(const void* cache, size_t length) override {}

  const void* readHistogramCache(size_t& length) override {
    length = mHistogramCache.size();
    return length ? &mHistogramCache[0] : nullptr;
  }

  void writeHistogramCache(const void* cache, size_t length) override {
    mHistogramCache.clear();
    std::copy_n(reinterpret_cast<const char*>(cache), length,
                std::back_inserter(mHistogramCache));
  }

  void reset(double cutoff, double quantile) {
    mCutoff = cutoff;
    mQuantile = quantile;
  }

 private:
  int index;
  int mFirstBatch;
  float* mData;
  double mCutoff, mQuantile;
  bool mReadCache{true};

  size_t mInputCount;
  void* mDeviceInput{nullptr};
  std::vector<char> mCalibrationCache, mHistogramCache;
};

ICudaEngine* onnxToTRTModel(
    const std::string& modelFile,  // name of the onnx model
    unsigned int maxBatchSize,
    DataType dataType)  // batch size - NB must be at least as large as the
                        // batch we want to run with
{
  // create the builder
  IBuilder* builder = createInferBuilder(gLogger);

  nvonnxparser::IOnnxConfig* config = nvonnxparser::createONNXConfig();
  config->setModelFileName(modelFile.c_str());

  nvonnxparser::IONNXParser* parser = nvonnxparser::createONNXParser(*config);

  // Optional - uncomment below lines to view network layer information
  // config->setPrintLayerInfo(true);
  // parser->reportParsingInfo();

  if (!parser->parse(modelFile.c_str(), dataType)) {
    exit(EXIT_FAILURE);
  }

  if (!parser->convertToTRTNetwork()) {
    exit(EXIT_FAILURE);
  }
  nvinfer1::INetworkDefinition* network = parser->getTRTNetwork();
  auto inputDims = network->getInput(0)->getDimensions();
  INPUT_C = inputDims.d[0];
  INPUT_H = inputDims.d[1];
  INPUT_W = inputDims.d[2];
  auto outputDims = network->getOutput(0)->getDimensions();
  OUTPUT_SIZE = B * outputDims.d[0] * outputDims.d[1] * outputDims.d[2];

  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 30);
  builder->setAverageFindIterations(1);
  builder->setMinFindIterations(1);
  builder->setInt8Mode(dataType == DataType::kINT8);
  Calibrator calibrator;
  builder->setInt8Calibrator(&calibrator);

  ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);

  // we don't need the network any more, and we can destroy the parser
  network->destroy();
  parser->destroy();
  builder->destroy();

  return engine;
}

float doInference(IExecutionContext& context, float* input, int batchSize,
                  int repeat) {
  const ICudaEngine& engine = context.getEngine();
  // input and output buffer pointers that we pass to the engine - the engine
  // requires exactly IEngine::getNbBindings(), of these, but in this case we
  // know that there is exactly one input and one output.
  assert(engine.getNbBindings() == 2);
  void* buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  int inputIndex, outputIndex;
  for (int b = 0; b < engine.getNbBindings(); ++b) {
    if (engine.bindingIsInput(b))
      inputIndex = b;
    else
      outputIndex = b;
  }

  // create GPU buffers and a stream
  CHECK(cudaMalloc(&buffers[inputIndex],
                   batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex],
                   batchSize * OUTPUT_SIZE * sizeof(float)));

  CHECK(cudaMemcpy(buffers[inputIndex], input,
                   batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                   cudaMemcpyHostToDevice));
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  cudaEvent_t start, stop;
  CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
  CHECK(cudaEventCreateWithFlags(&stop, cudaEventBlockingSync));

  cudaEventRecord(start, stream);

  for (int i = 0; i < repeat; i++) {
    context.enqueue(batchSize, buffers, stream, nullptr);
  }
  cudaEventRecord(stop, stream);
  CHECK(cudaEventSynchronize(stop));
  CHECK(cudaStreamSynchronize(stream));

  float elapsedTime;
  CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  // release the stream and the buffers
  CHECK(cudaStreamDestroy(stream));
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
  return elapsedTime;
}

int main(int argc, char** argv) {
  // create a TensorRT model from the onnx model and serialize it to a stream
  if (argc < 3) {
    std::cout << argv[0] << " model_name test_int8=1" << std::endl;
    exit(EXIT_FAILURE);
  }

  MODEL_NAME = argv[1];
  if (argc >= 2 && argv[2][0] == '0') TEST_INT8 = false;

  auto dtype = TEST_INT8 ? DataType::kINT8 : DataType::kFLOAT;

  auto engine = onnxToTRTModel(MODEL_NAME, B, dtype);

  auto data = new float[B * INPUT_C * INPUT_H * INPUT_W];
  for (int i = 0; i < B * INPUT_C * INPUT_H * INPUT_W; i++) data[i] = randF();

  // deserialize the engine
  IRuntime* runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);

  // run inference
  nvtxRangeId_t id1 = nvtxRangeStartA("Start");
  auto time = doInference(*context, data, B, REPEAT);
  nvtxRangeEnd(id1);

  std::cout << "TensorRT " << MODEL_NAME << ' '
            << (TEST_INT8 ? "int8" : "fp32");
  std::cout << " Avg. Time: " << time / REPEAT << "ms" << std::endl;

  // destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();
  delete[] data;

  return 0;
}
