#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <grpcpp/grpcpp.h>

#ifdef BAZEL_BUILD
#include "examples/protos/helloworld.grpc.pb.h"
#else
#include "verifier.grpc.pb.h"
#endif

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using pai::pouw::verification::Request;
using pai::pouw::verification::Response;
using pai::pouw::verification::Verifier;
using pai::pouw::verification::Response_ReturnCode;
using pai::pouw::verification::Response_ReturnCode_GENERAL_ERROR;

class VerificationClient {
 public:
  VerificationClient(std::shared_ptr<Channel> channel)
      : stub_(Verifier::NewStub(channel)) {}

  std::pair<Response_ReturnCode, std::string> TestServer() {

    Request request;
    request.set_msg_history_id("100");
    request.set_msg_id("100");
    request.set_nonce("1");

    Response response;
    ClientContext context;

    // The actual RPC.
    Status status = stub_->Verify(&context, request, &response);

    // Act upon its status.
    if (status.ok()) {
      return std::make_pair(response.code(), response.description());
    } else {
      std::cout << status.error_code() << ": " << status.error_message()
                << std::endl;
      return std::make_pair(Response_ReturnCode_GENERAL_ERROR, "RPC failed");
    }
  }

 private:
  std::unique_ptr<Verifier::Stub> stub_;
};

int main(int argc, char** argv) {
  VerificationClient greeter(grpc::CreateChannel(
      "localhost:50011", grpc::InsecureChannelCredentials()));
  auto result = greeter.TestServer();
  std::cout << "Code:" << int(result.first) << std::endl;
  std::cout << "Description: " << result.second << std::endl;

  return 0;
}
